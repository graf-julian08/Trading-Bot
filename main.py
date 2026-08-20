"""
main.py — The Orchestrator (24/7 Trading Loop)
================================================
TITANIUM UPGRADE: Event-driven architecture with sub-200ms reaction times.

  1. Starts WebSocket connection for real-time kline + bookTicker data.
  2. Initialises local OrderManager for API-independent position tracking.
  3. Fetches candle data and computes technical indicators.
  4. Generates signals via classical strategy + 5-model ensemble.
  5. Executes via Limit Chase (market orders BANNED).
  6. Monitors positions using WS prices (sub-200ms vs old 60s).
  7. Background ML retraining every 4 hours (non-blocking).
  8. Sleeps 5s between iterations (was 60s).

Graceful shutdown on SIGINT/SIGTERM.  Top-level exception handler logs
errors and sends Telegram alerts, then retries.

Usage:
    python main.py              # Normal run
    python main.py --dry-run    # One iteration, then exit (for testing)
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import multiprocessing
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import cfg
from database import Database
from data_engine import DataEngine
from ai_model import TradePredictor, _ema, _rsi, _macd, _sma, _adx, _atr
from execution_engine import ExecutionEngine
from risk_manager import RiskManager
from notifications import TelegramNotifier
from ws_manager import ConnectionManager
from state_manager import OrderManager

# ============================================================================
# Logging Setup
# ============================================================================

def _setup_logging() -> None:
    """Configure root logger with both file and console handlers."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    # Add file handler if configured.
    if cfg.LOG_FILE:
        file_handler = logging.FileHandler(cfg.LOG_FILE, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )
    # Silence noisy third-party loggers.
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ============================================================================
# Paper Balance Tracker (Dynamic Equity for Paper Mode)
# ============================================================================

class PaperBalanceTracker:
    """
    Track simulated portfolio equity during paper trading.

    Updates the balance on every trade open (subtract cost + fee) and
    trade close (add back proceeds - fee).  This ensures:
      - get_equity() returns a realistic evolving balance
      - The kill switch can trigger in paper mode
      - Position sizing uses actual simulated equity
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self._balance = initial_balance
        self._initial = initial_balance
        logger.info(
            "Paper balance tracker initialised: %.2f USDT.", initial_balance,
        )

    @property
    def balance(self) -> float:
        """Current simulated equity."""
        return self._balance

    @property
    def initial(self) -> float:
        """Starting capital for reference."""
        return self._initial

    def on_trade_opened(
        self, cost: float, fee: float
    ) -> None:
        """
        Debit the account when a new position is opened.

        Parameters
        ----------
        cost : float
            Total trade cost (price × amount).
        fee : float
            Entry fee paid.
        """
        self._balance -= (cost + fee)
        logger.info(
            "[PAPER] Balance after trade OPEN: %.2f (debited %.2f cost + %.4f fee)",
            self._balance, cost, fee,
        )

    def on_trade_closed(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        amount: float,
        entry_cost: float,
        exit_fee: float,
    ) -> None:
        """
        Credit the account when a position is closed.

        For a BUY trade:
            - We debited entry_cost on open.
            - Now we credit back: exit_price × amount - exit_fee

        For a SELL trade:
            - We credited entry_cost on open (short proceeds).
            - Now we debit: exit_price × amount + exit_fee
        """
        if side == "buy":
            # We sold: receive proceeds minus exit fee
            proceeds = exit_price * amount - exit_fee
            self._balance += proceeds
        else:
            # We bought back to close a sell: debit the buy cost + exit fee
            buyback_cost = exit_price * amount + exit_fee
            self._balance -= buyback_cost

        logger.info(
            "[PAPER] Balance after trade CLOSE: %.2f "
            "(side=%s, entry=%.4f, exit=%.4f, amount=%.8f)",
            self._balance, side, entry_price, exit_price, amount,
        )


# Module-level singleton — initialised in run_bot().
_paper_tracker: Optional[PaperBalanceTracker] = None

# ============================================================================
# Signal Generation (Classical Strategy)
# ============================================================================

def generate_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Generate a trade signal based on the hybrid strategy:
      - Trend Following: EMA crossover (fast > slow > signal).
      - Momentum: RSI and MACD confirmation.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: close, and will have indicators added.

    Returns
    -------
    str or None
        'buy', 'sell', or None (no signal).
    """
    if len(df) < cfg.EMA_LONG + 10:
        return None

    # Use the last row for signal evaluation.
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ---- Trend: EMA alignment ----
    # Bullish: fast > slow > signal, all above long-term EMA.
    ema_fast = last.get("ema_fast")
    ema_slow = last.get("ema_slow")
    ema_signal = last.get("ema_signal")
    ema_long = last.get("ema_long")
    close = last["close"]

    if any(v is None or np.isnan(v) for v in [ema_fast, ema_slow, ema_signal, ema_long]):
        return None

    bullish_trend = (
        ema_fast > ema_slow > ema_signal
        and close > ema_long
    )
    bearish_trend = (
        ema_fast < ema_slow < ema_signal
        and close < ema_long
    )

    # ---- Momentum: RSI ----
    rsi = last.get("rsi_14", 50)
    if np.isnan(rsi):
        rsi = 50

    rsi_bullish = cfg.RSI_OVERSOLD < rsi < cfg.RSI_OVERBOUGHT  # not overbought
    rsi_bearish = cfg.RSI_OVERSOLD < rsi < cfg.RSI_OVERBOUGHT  # not oversold

    # ---- Momentum: MACD histogram turning positive/negative ----
    macd_hist = last.get("macd_histogram", 0)
    prev_macd_hist = prev.get("macd_histogram", 0)
    if np.isnan(macd_hist):
        macd_hist = 0
    if np.isnan(prev_macd_hist):
        prev_macd_hist = 0

    macd_bullish = macd_hist > 0 and prev_macd_hist <= 0  # histogram just turned positive
    macd_bearish = macd_hist < 0 and prev_macd_hist >= 0  # histogram just turned negative

    # ---- Confluence: all three must agree ----
    if bullish_trend and rsi_bullish and macd_bullish:
        return "buy"

    if bearish_trend and rsi_bearish and macd_bearish:
        return "sell"

    return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators needed for signal generation.
    This is separate from the ML feature engineering to keep concerns clean.
    Uses hand-rolled indicator functions (no external TA library).
    """
    df = df.copy()

    # EMAs
    df["ema_fast"] = _ema(df["close"], span=cfg.EMA_FAST)
    df["ema_slow"] = _ema(df["close"], span=cfg.EMA_SLOW)
    df["ema_signal"] = _ema(df["close"], span=cfg.EMA_SIGNAL)
    df["ema_long"] = _ema(df["close"], span=cfg.EMA_LONG)

    # RSI
    df["rsi_14"] = _rsi(df["close"], period=cfg.RSI_PERIOD)

    # MACD
    _, _, macd_hist = _macd(df["close"], fast=cfg.EMA_FAST, slow=cfg.EMA_SLOW, signal=9)
    df["macd_histogram"] = macd_hist

    return df


# ============================================================================
# Get Account Equity
# ============================================================================

async def get_equity(data_engine: DataEngine) -> float:
    """
    Fetch the total account equity (free + used) in the quote currency.

    For paper trading, returns the dynamic paper balance that tracks
    all simulated trade P&L, fees, and costs.
    """
    global _paper_tracker
    if cfg.PAPER_TRADE:
        if _paper_tracker is None:
            _paper_tracker = PaperBalanceTracker(initial_balance=10_000.0)
        return _paper_tracker.balance

    try:
        balance = await data_engine.fetch_balance()
        # Sum all USD-equivalent balances.
        total = balance.get("total", {})
        # Prefer USDT, then USD, then sum everything.
        for quote in ("USDT", "USD", "BUSD", "USDC"):
            if quote in total and total[quote] > 0:
                return float(total[quote])
        # Fallback: sum all.
        return sum(float(v) for v in total.values() if v)
    except Exception as e:
        logger.error("Failed to fetch equity: %s", e)
        return 0.0


# ============================================================================
# Background Retraining (Titanium: non-blocking via multiprocessing)
# ============================================================================

def _background_retrain(predictor: TradePredictor, df: pd.DataFrame, symbol: str) -> None:
    """
    Run model training in a separate process.

    This function is the target for multiprocessing.Process. It trains
    the model, saves it to disk, and exits. The main process will
    hot-reload the model file on the next iteration.

    NOTE: This runs in a SEPARATE process — it has its own memory space.
    The predictor object is pickled and sent. After training, it saves
    the model to the same file path that the main process watches.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    _log = logging.getLogger(f"retrain.{symbol}")

    try:
        _log.info("🧠 Background retrain started for %s (%d candles)", symbol, len(df))
        metrics = predictor.train(df)
        accuracy = metrics.get("balanced_accuracy", 0) if isinstance(metrics, dict) else metrics
        _log.info(
            "🧠 Background retrain complete for %s: accuracy=%.4f",
            symbol, accuracy,
        )
        # Save model synchronously (we're in a separate process)
        import asyncio
        asyncio.run(predictor.save_model())
        _log.info("🧠 Model saved for %s", symbol)
    except Exception as exc:
        _log.error("🧠 Background retrain FAILED for %s: %s", symbol, exc)


# ============================================================================
# Main Loop
# ============================================================================

async def run_bot(dry_run: bool = False) -> None:
    """
    The main async trading loop.

    Parameters
    ----------
    dry_run : bool
        If True, run a single iteration and exit (for testing).
    """
    _setup_logging()
    logger.info("=" * 60)
    logger.info("  TRADING BOT STARTING — GOD MODE v3")
    logger.info("  Mode: %s", "PAPER" if cfg.PAPER_TRADE else "LIVE")
    logger.info("  Pairs: %s", ", ".join(cfg.TRADING_PAIRS))
    logger.info("  Timeframe: %s", cfg.TIMEFRAME)
    logger.info("  Ensemble: 5-model consensus (XGB×2 + LGBM + CatBoost + RF)")
    logger.info("  Consensus threshold: %.0f%%", cfg.ENSEMBLE_CONSENSUS_THRESHOLD * 100)
    logger.info("  Reality filter: %.3f%% fee, %.3f%% slippage, %.0f× multiplier",
                cfg.TRADING_FEE_PCT * 100, cfg.SLIPPAGE_PCT * 100, cfg.REALITY_FILTER_MULTIPLIER)
    logger.info("  Circuit breaker: %d losses → %dh cooldown",
                cfg.CIRCUIT_BREAKER_LOSSES, int(cfg.CIRCUIT_BREAKER_COOLDOWN_HOURS))
    logger.info("="* 60)

    # ---- Initialise components ----
    db = Database()
    await db.initialise()

    data_engine = DataEngine()
    await data_engine.initialise()

    notifier = TelegramNotifier()
    risk = RiskManager(db=db, notifier=notifier)

    # ---- TITANIUM: WebSocket Connection Manager ----
    ws_mgr: Optional[ConnectionManager] = None
    if cfg.WS_ENABLED:
        ws_mgr = ConnectionManager(symbols=cfg.TRADING_PAIRS)
        await ws_mgr.start()
        logger.info("🔌 WebSocket manager started — sub-200ms price feed active.")

    # ---- TITANIUM: Local Order State Manager ----
    order_mgr = OrderManager()
    await order_mgr.reconcile_with_exchange(data_engine.exchange)
    logger.info("📋 OrderManager reconciled — %d open positions tracked.",
                len(order_mgr.get_open_positions()))

    # ---- Execution engine with Titanium upgrades ----
    executor = ExecutionEngine(
        data_engine=data_engine,
        db=db,
        notifier=notifier,
        ws_manager=ws_mgr,
        order_manager=order_mgr,
    )

    # Per-symbol ML models.
    predictors: Dict[str, TradePredictor] = {}
    for symbol in cfg.TRADING_PAIRS:
        predictor = TradePredictor(db=db)
        await predictor.initialise(symbol)
        predictors[symbol] = predictor

    # Send startup notification.
    await notifier.send_startup(cfg.TRADING_PAIRS, cfg.PAPER_TRADE)

    # ---- Graceful shutdown handler ----
    shutdown_event = asyncio.Event()

    def _signal_handler(sig, frame):
        logger.info("Received signal %s — initiating shutdown.", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ---- Initial equity for risk management ----
    equity = await get_equity(data_engine)
    await risk.start_of_day(equity)
    logger.info("Starting equity: %.2f", equity)

    # ---- Initial training pass ----
    for symbol in cfg.TRADING_PAIRS:
        predictor = predictors[symbol]
        if predictor.needs_retrain():
            logger.info("Running initial training for %s ...", symbol)
            try:
                hist_df = await data_engine.fetch_ohlcv_history(
                    symbol, limit=cfg.ML_TRAINING_CANDLES
                )
                if len(hist_df) >= 200:
                    metrics = predictor.train(hist_df)
                    accuracy = metrics.get("balanced_accuracy", 0) if isinstance(metrics, dict) else metrics
                    await predictor.save_model()
                    await notifier.send_model_retrained(
                        symbol, accuracy, len(hist_df)
                    )
                else:
                    logger.warning(
                        "Not enough data to train for %s (%d candles).",
                        symbol, len(hist_df),
                    )
            except Exception as e:
                logger.error("Initial training failed for %s: %s", symbol, e)

    # ---- Main loop ----
    iteration = 0
    # Alpha v2: Derivatives data cache (fetched every N seconds)
    _derivatives_cache: Dict[str, Dict] = {}
    _derivatives_last_fetch: float = 0.0
    _fear_greed_cache: Dict = {"value": 50, "classification": "Neutral"}

    while not shutdown_event.is_set():
        iteration += 1
        loop_start = asyncio.get_event_loop().time()

        try:
            # Heartbeat check: detect zombie connections.
            if not await data_engine.check_heartbeat():
                logger.warning(
                    "Heartbeat triggered reconnect — restarting iteration."
                )
                # After reconnect, skip this iteration to stabilise.
                await asyncio.sleep(5.0)
                continue

            # Refresh equity.
            equity = await get_equity(data_engine)

            # Day rollover check.
            await risk.start_of_day(equity)

            # Kill switch check.
            if await risk.check_kill_switch(equity):
                logger.warning("Kill switch active — skipping this iteration.")
                await asyncio.sleep(cfg.LOOP_INTERVAL_SECONDS)
                continue

            # ---- Process each symbol ----
            for symbol in cfg.TRADING_PAIRS:
                try:
                    # 1. Fetch live data.
                    df = await data_engine.fetch_live_ohlcv(symbol, limit=500)
                    if df.empty or len(df) < cfg.EMA_LONG + 10:
                        logger.warning(
                            "Insufficient candles for %s (%d) — skipping.",
                            symbol, len(df),
                        )
                        continue

                    # 1b. Alpha v2: Fetch derivatives data (throttled)
                    if cfg.DERIVATIVES_DATA_ENABLED:
                        now = time.time()
                        if now - _derivatives_last_fetch > cfg.DERIVATIVES_FETCH_INTERVAL_SECONDS:
                            try:
                                funding = await data_engine.fetch_funding_rate(symbol)
                                oi = await data_engine.fetch_open_interest(symbol)
                                fg = await data_engine.fetch_fear_greed_index()
                                liq = await data_engine.fetch_liquidation_data(symbol)

                                _derivatives_cache[symbol] = {
                                    **funding, **oi, **liq,
                                }
                                _fear_greed_cache = fg
                                _derivatives_last_fetch = now

                                logger.debug(
                                    "[%s] Derivatives updated: funding=%.6f, "
                                    "OI_Δ5m=%.4f, F&G=%d",
                                    symbol,
                                    funding.get("funding_rate", 0),
                                    oi.get("oi_change_5m", 0),
                                    fg.get("value", 50),
                                )
                            except Exception as deriv_err:
                                logger.warning(
                                    "Derivatives fetch failed for %s: %s",
                                    symbol, deriv_err,
                                )

                        # Inject derivatives columns into every row of df
                        deriv = _derivatives_cache.get(symbol, {})
                        fg = _fear_greed_cache
                        df["funding_rate"] = deriv.get("funding_rate", 0.0)
                        df["funding_rate_zscore"] = deriv.get("funding_rate_zscore", 0.0)
                        df["oi_change_5m"] = deriv.get("oi_change_5m", 0.0)
                        df["oi_change_1h"] = deriv.get("oi_change_1h", 0.0)
                        df["fear_greed_norm"] = fg.get("value", 50) / 100.0  # normalize 0-1
                        # Liquidation imbalance: liq_buy / (liq_buy + liq_sell)
                        liq_buy = deriv.get("liq_buy_volume", 0.0)
                        liq_sell = deriv.get("liq_sell_volume", 0.0)
                        liq_total = liq_buy + liq_sell + 1e-10
                        df["liq_imbalance"] = liq_buy / liq_total  # >0.5 = more longs liquidated
                        # Extreme flags
                        df["extreme_fear_flag"] = 1.0 if fg.get("value", 50) < 20 else 0.0
                        df["extreme_greed_flag"] = 1.0 if fg.get("value", 50) > 80 else 0.0

                    # 1c. Alpha v2: Order Flow (from WebSocket aggTrade)
                    if ws_mgr is not None and ws_mgr.is_connected:
                        flow = ws_mgr.get_trade_flow(symbol)
                        df["cvd_1m"] = flow.get("cvd_1m", 0.0)
                        df["buy_sell_ratio"] = flow.get("buy_sell_ratio", 1.0)
                        df["large_trade_ratio"] = flow.get("large_trade_ratio", 0.0)
                        df["trade_intensity"] = flow.get("trade_intensity", 0.0)

                    # 1d. Alpha v2: Cross-Asset Correlation (Tag 5)
                    # BTC is the reference asset for all alt cross-correlation.
                    if cfg.DERIVATIVES_DATA_ENABLED:
                        try:
                            if symbol != "BTC/USDT":
                                btc_df = await data_engine.fetch_live_ohlcv("BTC/USDT", limit=30)
                                if not btc_df.empty and len(btc_df) >= 10:
                                    btc_ret = btc_df["close"].pct_change(10).iloc[-1]
                                    alt_ret = df["close"].pct_change(10).iloc[-1] if len(df) >= 10 else 0.0
                                    df["btc_dominance_proxy"] = btc_ret
                                    df["cross_asset_momentum"] = btc_ret - alt_ret
                                    df["relative_strength"] = alt_ret - btc_ret
                                else:
                                    df["btc_dominance_proxy"] = 0.0
                                    df["cross_asset_momentum"] = 0.0
                                    df["relative_strength"] = 0.0
                            else:
                                # For BTC itself, relative strength = 0
                                btc_ret = df["close"].pct_change(10).iloc[-1] if len(df) >= 10 else 0.0
                                df["btc_dominance_proxy"] = btc_ret
                                df["cross_asset_momentum"] = 0.0
                                df["relative_strength"] = 0.0
                        except Exception as cross_err:
                            logger.debug("Cross-asset data unavailable: %s", cross_err)

                    # 2. Compute classical indicators.
                    df = compute_indicators(df)

                    # 3. Generate signal from classical strategy.
                    signal_direction = generate_signal(df)

                    if signal_direction is not None:
                        logger.info(
                            "[%s] Classical signal: %s", symbol, signal_direction
                        )

                        # 3a. SPOT-ONLY GUARD: Block sell signals when not holding.
                        #     On Binance Spot you CANNOT short. A "sell" signal
                        #     only makes sense if we already hold the asset.
                        if signal_direction == "sell":
                            open_positions = await db.get_open_trades(symbol)
                            buy_positions = [
                                t for t in open_positions if t["side"] == "buy"
                            ]
                            if not buy_positions:
                                logger.info(
                                    "Trade REJECTED: [%s] Reason: SELL with no position "
                                    "(Spot-only, cannot short)",
                                    symbol,
                                )
                                continue  # Skip to next symbol

                        # 4. Validate with 5-model consensus ensemble.
                        predictor = predictors[symbol]
                        ml_prob, agreement = predictor.predict_with_consensus(df)
                        logger.info(
                            "[%s] Ensemble: prob=%.4f, agreement=%.4f "
                            "(threshold=%.2f, consensus=%.2f, min_agreement=%.2f)",
                            symbol, ml_prob, agreement,
                            cfg.ML_THRESHOLD, cfg.ENSEMBLE_CONSENSUS_THRESHOLD,
                            cfg.ENSEMBLE_MIN_AGREEMENT,
                        )

                        # 4a. AGREEMENT GATE: If models contradict each other,
                        #     reject the trade. This prevents a single rogue model
                        #     from contaminating the average.
                        if agreement < cfg.ENSEMBLE_MIN_AGREEMENT:
                            logger.info(
                                "Trade REJECTED: [%s] Reason: Jury Agreement %.4f < %.2f",
                                symbol, agreement, cfg.ENSEMBLE_MIN_AGREEMENT,
                            )
                            continue  # Skip to next symbol — silence is safer

                        # ============================================================
                        # PHASE 3: REGIME DISCRIMINATOR
                        # ============================================================
                        current_price = float(df["close"].iloc[-1])

                        # ---- Mega-Trend Filter (SMA 200) ----
                        # If price < SMA 200, we are in Bear Mode.
                        # Block all long trades unless confidence is extreme (>0.90).
                        sma_200 = _sma(df["close"], 200)
                        sma_200_val = float(sma_200.iloc[-1]) if not sma_200.isna().iloc[-1] else None

                        if sma_200_val is not None:
                            if signal_direction == "buy" and current_price < sma_200_val:
                                if ml_prob < 0.90:
                                    logger.info(
                                        "Trade REJECTED: [%s] Reason: MEGA-TREND FILTER — "
                                        "Price %.2f < SMA200 %.2f (Bear Mode). "
                                        "Ensemble=%.4f < 0.90 override threshold.",
                                        symbol, current_price, sma_200_val, ml_prob,
                                    )
                                    continue  # Don't catch falling knives
                                else:
                                    logger.warning(
                                        "[%s] MEGA-TREND OVERRIDE: Price below SMA200 but "
                                        "Ensemble=%.4f >= 0.90 — allowing trade.",
                                        symbol, ml_prob,
                                    )

                        # ---- Chop Filter (ADX < 25 = no trend) ----
                        # In choppy/sideways markets, only take perfect setups.
                        adx_series = _adx(df["high"], df["low"], df["close"], period=14)
                        adx_val = float(adx_series.iloc[-1]) if not adx_series.isna().iloc[-1] else 50.0

                        chop_agreement_threshold = cfg.ENSEMBLE_MIN_AGREEMENT  # default 0.85
                        if adx_val < 25:
                            chop_agreement_threshold = 0.95  # Only perfect setups in chop
                            if agreement < chop_agreement_threshold:
                                logger.info(
                                    "Trade REJECTED: [%s] Reason: CHOP FILTER — "
                                    "ADX=%.1f < 25 (Sideways Market). "
                                    "Agreement=%.4f < 0.95 chop threshold.",
                                    symbol, adx_val, agreement,
                                )
                                continue  # Don't trade breakouts in flat markets

                        if ml_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD:
                            # 5. Check risk manager (kill switch + circuit breaker).
                            if await risk.can_open_trade(equity):
                                # 6. Calculate position sizing.
                                # (current_price already set by Regime Discriminator above)
                                sl = risk.calculate_stop_loss(
                                    current_price, signal_direction
                                )
                                tp = risk.calculate_take_profit(
                                    current_price, signal_direction
                                )

                                # 6a. Compute ATR for reality filter 
                                #     and volatility-targeted sizing.
                                atr_val = float(df["close"].rolling(cfg.VOL_TARGET_ATR_LOOKBACK).apply(
                                    lambda x: (x.max() - x.min()) / x.mean()
                                ).iloc[-1]) if len(df) >= cfg.VOL_TARGET_ATR_LOOKBACK else 0.0

                                # Median ATR for volatility targeting.
                                if "atr" in df.columns:
                                    atr_series = df["atr"].dropna()
                                else:
                                    high_low = df["high"] - df["low"]
                                    atr_series = high_low.rolling(cfg.VOL_TARGET_ATR_LOOKBACK).mean().dropna()
                                current_atr_pct = float(atr_series.iloc[-1] / current_price) if len(atr_series) > 0 else 0.0
                                median_atr_pct = float(atr_series.median() / current_price) if len(atr_series) > 0 else current_atr_pct

                                # ---- ALPHA BOOSTER 1: Kelly Lite Confidence Sizing ----
                                # Scale risk by model confidence instead of flat %.
                                confidence_risk = risk.calculate_confidence_risk(ml_prob)
                                # Temporarily use confidence-scaled risk for position sizing.
                                original_risk = cfg.MAX_RISK_PER_TRADE
                                cfg.MAX_RISK_PER_TRADE = confidence_risk
                                try:
                                    amount = risk.calculate_position_size(
                                        entry_price=current_price,
                                        stop_loss_price=sl,
                                        equity=equity,
                                    )
                                finally:
                                    cfg.MAX_RISK_PER_TRADE = original_risk

                                # 6b. Volatility-adjusted sizing.
                                if current_atr_pct > 0 and median_atr_pct > 0:
                                    amount = risk.calculate_volatility_adjusted_size(
                                        base_size=amount,
                                        current_atr_pct=current_atr_pct,
                                        median_atr_pct=median_atr_pct,
                                    )

                                if amount > 0:
                                    # 7. Execute trade (with reality filter)!
                                    trade_id = await executor.open_trade(
                                        symbol=symbol,
                                        side=signal_direction,
                                        amount=amount,
                                        entry_price=current_price,
                                        stop_loss=sl,
                                        take_profit=tp,
                                        ml_probability=ml_prob,
                                        atr_pct=current_atr_pct,
                                        adx_value=adx_val,
                                    )

                                    # 7a. Update paper balance on trade open.
                                    if trade_id is not None and cfg.PAPER_TRADE and _paper_tracker is not None:
                                        cost = current_price * amount
                                        fee = cost * cfg.TRADING_FEE_PCT
                                        _paper_tracker.on_trade_opened(cost=cost, fee=fee)

                        else:
                            logger.info(
                                "Trade REJECTED: [%s] Reason: Consensus %.4f < %.2f "
                                "(agreement=%.4f, direction=%s)",
                                symbol, ml_prob,
                                cfg.ENSEMBLE_CONSENSUS_THRESHOLD, agreement,
                                signal_direction,
                            )

                    # 8. Rolling accuracy is now tracked from ACTUAL TRADE
                    #    OUTCOMES (TP hit vs SL hit), not price direction.
                    #    See the closed_trades loop below for the real update.
                    else:
                        # ---- SCAN RESULTS: no signal found ----
                        last_price = float(df["close"].iloc[-1])
                        logger.info(
                            "Scan results: [%s] No signal — price=%.2f, "
                            "RSI=%.1f, MACD_hist=%.4f",
                            symbol, last_price,
                            float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 0.0,
                            float(df["macd_histogram"].iloc[-1]) if "macd_histogram" in df.columns else 0.0,
                        )

                except Exception as e:
                    logger.error(
                        "Error processing %s: %s\n%s",
                        symbol, e, traceback.format_exc(),
                    )
                    await notifier.send_error_alert(
                        f"Error processing {symbol}:\n{traceback.format_exc()}"
                    )

            # ---- Monitor open positions + record results for circuit breaker ----
            closed_trades: List[Dict] = await executor.monitor_open_positions()
            if closed_trades:
                for closed in closed_trades:
                    pnl = closed.get("pnl", 0.0)
                    trade_symbol = closed.get("symbol", "")
                    reason = closed.get("reason", "")
                    risk.record_trade_result(is_win=(pnl > 0))
                    logger.info(
                        "Circuit breaker updated: trade #%d %s (PnL=%.4f)",
                        closed.get("trade_id", 0),
                        "WIN" if pnl > 0 else "LOSS",
                        pnl,
                    )

                    # 8b. Update rolling accuracy with REAL trade outcome.
                    #     actual=1 means TP was hit (profitable),
                    #     actual=0 means SL was hit (loss).
                    #     This replaces the old "did price go up" check.
                    if trade_symbol in predictors:
                        actual_outcome = 1 if reason == "take_profit" else 0
                        # Use the ML confidence at entry as the predicted prob.
                        # We don't have it stored here, so use 0.5 as neutral.
                        # The key is whether actual_outcome == predicted_class.
                        entry_ml_prob = closed.get("ml_probability", 0.7)
                        predictors[trade_symbol].update_rolling_accuracy(
                            actual=actual_outcome,
                            predicted_prob=entry_ml_prob,
                        )
                        logger.info(
                            "[%s] Rolling accuracy updated: outcome=%s (%s)",
                            trade_symbol,
                            "TP_HIT" if actual_outcome == 1 else "SL_HIT",
                            reason,
                        )

                    # Update paper balance on trade close.
                    if cfg.PAPER_TRADE and _paper_tracker is not None:
                        _paper_tracker.on_trade_closed(
                            side=closed.get("side", "buy"),
                            entry_price=closed.get("entry_price", 0),
                            exit_price=closed.get("exit_price", 0),
                            amount=closed.get("amount", 0),
                            entry_cost=closed.get("entry_price", 0) * closed.get("amount", 0),
                            exit_fee=closed.get("exit_fee", 0),
                        )

            # ---- TITANIUM: 4-Hour Background Retrain ----
            for symbol in cfg.TRADING_PAIRS:
                predictor = predictors[symbol]
                if predictor.needs_retrain():
                    logger.info(
                        "🧠 BACKGROUND RETRAIN: Launching retrain for %s ...",
                        symbol,
                    )
                    try:
                        hist_df = await data_engine.fetch_ohlcv_history(
                            symbol, limit=cfg.ML_TRAINING_CANDLES
                        )
                        if len(hist_df) >= 200:
                            # Background retrain via multiprocessing
                            # (non-blocking — doesn't freeze the trading loop)
                            retrain_process = multiprocessing.Process(
                                target=_background_retrain,
                                args=(predictor, hist_df, symbol),
                                daemon=True,
                            )
                            retrain_process.start()
                            logger.info(
                                "🧠 Retrain process launched for %s (PID=%d)",
                                symbol, retrain_process.pid,
                            )
                            # We don't wait — it trains in the background.
                            # On next iteration, we check if a new model file
                            # exists and hot-reload it.
                    except Exception as e:
                        logger.error(
                            "Retrain failed for %s: %s", symbol, e,
                        )

            # ---- Cleanup old orders from state manager ----
            order_mgr.cleanup_old_orders(max_age_hours=24.0)

            # ---- TITANIUM P0 FIX: Periodic State Reconciliation ----
            # Every 5 minutes, force-sync local state with the exchange
            # via REST. This catches any dropped WebSocket messages and
            # prevents phantom/invisible positions from accumulating.
            try:
                reconciliation_age = time.time() - order_mgr._last_reconciliation_time
                if reconciliation_age > 300:  # 5 minutes
                    logger.info(
                        "🔄 PERIODIC RECONCILIATION: Last sync was %.0fs ago "
                        "— re-syncing with exchange...",
                        reconciliation_age,
                    )
                    await order_mgr.reconcile_with_exchange(data_engine.exchange)
                    logger.info(
                        "✅ Periodic reconciliation complete — %d positions tracked.",
                        len(order_mgr.get_open_positions()),
                    )
            except Exception as e:
                logger.error(
                    "⚠️ Periodic reconciliation failed (non-fatal): %s", e,
                )

            # ---- Update daily PnL ----
            try:
                open_trades = await db.get_open_trades()
                trade_count = len(open_trades)
                # Simple realised PnL: equity change from start of day.
                realised_pnl = equity - risk._starting_equity
                await risk.update_daily_pnl(equity, realised_pnl, trade_count)
            except Exception as e:
                logger.error("Failed to update daily PnL: %s", e)

            # ---- Log iteration stats ----
            elapsed = asyncio.get_event_loop().time() - loop_start
            open_count = await db.count_open_trades()
            logger.info(
                "Heartbeat: iteration=#%d, elapsed=%.2fs, "
                "equity=%.2f, open_trades=%d, mode=%s",
                iteration, elapsed, equity, open_count,
                "PAPER" if cfg.PAPER_TRADE else "LIVE",
            )

        except Exception as e:
            logger.critical(
                "CRITICAL ERROR in main loop: %s\n%s",
                e, traceback.format_exc(),
            )
            await notifier.send_error_alert(
                f"CRITICAL main loop error:\n{traceback.format_exc()}"
            )

        # ---- Dry-run exit ----
        if dry_run:
            logger.info("Dry run complete — exiting.")
            break

        # ---- Sleep ----
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=cfg.LOOP_INTERVAL_SECONDS,
            )
        except asyncio.TimeoutError:
            pass  # normal — timeout means we should loop again

    # ---- Cleanup ----
    logger.info("Shutting down...")
    if ws_mgr:
        await ws_mgr.stop()
        logger.info("🔌 WebSocket manager stopped.")
    await notifier.send_shutdown("Graceful shutdown")
    await notifier.close()
    await data_engine.close()
    await db.close()
    logger.info("Shutdown complete.")


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Parse CLI args and launch the async bot."""
    parser = argparse.ArgumentParser(
        description="Self-Learning AI Trading Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single iteration and exit (for testing).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_bot(dry_run=args.dry_run))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
