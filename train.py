"""
train.py — Advanced Offline Model Training with Realistic Backtesting
=======================================================================
Downloads historical OHLCV from Binance **public API** (free) and trains
the advanced ensemble with honest metrics and realistic backtesting.

Key features:
  - Honest metrics: Balanced Accuracy, Precision, Recall, F1, AUC-ROC
  - Class imbalance handling (scale_pos_weight, class_weight)
  - Realistic backtesting with trading fees, slippage, equity curve
  - Max drawdown, profit factor, Sharpe ratio
  - Baseline comparison (naive model) to prove real skill

Usage:
    python train.py                  # Train with 20k candles
    python train.py --candles 30000  # More data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import cfg
from database import Database
from data_engine import DataEngine
from ai_model import TradePredictor, _create_trade_target

# ============================================================================
# Logging
# ============================================================================

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)


logger = logging.getLogger("train")

# ============================================================================
# Display
# ============================================================================

def banner(text: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {text}")
    print(f"{'=' * 64}")


def print_training_report(
    symbol: str, timeframe: str, candle_count: int,
    metrics: dict, feature_importance: dict,
    training_time: float, backtest: dict,
) -> None:
    """Detailed report with honest metrics and backtest results."""
    print(f"\n  📊 Training Report: {symbol} [{timeframe}]")
    print(f"  {'─' * 56}")
    print(f"  📈 Candles used      : {candle_count:,}")
    print(f"  🧠 Model             : 4-model ensemble (3×XGB + RF)")
    print(f"  ⏱️  Training time     : {training_time:.1f}s")

    # ---- Honest Metrics ----
    print(f"\n  📏 HONEST METRICS (what matters for real trading):")
    ba = metrics.get("balanced_accuracy", 0)
    bl = metrics.get("naive_baseline", 0.75)
    skill = metrics.get("skill", 0)
    print(f"     Balanced Accuracy : {ba:.2%}  (vs {bl:.0%} naive baseline)")
    print(f"     Model Skill       : {skill:+.2%}  ({'✅ better than random' if skill > 0.01 else '⚠️ near random'})")
    print(f"     Precision         : {metrics.get('precision', 0):.2%}  (when model says 'trade', how often correct)")
    print(f"     Recall            : {metrics.get('recall', 0):.2%}  (of all good trades, how many found)")
    print(f"     F1 Score          : {metrics.get('f1', 0):.2%}  (harmonic mean of precision + recall)")
    print(f"     AUC-ROC           : {metrics.get('auc_roc', 0):.4f}  (0.5=random, 1.0=perfect)")

    # Verdict based on honest metrics
    auc = metrics.get("auc_roc", 0.5)
    if auc >= 0.65:
        v = "🟢 STRONG — Real predictive power detected"
    elif auc >= 0.58:
        v = "🟡 PROMISING — Some predictive skill, keep training"
    elif auc >= 0.52:
        v = "🟠 MARGINAL — Slight edge, needs more data"
    else:
        v = "🔴 WEAK — Model not learning meaningful patterns yet"
    print(f"     Verdict           : {v}")

    # ---- Backtest Results ----
    if backtest and backtest.get("n_trades", 0) > 0:
        print(f"\n  💰 REALISTIC BACKTEST (last 20% of data, with fees):")
        print(f"     Trades taken      : {backtest['n_trades']}")
        print(f"     Win rate          : {backtest['win_rate']:.1%}")
        print(f"     Total P&L         : {backtest['total_pnl']:+.2%}")
        print(f"     Avg trade P&L     : {backtest['avg_trade_pnl']:+.3%}")
        print(f"     Max drawdown      : {backtest['max_drawdown']:.2%}")
        print(f"     Profit factor     : {backtest['profit_factor']:.2f}x")
        print(f"     Sharpe ratio      : {backtest['sharpe']:.2f}")
        print(f"     Fees paid         : {backtest['total_fees']:.2%}")

        if backtest["total_pnl"] > 0:
            monthly = (1 + backtest["total_pnl"]) ** (720 / max(backtest["n_hours"], 1)) - 1
            print(f"     Est. monthly      : {monthly:+.1%} (projected)")
            print(f"     💰 Strategy is profitable after fees!")
        else:
            print(f"     ⚠️  Not profitable yet — keep training")

    # ---- Feature Importance ----
    if feature_importance:
        print(f"\n  🏆 Top 15 Features:")
        sorted_f = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, imp) in enumerate(sorted_f[:15], 1):
            bar = "█" * max(1, int(imp * 80))
            print(f"     {rank:2d}. {feat:22s} {imp:.4f}  {bar}")

    print()


# ============================================================================
# Realistic Backtesting (with fees, slippage, drawdown)
# ============================================================================

TRADING_FEE = 0.001      # 0.1% per trade (Binance maker/taker)
SLIPPAGE = 0.0005        # 0.05% slippage
TP_PCT = 0.01            # 1% take profit
SL_PCT = 0.005           # 0.5% stop loss
MAX_HOLD = 8             # Max hold candles


def backtest_model(predictor: TradePredictor, df: pd.DataFrame, test_frac: float = 0.2) -> dict:
    """
    Realistic backtest with trading fees, slippage, and equity tracking.

    Simulates actual trading on the last test_frac of the data:
    - Only trades when model confidence > threshold
    - Deducts 0.1% fee per trade + 0.05% slippage
    - Tracks equity curve, max drawdown, Sharpe ratio
    """
    feature_df = predictor.engineer_features(df)
    if len(feature_df) < 500:
        return {}

    split_idx = int(len(feature_df) * (1 - test_frac))
    test_df = feature_df.iloc[split_idx:].copy()

    if len(test_df) < 100:
        return {}

    # Create actual trade outcomes for test period
    test_df = test_df.copy()
    test_df["actual_target"] = _create_trade_target(
        test_df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD,
    )
    test_df = test_df.dropna(subset=["actual_target"]).copy()

    if len(test_df) < 50:
        return {}

    # Get ensemble predictions
    X = test_df[TradePredictor._FEATURE_COLS].values
    probas = []
    for model in predictor._models:
        try:
            p = model.predict_proba(X)[:, 1]
            probas.append(p)
        except Exception:
            pass

    if not probas:
        return {}

    avg_proba = np.mean(probas, axis=0)

    # Simulate trading
    threshold = 0.50
    equity = 10000.0
    equity_curve = [equity]
    peak_equity = equity
    max_dd = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_fees_paid = 0.0
    trade_returns = []

    cooldown = 0  # prevent overlapping trades

    for i in range(len(avg_proba)):
        if cooldown > 0:
            cooldown -= 1
            equity_curve.append(equity)
            continue

        if avg_proba[i] >= threshold:
            actual = test_df["actual_target"].values[i]
            # Trade cost: entry fee + exit fee + slippage
            cost_pct = 2 * TRADING_FEE + SLIPPAGE  # ~0.25% total

            if actual == 1:
                # Win: TP hit → gain tp_pct minus costs
                net = TP_PCT - cost_pct
                wins += 1
                gross_profit += TP_PCT
            else:
                # Loss: SL hit → lose sl_pct plus costs
                net = -(SL_PCT + cost_pct)
                losses += 1
                gross_loss += SL_PCT

            total_fees_paid += cost_pct
            equity *= (1 + net)
            trade_returns.append(net)
            cooldown = MAX_HOLD  # Don't trade again while in position

        equity_curve.append(equity)

        # Track max drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_dd:
            max_dd = dd

    n_trades = wins + losses
    if n_trades == 0:
        return {"n_trades": 0}

    total_pnl = (equity / 10000.0) - 1.0
    avg_pnl = np.mean(trade_returns) if trade_returns else 0
    profit_factor = gross_profit / max(gross_loss, 0.0001)

    # Sharpe ratio (annualized from hourly returns)
    if len(trade_returns) > 1 and np.std(trade_returns) > 0:
        sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(252 * 24 / MAX_HOLD)
    else:
        sharpe = 0.0

    # Hours in test period
    n_hours = len(test_df)

    return {
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n_trades,
        "total_pnl": total_pnl,
        "avg_trade_pnl": avg_pnl,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "total_fees": total_fees_paid,
        "n_hours": n_hours,
    }


# ============================================================================
# Training Pipeline
# ============================================================================

async def download_data(data_engine, symbol, timeframe, candle_count):
    print(f"\n  ⬇️  Downloading {candle_count:,} candles for {symbol} [{timeframe}] ...")
    df = await data_engine.fetch_ohlcv_history(symbol=symbol, timeframe=timeframe, limit=candle_count)
    if df.empty:
        logger.error("No data for %s", symbol)
        return df
    print(f"  📅 Range : {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"  📊 Got   : {len(df):,} candles")
    return df


async def train_symbol(symbol, timeframe, candle_count, data_engine, db) -> dict:
    """Full training pipeline for one symbol."""
    df = await download_data(data_engine, symbol, timeframe, candle_count)
    if len(df) < 600:
        logger.warning("Only %d candles — need >= 600.", len(df))
        return {}

    # ---- Enrich with historical derivatives data (closes serving skew) ----
    if cfg.DERIVATIVES_DATA_ENABLED:
        print(f"  🧬 Enriching training data with historical funding/OI ...")
        df = await data_engine.enrich_training_data(df, symbol)
        n_funding = (df.get("funding_rate", pd.Series([0])) != 0).sum()
        n_oi = (df.get("oi_change_5m", pd.Series([0])) != 0).sum()
        print(f"  ✅ Enriched: {n_funding:,} funding rate points, {n_oi:,} OI change points")
    else:
        print("  ⚠️  DERIVATIVES_DATA_ENABLED=false — training WITHOUT advanced features")

    predictor = TradePredictor(db=db)
    await predictor.initialise(symbol)

    t0 = time.time()
    metrics = predictor.train(df)
    training_time = time.time() - t0

    if not metrics or metrics.get("balanced_accuracy", 0) <= 0:
        logger.error("Training failed for %s", symbol)
        return {}

    await predictor.save_model()

    # Backtest
    backtest = backtest_model(predictor, df)

    # Feature importance
    feature_importance = predictor.get_feature_importance()

    print_training_report(
        symbol=symbol, timeframe=timeframe, candle_count=len(df),
        metrics=metrics, feature_importance=feature_importance,
        training_time=training_time, backtest=backtest,
    )

    return metrics


# ============================================================================
# History Tracking
# ============================================================================

_HISTORY_FILE = Path(__file__).parent / "training_history.json"


def _load_history():
    if _HISTORY_FILE.exists():
        try:
            return json.loads(_HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save_history(h):
    _HISTORY_FILE.write_text(json.dumps(h, indent=2, default=str))


def _record(symbol, tf, metrics, candles):
    h = _load_history()
    h.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "timeframe": tf,
        "balanced_accuracy": round(metrics.get("balanced_accuracy", 0), 4),
        "auc_roc": round(metrics.get("auc_roc", 0), 4),
        "f1": round(metrics.get("f1", 0), 4),
        "precision": round(metrics.get("precision", 0), 4),
        "skill": round(metrics.get("skill", 0), 4),
        "candles": candles,
        "version": "v3_honest",
    })
    _save_history(h)


def _print_trend():
    history = _load_history()
    v3 = [h for h in history if h.get("version") == "v3_honest"]
    if not v3:
        return

    banner("📈 ACCURACY TREND (Honest Metrics)")

    symbols = sorted(set(h["symbol"] for h in v3))
    for symbol in symbols:
        entries = [h for h in v3 if h["symbol"] == symbol][-20:]
        if not entries:
            continue
        print(f"\n  {symbol}:")
        print(f"  {'Timestamp':16s}  {'Bal.Acc':>7s}  {'AUC':>6s}  {'F1':>6s}  {'Skill':>7s}")
        print(f"  {'─' * 50}")
        for e in entries:
            ts = e["timestamp"][:16].replace("T", " ")
            ba = e.get("balanced_accuracy", 0)
            auc = e.get("auc_roc", 0)
            f1 = e.get("f1", 0)
            skill = e.get("skill", 0)
            ind = "🟢" if auc >= 0.60 else ("🟡" if auc >= 0.55 else "🔴")
            print(f"  {ts}  {ba:6.2%}  {auc:5.4f}  {f1:5.4f}  {skill:+6.2%}  {ind}")
    print()


# ============================================================================
# Main
# ============================================================================

async def run_training(pairs, timeframe, candle_count):
    _setup_logging()

    banner("🤖 AI TRADING BOT — HONEST TRAINING (v3)")
    print(f"  Architecture   : 4-model ensemble + class imbalance fix")
    print(f"  Key metric     : AUC-ROC (0.5=random, 1.0=perfect)")
    print(f"  Backtesting    : Realistic (0.1% fees, 0.05% slippage)")
    print(f"  Pairs          : {', '.join(pairs)}")
    print(f"  Timeframe      : {timeframe}")
    print(f"  Candles        : {candle_count:,}")
    print(f"  Cost           : FREE (Binance public API)")
    print(f"  Started        : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    db = Database()
    await db.initialise()
    data_engine = DataEngine()
    await data_engine.initialise()

    results = {}
    for symbol in pairs:
        try:
            metrics = await train_symbol(symbol, timeframe, candle_count, data_engine, db)
            results[symbol] = metrics
            if metrics:
                _record(symbol, timeframe, metrics, candle_count)
        except Exception as e:
            logger.error("Training failed for %s: %s", symbol, e, exc_info=True)
            results[symbol] = {}

    # Summary
    banner("📋 TRAINING SUMMARY (Honest Metrics)")
    for sym, m in results.items():
        if not m:
            print(f"  ❌ {sym:12s} : FAILED")
            continue
        auc = m.get("auc_roc", 0)
        skill = m.get("skill", 0)
        st = "🟢" if auc >= 0.60 else ("🟡" if auc >= 0.55 else "🔴")
        print(f"  {st} {sym:12s} : AUC={auc:.4f}  Bal.Acc={m.get('balanced_accuracy',0):.2%}  Skill={skill:+.2%}")

    _print_trend()

    banner("📝 NEXT STEPS")
    avg_auc = np.mean([m.get("auc_roc", 0.5) for m in results.values() if m]) if results else 0.5
    if avg_auc >= 0.60:
        print("  🟢 Model has REAL predictive skill!")
        print("  → Ready for paper trading: python main.py")
    elif avg_auc >= 0.55:
        print("  🟡 Model shows promising skill — keep training daily.")
        print("  → Run: python train.py --candles 30000")
    else:
        print("  🔴 Model needs more training to develop real skill.")
        print("  → Run daily: python train.py --candles 20000")
        print("  → More data = more market conditions = better learning.")
    print()

    await data_engine.close()
    await db.close()


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot Training (v3 — Honest)")
    parser.add_argument("--pairs", nargs="+", default=cfg.TRADING_PAIRS)
    parser.add_argument("--timeframe", default=cfg.TIMEFRAME)
    parser.add_argument("--candles", type=int, default=max(cfg.ML_TRAINING_CANDLES, 20000))
    args = parser.parse_args()

    try:
        asyncio.run(run_training(args.pairs, args.timeframe, args.candles))
    except KeyboardInterrupt:
        print("\nTraining interrupted.")


if __name__ == "__main__":
    main()
