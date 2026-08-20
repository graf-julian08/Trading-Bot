"""
risk_manager.py — Position Sizing, Compounding, Kill Switch & Fort Knox Mode
==============================================================================
Centralises all risk-related decisions:

  1. **Position sizing**: calculates the maximum position based on the
     risk-per-trade limit and the stop-loss distance.
  2. **Compounding**: sizes from total equity (profits included) if
     COMPOUND_MODE is enabled.
  3. **Volatility targeting**: scales position size inversely proportional
     to current ATR — high volatility = smaller positions automatically.
  4. **Daily drawdown kill switch**: halts trading if the daily PnL drops
     below the configured threshold.
  5. **Circuit breaker**: after N consecutive losses, forces a cooldown
     period to prevent algorithmic drift / revenge trading.
  6. **Max concurrent positions** guard.

Usage:
    risk = RiskManager(db=database_instance, notifier=notifier)
    await risk.start_of_day(current_equity)
    size = risk.calculate_position_size(entry_price, stop_loss_price, equity)
    size = risk.calculate_volatility_adjusted_size(size, current_atr_pct, median_atr_pct)
    risk.record_trade_result(is_win=False)
    if await risk.is_kill_switch_active():
        stop_trading()
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management gatekeeper for the trading bot (Fort Knox mode)."""

    def __init__(self, db, notifier) -> None:
        """
        Parameters
        ----------
        db : database.Database
        notifier : notifications.TelegramNotifier
        """
        self._db = db
        self._notifier = notifier

        # Cached state for the current trading day.
        self._today_str: str = ""
        self._starting_equity: float = 0.0
        self._kill_switch_active: bool = False

        # ---- Circuit Breaker state ----
        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0.0  # UNIX timestamp

    # ------------------------------------------------------------------
    # Daily Lifecycle
    # ------------------------------------------------------------------

    async def start_of_day(self, current_equity: float) -> None:
        """
        Initialise (or refresh) the daily PnL tracker.

        Call at bot startup and at midnight UTC rollover.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if today == self._today_str:
            return  # already initialised for today

        self._today_str = today
        self._starting_equity = current_equity
        self._kill_switch_active = False

        # Persist to database.
        await self._db.upsert_daily_pnl(
            date_str=today,
            starting_equity=current_equity,
        )
        logger.info(
            "Day started: %s — equity=%.2f", today, current_equity,
        )

    async def update_daily_pnl(
        self,
        current_equity: float,
        realised_pnl: float,
        trade_count: int,
    ) -> None:
        """
        Update the daily PnL record and check the kill-switch condition.
        """
        if not self._today_str:
            return

        await self._db.upsert_daily_pnl(
            date_str=self._today_str,
            starting_equity=self._starting_equity,
            ending_equity=current_equity,
            realised_pnl=realised_pnl,
            trade_count=trade_count,
            kill_switch_triggered=self._kill_switch_active,
        )

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        equity: float,
        initial_capital: Optional[float] = None,
    ) -> float:
        """
        Calculate the position size based on the risk-per-trade limit.

        The formula:
            risk_amount = equity * MAX_RISK_PER_TRADE
            risk_per_unit = |entry_price - stop_loss_price|
            position_size = risk_amount / risk_per_unit

        If COMPOUND_MODE is True, `equity` should be the current
        total equity.  If False, pass `initial_capital` as equity.

        Returns
        -------
        float
            Position size in base-asset units. Returns 0 if inputs
            are invalid.
        """
        # Use initial capital if compounding is disabled.
        if not cfg.COMPOUND_MODE and initial_capital is not None:
            equity = initial_capital

        if equity <= 0 or entry_price <= 0:
            logger.warning("Invalid equity or entry price for sizing.")
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            logger.warning("Stop loss too close to entry — cannot size position.")
            return 0.0

        risk_amount = equity * cfg.MAX_RISK_PER_TRADE
        position_size = risk_amount / risk_per_unit

        # Convert position size to cost and ensure it doesn't exceed equity.
        position_cost = position_size * entry_price
        max_cost = equity * 0.95  # never use more than 95% of equity
        if position_cost > max_cost:
            position_size = max_cost / entry_price
            logger.info(
                "Position capped at 95%% of equity: %.8f units.", position_size,
            )

        logger.info(
            "Position size: %.8f units (risk=%.2f, SL dist=%.8f, equity=%.2f).",
            position_size, risk_amount, risk_per_unit, equity,
        )
        return position_size

    # ------------------------------------------------------------------
    # Dynamic Confidence Sizing (Kelly Lite)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_confidence_risk(model_probability: float) -> float:
        """
        Kelly Lite: Scale risk % by model confidence.

        Instead of risking a flat MAX_RISK_PER_TRADE on every trade,
        this method scales position risk proportional to how confident
        the model is.

        Formula:
            edge = probability - 0.50  (range: 0.0 to 0.50)
            risk_pct = BASE_RISK × edge × SCALER

        Examples (with BASE_RISK=0.02, SCALER=5.0):
            55% confidence → edge=0.05 → risk = 0.02 × 0.05 × 5.0 = 0.005 (0.5%)
            70% confidence → edge=0.20 → risk = 0.02 × 0.20 × 5.0 = 0.020 (2.0%)
            85% confidence → edge=0.35 → risk = 0.02 × 0.35 × 5.0 = 0.035 → CAPPED at MAX_RISK_PER_TRADE

        Safety:
            - Hard cap at cfg.MAX_RISK_PER_TRADE (which itself is clamped to 5%)
            - Floor at 0.1% (never risk less than 0.1% — avoids dust orders)
            - If disabled, returns cfg.MAX_RISK_PER_TRADE (flat risk)

        Parameters
        ----------
        model_probability : float
            Ensemble model probability (0.0 to 1.0).

        Returns
        -------
        float
            Risk fraction per trade (e.g., 0.01 = 1%).
        """
        if not cfg.CONFIDENCE_SIZING_ENABLED:
            return cfg.MAX_RISK_PER_TRADE

        # Edge over random (50%)
        edge = max(model_probability - 0.50, 0.0)

        # Kelly Lite formula
        raw_risk = cfg.MAX_RISK_PER_TRADE * edge * cfg.CONFIDENCE_SIZING_SCALER

        # Floor: never risk less than 0.1% (prevents dust orders)
        risk_pct = max(raw_risk, 0.001)

        # Ceiling: hard cap at MAX_RISK_PER_TRADE (itself clamped to 5%)
        risk_pct = min(risk_pct, cfg.MAX_RISK_PER_TRADE)

        logger.info(
            "🎯 KELLY LITE: prob=%.2f%% → edge=%.2f → risk=%.3f%% "
            "(base=%.2f%%, scaler=%.1f, cap=%.2f%%)",
            model_probability * 100, edge, risk_pct * 100,
            cfg.MAX_RISK_PER_TRADE * 100, cfg.CONFIDENCE_SIZING_SCALER,
            cfg.MAX_RISK_PER_TRADE * 100,
        )
        return risk_pct

    # ------------------------------------------------------------------
    # Volatility-Targeted Position Sizing
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_volatility_adjusted_size(
        base_size: float,
        current_atr_pct: float,
        median_atr_pct: float,
    ) -> float:
        """
        Scale the position size inversely proportional to current volatility.

        Formula:
            adjusted = base_size × (median_atr / current_atr) × RISK_SCALAR

        If market is 2× more volatile than median → position halves.
        If market is 0.5× less volatile → position doubles (capped at 2× base).

        Parameters
        ----------
        base_size : float
            Raw position size from calculate_position_size().
        current_atr_pct : float
            Current ATR as fraction of price (e.g. 0.012 = 1.2%).
        median_atr_pct : float
            Median/average ATR over the lookback period.

        Returns
        -------
        float
            Volatility-adjusted position size.
        """
        if current_atr_pct <= 0 or median_atr_pct <= 0:
            return base_size

        vol_ratio = median_atr_pct / current_atr_pct
        scaled = base_size * vol_ratio * cfg.VOL_TARGET_RISK_SCALAR

        # Cap the upside at 2× base to prevent over-leverage in calm markets.
        max_size = base_size * 2.0
        adjusted = min(scaled, max_size)

        if abs(adjusted - base_size) / max(base_size, 1e-10) > 0.1:
            logger.info(
                "🎯 Volatility targeting: base=%.8f → adjusted=%.8f "
                "(ATR=%.3f%%, median=%.3f%%, ratio=%.2f).",
                base_size, adjusted,
                current_atr_pct * 100, median_atr_pct * 100, vol_ratio,
            )

        return adjusted

    # ------------------------------------------------------------------ #
    # PHASE 3: ATR-Based Volatility Targeting (Direct Formula)
    # ------------------------------------------------------------------ #
    @staticmethod
    def calculate_atr_position_size(
        equity: float,
        target_risk_pct: float,
        current_atr: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """
        Direct ATR-based position sizing.

        Formula:
            Position_Size = (Equity × Target_Risk) / (ATR × Multiplier)

        If ATR doubles (crash/pump), position size automatically halves.
        This keeps the dollar risk per trade constant.

        Parameters
        ----------
        equity : float
            Current account equity.
        target_risk_pct : float
            Target risk per trade as fraction (e.g. 0.02 = 2%).
        current_atr : float
            Current ATR value in price units.
        atr_multiplier : float
            How many ATRs to use as the risk distance (default 2.0).

        Returns
        -------
        float
            Position size in base-asset units. Returns 0 if inputs invalid.
        """
        if equity <= 0 or current_atr <= 0 or atr_multiplier <= 0:
            logger.warning(
                "Invalid inputs for ATR sizing: equity=%.2f, ATR=%.4f, mult=%.1f",
                equity, current_atr, atr_multiplier,
            )
            return 0.0

        risk_amount = equity * target_risk_pct
        risk_distance = current_atr * atr_multiplier
        position_size = risk_amount / risk_distance

        logger.info(
            "🎯 ATR Position Size: %.8f units "
            "(equity=%.2f, risk=%.2f%%, ATR=%.4f, mult=%.1f, "
            "risk_amount=%.2f, risk_dist=%.4f).",
            position_size, equity, target_risk_pct * 100,
            current_atr, atr_multiplier, risk_amount, risk_distance,
        )
        return position_size

    # ------------------------------------------------------------------
    # Stop-Loss / Take-Profit Defaults
    # ------------------------------------------------------------------

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate the default stop-loss price.

        Parameters
        ----------
        entry_price : float
        side : str
            'buy' or 'sell'.

        Returns
        -------
        float
            Stop-loss price.
        """
        if side == "buy":
            return entry_price * (1 - cfg.STOP_LOSS_PCT)
        else:
            return entry_price * (1 + cfg.STOP_LOSS_PCT)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate the default take-profit price.
        """
        if side == "buy":
            return entry_price * (1 + cfg.TAKE_PROFIT_PCT)
        else:
            return entry_price * (1 - cfg.TAKE_PROFIT_PCT)

    # ------------------------------------------------------------------
    # Circuit Breaker (consecutive loss cooldown)
    # ------------------------------------------------------------------

    def record_trade_result(self, is_win: bool) -> None:
        """
        Record a trade outcome for the circuit breaker.

        After CIRCUIT_BREAKER_LOSSES consecutive losses, the bot enters
        a cooldown period of CIRCUIT_BREAKER_COOLDOWN_HOURS.

        Parameters
        ----------
        is_win : bool
            True if the trade was profitable, False if loss.
        """
        if is_win:
            if self._consecutive_losses > 0:
                logger.info(
                    "🟢 Circuit breaker reset: winning trade after %d consecutive losses.",
                    self._consecutive_losses,
                )
            self._consecutive_losses = 0
            return

        self._consecutive_losses += 1
        logger.warning(
            "🔴 Consecutive loss #%d / %d",
            self._consecutive_losses, cfg.CIRCUIT_BREAKER_LOSSES,
        )

        if self._consecutive_losses >= cfg.CIRCUIT_BREAKER_LOSSES:
            cooldown_seconds = cfg.CIRCUIT_BREAKER_COOLDOWN_HOURS * 3600
            self._cooldown_until = time.time() + cooldown_seconds
            logger.critical(
                "🚨 CIRCUIT BREAKER TRIGGERED: %d consecutive losses. "
                "Cooldown for %.1f hours (until %s UTC).",
                self._consecutive_losses,
                cfg.CIRCUIT_BREAKER_COOLDOWN_HOURS,
                datetime.fromtimestamp(self._cooldown_until, tz=timezone.utc)
                .strftime("%Y-%m-%d %H:%M"),
            )

    @property
    def is_circuit_breaker_active(self) -> bool:
        """True if the bot is in a circuit breaker cooldown period."""
        if self._cooldown_until <= 0:
            return False
        if time.time() >= self._cooldown_until:
            # Cooldown expired — reset.
            if self._cooldown_until > 0:
                logger.info(
                    "⏰ Circuit breaker cooldown expired. Resuming trading. "
                    "Resetting consecutive loss counter."
                )
                self._cooldown_until = 0.0
                self._consecutive_losses = 0
            return False
        remaining = (self._cooldown_until - time.time()) / 60
        logger.info("🕐 Circuit breaker active — %.0f minutes remaining.", remaining)
        return True

    # ------------------------------------------------------------------
    # Kill Switch
    # ------------------------------------------------------------------

    async def check_kill_switch(self, current_equity: float) -> bool:
        """
        Check whether the daily drawdown has breached the kill-switch
        threshold.

        Returns True if trading should be halted.
        """
        if self._kill_switch_active:
            return True

        if self._starting_equity <= 0:
            return False

        daily_change_pct = (
            (current_equity - self._starting_equity) / self._starting_equity
        )

        if daily_change_pct <= cfg.DAILY_DRAWDOWN_LIMIT:
            self._kill_switch_active = True
            logger.critical(
                "🚨 KILL SWITCH TRIGGERED: daily change %.2f%% <= limit %.2f%%",
                daily_change_pct * 100, cfg.DAILY_DRAWDOWN_LIMIT * 100,
            )

            # Persist the kill switch state.
            await self._db.upsert_daily_pnl(
                date_str=self._today_str,
                starting_equity=self._starting_equity,
                ending_equity=current_equity,
                realised_pnl=current_equity - self._starting_equity,
                trade_count=0,
                kill_switch_triggered=True,
            )

            # Alert the user.
            await self._notifier.send_kill_switch_alert(
                daily_pnl_pct=daily_change_pct,
                threshold=cfg.DAILY_DRAWDOWN_LIMIT,
            )
            return True

        return False

    async def can_open_trade(self, current_equity: float) -> bool:
        """
        Comprehensive gate: checks kill switch, circuit breaker,
        and max position limits.

        Returns True if a new trade is allowed.
        """
        # Check kill switch.
        if await self.check_kill_switch(current_equity):
            logger.info("Trading halted — kill switch active.")
            return False

        # Check circuit breaker cooldown.
        if self.is_circuit_breaker_active:
            logger.info("Trading halted — circuit breaker cooldown active.")
            return False

        # Check max concurrent positions.
        open_count = await self._db.count_open_trades()
        if open_count >= cfg.MAX_OPEN_POSITIONS:
            logger.info(
                "Max positions reached (%d/%d) — skipping new trades.",
                open_count, cfg.MAX_OPEN_POSITIONS,
            )
            return False

        return True

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

