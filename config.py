"""
config.py — Central Configuration for the AI Trading Bot
=========================================================
All settings are loaded from environment variables where sensitive (API keys),
or set as constants for strategy / risk parameters.

Usage:
    from config import cfg
    print(cfg.TRADING_PAIRS)

Security:
    Create a `.env` file alongside this module (never commit it to git):
        EXCHANGE_API_KEY=your_key
        EXCHANGE_API_SECRET=your_secret
        TELEGRAM_BOT_TOKEN=123456:ABC-DEF
        TELEGRAM_CHAT_ID=987654321
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    """Read an environment variable with a fallback default."""
    return os.getenv(key, default)


def _env_required(key: str) -> str:
    """
    Read a REQUIRED environment variable. If missing, CRASH IMMEDIATELY.
    This is the Zero-Trust contract: no secrets = no startup.
    """
    value = os.getenv(key)
    if not value:
        raise SystemExit(
            f"\n"
            f"{'=' * 60}\n"
            f"  ☠️  FATAL: Missing required secret: {key}\n"
            f"{'=' * 60}\n"
            f"  The bot REFUSES to start without this key.\n"
            f"  Fix: Add {key}=your_value to your .env file\n"
            f"  Path: {_PROJECT_ROOT / '.env'}\n"
            f"{'=' * 60}\n"
        )
    return value


def _env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    return _env(key, str(default)).lower() in ("1", "true", "yes")


def _env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float."""
    return float(_env(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer."""
    return int(_env(key, str(default)))


# ============================================================================
#  Configuration Data Class
# ============================================================================

@dataclass
class Config:
    """
    Centralised configuration object.
    Critical secrets are validated at startup — missing = crash.
    """

    # ---- Exchange Connectivity ----
    EXCHANGE_ID: str = _env("EXCHANGE_ID", "binance")
    EXCHANGE_API_KEY: str = _env("EXCHANGE_API_KEY")
    EXCHANGE_API_SECRET: str = _env("EXCHANGE_API_SECRET")
    EXCHANGE_PASSWORD: str = _env("EXCHANGE_PASSWORD", "")  # some exchanges need this

    # If True, the bot uses the exchange's sandbox / testnet endpoint.
    # Default False — public data endpoints work without keys and have more data.
    EXCHANGE_SANDBOX: bool = _env_bool("EXCHANGE_SANDBOX", False)

    # ---- Paper Trading ----
    # CRITICAL: Default is True — no real money is touched until you flip this.
    PAPER_TRADE: bool = _env_bool("PAPER_TRADE", True)

    # ---- Trading Pairs ----
    TRADING_PAIRS: List[str] = field(
        default_factory=lambda: _env(
            "TRADING_PAIRS", "BTC/USDT,ETH/USDT"
        ).split(",")
    )

    # ---- Timeframe for OHLCV candles ----
    TIMEFRAME: str = _env("TIMEFRAME", "1h")  # e.g. 1m, 5m, 15m, 1h, 4h, 1d
    CANDLE_LIMIT: int = _env_int("CANDLE_LIMIT", 500)  # candles per fetch

    # ---- Strategy Parameters ----
    # EMA periods used for trend following
    EMA_FAST: int = _env_int("EMA_FAST", 12)
    EMA_SLOW: int = _env_int("EMA_SLOW", 26)
    EMA_SIGNAL: int = _env_int("EMA_SIGNAL", 50)
    EMA_LONG: int = _env_int("EMA_LONG", 200)

    # RSI
    RSI_PERIOD: int = _env_int("RSI_PERIOD", 14)
    RSI_OVERBOUGHT: float = _env_float("RSI_OVERBOUGHT", 70.0)
    RSI_OVERSOLD: float = _env_float("RSI_OVERSOLD", 30.0)

    # MACD (uses EMA_FAST / EMA_SLOW / EMA_SIGNAL above)

    # Bollinger Bands
    BB_PERIOD: int = _env_int("BB_PERIOD", 20)
    BB_STD: float = _env_float("BB_STD", 2.0)

    # ATR for volatility
    ATR_PERIOD: int = _env_int("ATR_PERIOD", 14)

    # ---- Machine Learning ----
    # Minimum probability from the XGBoost model to accept a trade signal.
    ML_THRESHOLD: float = _env_float("ML_THRESHOLD", 0.65)

    # Ensemble consensus: minimum average probability across ALL models.
    ENSEMBLE_CONSENSUS_THRESHOLD: float = _env_float("ENSEMBLE_CONSENSUS_THRESHOLD", 0.70)

    # Minimum agreement score (1.0 - std_dev of model probabilities).
    # Ensures ALL models roughly agree. If one model strongly disagrees,
    # the trade is rejected. 0.85 means std_dev of probs must be < 0.15.
    ENSEMBLE_MIN_AGREEMENT: float = _env_float("ENSEMBLE_MIN_AGREEMENT", 0.85)

    # Number of historical candles used for training the model.
    ML_TRAINING_CANDLES: int = _env_int("ML_TRAINING_CANDLES", 10000)

    # How many hours between automatic re-training sessions (Titanium: 4h).
    ML_RETRAIN_INTERVAL_HOURS: float = _env_float("ML_RETRAIN_INTERVAL_HOURS", 4.0)

    # Minimum rolling accuracy before forced retrain.
    # 0.52 is realistic for financial time series — prevents infinite retrain loops.
    ML_MIN_ACCURACY: float = _env_float("ML_MIN_ACCURACY", 0.52)

    # Walk-forward cross-validation splits.
    ML_CV_SPLITS: int = _env_int("ML_CV_SPLITS", 5)

    # ---- Risk Management ----
    # Maximum fraction of total equity risked per trade (1% = 0.01).
    MAX_RISK_PER_TRADE: float = _env_float("MAX_RISK_PER_TRADE", 0.02)

    # Stop-loss percentage below entry price.
    STOP_LOSS_PCT: float = _env_float("STOP_LOSS_PCT", 0.02)

    # Take-profit percentage above entry price.
    TAKE_PROFIT_PCT: float = _env_float("TAKE_PROFIT_PCT", 0.04)

    # Trailing stop activation: once price is this % above entry, trailing kicks in.
    TRAILING_STOP_ACTIVATION_PCT: float = _env_float("TRAILING_STOP_ACTIVATION_PCT", 0.025)

    # Trailing stop distance (trail behind peak by this %).
    TRAILING_STOP_DISTANCE_PCT: float = _env_float("TRAILING_STOP_DISTANCE_PCT", 0.015)

    # Maximum spread (ask-bid / mid) to accept a trade.
    MAX_SPREAD_PCT: float = _env_float("MAX_SPREAD_PCT", 0.002)

    # ---- Reality Filter (Fee/Slippage Gate) ----
    TRADING_FEE_PCT: float = _env_float("TRADING_FEE_PCT", 0.001)        # 0.1%
    SLIPPAGE_PCT: float = _env_float("SLIPPAGE_PCT", 0.0005)             # 0.05%
    REALITY_FILTER_MULTIPLIER: float = _env_float("REALITY_FILTER_MULTIPLIER", 2.0)

    # Minimum notional value per order (Binance enforces ~$5-$10).
    # Set conservatively to $10 to avoid API rejections.
    MIN_NOTIONAL_USDT: float = _env_float("MIN_NOTIONAL_USDT", 10.0)

    # ---- Data Engine Heartbeat ----
    # If no data arrives for this many seconds, force a hard reconnect.
    # Detects "zombie connections" where socket is open but dead.
    DATA_HEARTBEAT_TIMEOUT_SECONDS: float = _env_float("DATA_HEARTBEAT_TIMEOUT_SECONDS", 60.0)

    # ---- Circuit Breaker ----
    CIRCUIT_BREAKER_LOSSES: int = _env_int("CIRCUIT_BREAKER_LOSSES", 3)
    CIRCUIT_BREAKER_COOLDOWN_HOURS: float = _env_float("CIRCUIT_BREAKER_COOLDOWN_HOURS", 4.0)

    # ---- Volatility Targeting ----
    VOL_TARGET_ATR_LOOKBACK: int = _env_int("VOL_TARGET_ATR_LOOKBACK", 14)
    VOL_TARGET_RISK_SCALAR: float = _env_float("VOL_TARGET_RISK_SCALAR", 1.0)

    # Maximum number of concurrent open positions.
    MAX_OPEN_POSITIONS: int = _env_int("MAX_OPEN_POSITIONS", 3)

    # ---- Kill Switch ----
    # If daily PnL drops below this fraction (e.g. -5% = -0.05), halt trading.
    DAILY_DRAWDOWN_LIMIT: float = _env_float("DAILY_DRAWDOWN_LIMIT", -0.05)

    # ---- Compounding ----
    # If True, position sizes are calculated from total equity (including profits).
    # If False, position sizes are calculated from the initial deposit only.
    COMPOUND_MODE: bool = _env_bool("COMPOUND_MODE", True)

    # ---- Loop Timing (Titanium: 5s for high-frequency monitoring) ----
    # Seconds to sleep between main loop iterations.
    LOOP_INTERVAL_SECONDS: int = _env_int("LOOP_INTERVAL_SECONDS", 5)

    # ---- WebSocket Configuration ----
    WS_ENABLED: bool = _env_bool("WS_ENABLED", True)

    # ---- Limit Chase Configuration ----
    # Maximum allowed deviation from the initial signal price during limit chase.
    # If the chase price drifts beyond this % from initial_price, the chase is
    # ABORTED immediately. Prevents buying at the top during flash pumps.
    # 0.005 = 0.5% max slippage.
    LIMIT_CHASE_MAX_SLIPPAGE_PCT: float = _env_float("LIMIT_CHASE_MAX_SLIPPAGE_PCT", 0.005)

    # ---- Alpha Booster 1: Dynamic Confidence Sizing (Kelly Lite) ----
    # Scale position risk by model confidence instead of flat MAX_RISK_PER_TRADE.
    # Formula: risk = MAX_RISK_PER_TRADE * (prob - 0.50) * SCALER
    # 55% → half size, 70% → normal, 85%+ → capped at MAX_RISK_PER_TRADE.
    CONFIDENCE_SIZING_ENABLED: bool = _env_bool("CONFIDENCE_SIZING_ENABLED", True)
    CONFIDENCE_SIZING_SCALER: float = _env_float("CONFIDENCE_SIZING_SCALER", 5.0)

    # ---- Alpha Booster 2: Sniper Entry (Micro-Structure Optimization) ----
    # Wait for a micro-dip instead of buying at the top of a 1m candle.
    # Places a discounted limit order and waits before falling back to chase.
    SNIPER_ENTRY_ENABLED: bool = _env_bool("SNIPER_ENTRY_ENABLED", True)
    SNIPER_RSI_OVERBOUGHT: float = _env_float("SNIPER_RSI_OVERBOUGHT", 70.0)
    SNIPER_DISCOUNT_PCT: float = _env_float("SNIPER_DISCOUNT_PCT", 0.002)     # 0.2%
    SNIPER_WAIT_SECONDS: float = _env_float("SNIPER_WAIT_SECONDS", 60.0)

    # ---- Alpha Booster 3: Smart Exit (Dynamic Take Profit) ----
    # Adjust TP target based on ADX (trend strength).
    # Strong trend (ADX > 50): let winners run (+50% TP).
    # Choppy market (ADX < 20): take quick profits (-20% TP).
    SMART_EXIT_ENABLED: bool = _env_bool("SMART_EXIT_ENABLED", True)
    SMART_EXIT_TREND_ADX: float = _env_float("SMART_EXIT_TREND_ADX", 50.0)
    SMART_EXIT_CHOP_ADX: float = _env_float("SMART_EXIT_CHOP_ADX", 20.0)
    SMART_EXIT_TREND_TP_BOOST: float = _env_float("SMART_EXIT_TREND_TP_BOOST", 1.5)
    SMART_EXIT_CHOP_TP_REDUCTION: float = _env_float("SMART_EXIT_CHOP_TP_REDUCTION", 0.8)

    # ---- Alpha v2: Derivatives & Sentiment Data ----
    # Enable fetching of funding rate, open interest, fear/greed index,
    # and liquidation data from free public APIs.
    DERIVATIVES_DATA_ENABLED: bool = _env_bool("DERIVATIVES_DATA_ENABLED", True)
    # Funding rate magnitude considered "extreme" (contrarian signal).
    FUNDING_RATE_EXTREME_THRESHOLD: float = _env_float("FUNDING_RATE_EXTREME_THRESHOLD", 0.001)
    # OI change in 5min considered a "spike" (liquidation risk).
    OI_SPIKE_THRESHOLD: float = _env_float("OI_SPIKE_THRESHOLD", 0.05)
    # How often to poll slow derivatives APIs (seconds).  These endpoints
    # update slower than price data, so 60s is sufficient.
    DERIVATIVES_FETCH_INTERVAL_SECONDS: float = _env_float("DERIVATIVES_FETCH_INTERVAL_SECONDS", 60.0)

    # ---- Telegram Notifications ----
    TELEGRAM_BOT_TOKEN: str = _env("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = _env("TELEGRAM_CHAT_ID")
    TELEGRAM_ENABLED: bool = _env_bool("TELEGRAM_ENABLED", False)

    # ---- Database ----
    DB_PATH: str = _env("DB_PATH", str(_PROJECT_ROOT / "trading_bot.db"))

    # ---- Logging ----
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
    LOG_FILE: str = _env("LOG_FILE", str(_PROJECT_ROOT / "trading_bot.log"))

    def __post_init__(self) -> None:
        """
        ZERO-TRUST VALIDATION.

        Called automatically after dataclass creation.
        If critical secrets are missing, the bot CRASHES HERE.
        Not after 5 minutes. Not silently. HERE.
        """
        errors: list[str] = []

        # ---- API Keys: REQUIRED when trading live ----
        if not self.PAPER_TRADE:
            if not self.EXCHANGE_API_KEY:
                errors.append("EXCHANGE_API_KEY (required for LIVE trading)")
            if not self.EXCHANGE_API_SECRET:
                errors.append("EXCHANGE_API_SECRET (required for LIVE trading)")

        # ---- Telegram: REQUIRED when enabled ----
        if self.TELEGRAM_ENABLED:
            if not self.TELEGRAM_BOT_TOKEN:
                errors.append("TELEGRAM_BOT_TOKEN (required when TELEGRAM_ENABLED=true)")
            if not self.TELEGRAM_CHAT_ID:
                errors.append("TELEGRAM_CHAT_ID (required when TELEGRAM_ENABLED=true)")

        if errors:
            msg_lines = [
                "",
                "=" * 60,
                "  ☠️  FATAL: Missing required configuration",
                "=" * 60,
            ]
            for e in errors:
                msg_lines.append(f"  ❌  {e}")
            msg_lines += [
                "",
                f"  Fix: Add these to your .env file at:",
                f"  {_PROJECT_ROOT / '.env'}",
                "=" * 60,
                "",
            ]
            raise SystemExit("\n".join(msg_lines))

        # ---- .env file existence check ----
        env_path = _PROJECT_ROOT / ".env"
        if not env_path.exists():
            raise SystemExit(
                f"\n"
                f"{'=' * 60}\n"
                f"  ☠️  FATAL: .env file not found!\n"
                f"{'=' * 60}\n"
                f"  Expected at: {env_path}\n"
                f"  The bot REFUSES to start without a .env file.\n"
                f"  Create one with at minimum:\n"
                f"    PAPER_TRADE=true\n"
                f"{'=' * 60}\n"
            )

        # ---- FAT-FINGER CLAMP: Hardcoded absolute safety bounds ----
        # These are NON-NEGOTIABLE limits.  Even if the .env says
        # MAX_RISK_PER_TRADE=100, it will be clamped to 0.05 (5%).
        # This prevents a single typo from destroying the account.
        self._clamp_risk_parameters()

    def _clamp_risk_parameters(self) -> None:
        """
        Enforce absolute bounds on every risk-critical parameter.

        These limits are HARDCODED — they cannot be overridden by .env.
        Any value outside the safe range is silently clamped and logged
        at CRITICAL level so the user knows their config was overridden.
        """
        import logging as _log
        _logger = _log.getLogger("config.safety")

        def _clamp(name: str, value: float, lo: float, hi: float) -> float:
            clamped = min(max(value, lo), hi)
            if clamped != value:
                _logger.critical(
                    "🛡️  FAT-FINGER CLAMP: %s=%.6f is outside safe range "
                    "[%.6f, %.6f] — forced to %.6f",
                    name, value, lo, hi, clamped,
                )
            return clamped

        def _clamp_int(name: str, value: int, lo: int, hi: int) -> int:
            clamped = min(max(value, lo), hi)
            if clamped != value:
                _logger.critical(
                    "🛡️  FAT-FINGER CLAMP: %s=%d is outside safe range "
                    "[%d, %d] — forced to %d",
                    name, value, lo, hi, clamped,
                )
            return clamped

        # ---- Absolute bounds (non-negotiable) ----
        self.MAX_RISK_PER_TRADE = _clamp(
            "MAX_RISK_PER_TRADE", self.MAX_RISK_PER_TRADE, 0.001, 0.05,
        )
        self.STOP_LOSS_PCT = _clamp(
            "STOP_LOSS_PCT", self.STOP_LOSS_PCT, 0.001, 0.20,
        )
        self.TAKE_PROFIT_PCT = _clamp(
            "TAKE_PROFIT_PCT", self.TAKE_PROFIT_PCT, 0.001, 0.50,
        )
        # DAILY_DRAWDOWN_LIMIT must be negative (it's a loss threshold).
        self.DAILY_DRAWDOWN_LIMIT = _clamp(
            "DAILY_DRAWDOWN_LIMIT", self.DAILY_DRAWDOWN_LIMIT, -0.50, -0.01,
        )
        self.CIRCUIT_BREAKER_LOSSES = _clamp_int(
            "CIRCUIT_BREAKER_LOSSES", self.CIRCUIT_BREAKER_LOSSES, 1, 20,
        )
        self.MAX_OPEN_POSITIONS = _clamp_int(
            "MAX_OPEN_POSITIONS", self.MAX_OPEN_POSITIONS, 1, 10,
        )

        # ---- Alpha Booster clamps ----
        self.CONFIDENCE_SIZING_SCALER = _clamp(
            "CONFIDENCE_SIZING_SCALER", self.CONFIDENCE_SIZING_SCALER, 1.0, 10.0,
        )
        self.SNIPER_DISCOUNT_PCT = _clamp(
            "SNIPER_DISCOUNT_PCT", self.SNIPER_DISCOUNT_PCT, 0.0005, 0.01,
        )
        self.SNIPER_WAIT_SECONDS = _clamp(
            "SNIPER_WAIT_SECONDS", self.SNIPER_WAIT_SECONDS, 5.0, 300.0,
        )
        self.SMART_EXIT_TREND_TP_BOOST = _clamp(
            "SMART_EXIT_TREND_TP_BOOST", self.SMART_EXIT_TREND_TP_BOOST, 1.0, 3.0,
        )
        self.SMART_EXIT_CHOP_TP_REDUCTION = _clamp(
            "SMART_EXIT_CHOP_TP_REDUCTION", self.SMART_EXIT_CHOP_TP_REDUCTION, 0.3, 1.0,
        )


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------
cfg = Config()
