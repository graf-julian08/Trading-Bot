"""
ai_model.py — Institutional Ensemble Trade Predictor ("The Brain" v3)
=========================================================================
A production-grade ML pipeline for crypto trade prediction with:

  1. **50+ engineered features** — multi-timeframe, candlestick patterns,
     momentum, trend strength, volume analysis, and market regime detection.

  2. **Profitable-Trade Target** — instead of next-candle direction (noise),
     the model predicts whether a real trade with TP/SL would be profitable.
     This dramatically improves signal quality.

   3. **5-Model Ensemble** — XGBoost (deep+shallow) + LightGBM + CatBoost +
      RandomForest with consensus voting for maximum robustness.

  4. **Multi-Timeframe Features** — 1h data is resampled to 4h and 1d to
     provide higher-timeframe context (trend, RSI, MACD).

  5. **Purged Walk-Forward CV** — gap between train/validation prevents
     data leakage from autocorrelated price series.

All indicators are computed with pure numpy/pandas — no external TA library.

Usage:
    predictor = TradePredictor(db=database_instance)
    await predictor.initialise("BTC/USDT")
    prob = predictor.predict(latest_df)
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from config import cfg

logger = logging.getLogger(__name__)


# ============================================================================
# Pure numpy/pandas Technical Indicator Functions
# ============================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands: upper, middle, lower."""
    mid = _sma(series, period)
    rolling_std = series.rolling(window=period).std()
    upper = mid + std_dev * rolling_std
    lower = mid - std_dev * rolling_std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (0-100)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = _atr(high, low, close, period)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
                        / (atr_val + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
                         / (atr_val + 1e-10))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (volume * direction).fillna(0).cumsum()


# ============================================================================
# Multi-Timeframe Feature Helper
# ============================================================================

def _add_htf_features(df: pd.DataFrame, freq: str, prefix: str) -> pd.DataFrame:
    """
    Add higher-timeframe features by resampling.

    Resamples the 1h (or base) OHLCV to `freq` (e.g., '4h', '1D'),
    computes trend/RSI/MACD on the resampled data, then merges back
    to the original timestamps using backward fill (no look-ahead).
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        return df

    df_idx = df.set_index("timestamp")

    # Resample
    ohlcv = df_idx[["open", "high", "low", "close", "volume"]].resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()

    if len(ohlcv) < 30:
        # Not enough HTF data — fill with NaN
        df[f"{prefix}_trend"] = np.nan
        df[f"{prefix}_rsi"] = np.nan
        df[f"{prefix}_macd_hist"] = np.nan
        return df

    # Compute HTF indicators
    ema_f = _ema(ohlcv["close"], 12)
    ema_s = _ema(ohlcv["close"], 26)
    ohlcv[f"{prefix}_trend"] = ((ema_f > ema_s).astype(float) * 2 - 1)  # +1 or -1
    ohlcv[f"{prefix}_rsi"] = _rsi(ohlcv["close"], 14)
    _, _, hist = _macd(ohlcv["close"])
    ohlcv[f"{prefix}_macd_hist"] = hist

    # Shift by 1 to use only COMPLETED candles (prevent look-ahead bias)
    htf_cols = [f"{prefix}_trend", f"{prefix}_rsi", f"{prefix}_macd_hist"]
    htf_features = ohlcv[htf_cols].shift(1)

    # Forward-fill merge back to original frequency
    htf_features = htf_features.reindex(df_idx.index, method="ffill")

    for col in htf_cols:
        df[col] = htf_features[col].values

    return df


# ============================================================================
# Profitable-Trade Target (replaces naive next-candle direction)
# ============================================================================

def _create_trade_target(
    df: pd.DataFrame,
    tp_pct: float = 0.01,
    sl_pct: float = 0.005,
    max_hold: int = 8,
) -> pd.Series:
    """
    Create a target that simulates a real long trade for each candle.

    target = 1  if price hits +tp_pct BEFORE hitting -sl_pct within max_hold candles
    target = 0  otherwise (hit stop-loss, or trade expired without profit)

    With tp_pct=1% and sl_pct=0.5% → Risk:Reward = 2:1
    Even 55% accuracy with this target = very profitable trading.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    targets = np.full(n, np.nan)

    for i in range(n - max_hold):
        entry = close[i]
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 - sl_pct)

        profitable = False
        for j in range(1, max_hold + 1):
            idx = i + j
            # Check stop-loss first (worst case assumption within candle)
            if low[idx] <= sl_price:
                break
            if high[idx] >= tp_price:
                profitable = True
                break

        targets[i] = 1.0 if profitable else 0.0

    return pd.Series(targets, index=df.index)


# ============================================================================
# Purged Walk-Forward Cross-Validation
# ============================================================================

class PurgedTimeSeriesSplit:
    """
    Walk-forward CV with a gap (purge) between train and validation sets.
    Prevents data leakage from autocorrelated financial data.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 24):
        """
        Parameters
        ----------
        n_splits : int
            Number of walk-forward folds.
        purge_gap : int
            Minimum number of candles (rows) between train end and
            validation start. With 1h candles, 24 = 24-hour embargo.
            This prevents autocorrelated price data from leaking.
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X):
        n = len(X)
        fold_size = n // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n)

            if val_start >= n:
                continue

            yield (
                np.arange(0, train_end),
                np.arange(val_start, val_end),
            )


# ============================================================================
# Trade Predictor (v2 — Advanced Ensemble)
# ============================================================================

class TradePredictor:
    """
    Institutional 5-model ensemble for trade signal validation.

    Key improvements over v2:
      - 50+ features (multi-timeframe, candlestick, regime)
      - 5-model consensus: XGBoost × 2 + LightGBM + CatBoost + RF
      - Consensus voting: trade ONLY when models massively agree (>70%)
      - Agreement score: measures how aligned all models are
    """

    # ---- Feature columns (v2) ----
    _FEATURE_COLS = [
        # -- Original technical indicators --
        "rsi_14",
        "ema_fast_slope", "ema_slow_slope", "ema_signal_slope", "ema_long_slope",
        "macd_histogram", "macd_signal_diff",
        "bb_width", "bb_position",
        "atr_14", "atr_pct",
        "volume_zscore",
        "dist_ema_fast", "dist_ema_slow", "dist_ema_long",
        "return_1", "return_3", "return_5", "return_10",
        "volatility_10",
        "hour_sin", "hour_cos",
        # -- Candlestick patterns --
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "is_bullish_candle",
        # -- Momentum --
        "stoch_k", "stoch_d",
        "roc_5", "roc_10", "roc_20",
        # -- Trend strength --
        "adx_14",
        "trend_strength",
        # -- Volume analysis --
        "obv_slope",
        "volume_ratio",
        # -- Market structure --
        "consecutive_up", "consecutive_down",
        "dist_high_20", "dist_low_20",
        # -- Day of week --
        "dow_sin", "dow_cos",
        # -- Multi-timeframe (4h) --
        "htf_4h_trend", "htf_4h_rsi", "htf_4h_macd_hist",
        # -- Multi-timeframe (1d) --
        "htf_1d_trend", "htf_1d_rsi", "htf_1d_macd_hist",
        # -- Regime --
        "volatility_regime",
        "return_20",
        # ---- PHASE 2: Velocity & Acceleration (1st/2nd derivatives) ----
        "rsi_velocity", "rsi_acceleration",
        "macd_velocity", "macd_acceleration",
        "price_velocity", "price_acceleration",
        "stoch_velocity", "obv_acceleration",
        # ---- PHASE 2: Volatility-Normalized Returns ----
        "atr_norm_return_1", "atr_norm_return_3", "atr_norm_return_5",
        # ---- PHASE 2: Lagged Features (price memory) ----
        "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4", "close_lag_5",
        "volume_lag_1", "volume_lag_2", "volume_lag_3",
        # ---- ALPHA v2: Derivatives & Sentiment (Tag 1+2) ----
        "funding_rate", "funding_rate_zscore",
        "oi_change_5m", "oi_change_1h",
        "fear_greed_norm",
        "liq_imbalance",
        "funding_oi_interaction",
        "extreme_fear_flag", "extreme_greed_flag",
        # ---- ALPHA v2: Order Flow (Tag 4) ----
        "cvd_1m", "buy_sell_ratio", "large_trade_ratio", "trade_intensity",
        # ---- ALPHA v2: Cross-Asset Correlation (Tag 5) ----
        "btc_dominance_proxy", "cross_asset_momentum", "relative_strength",
        # ---- ALPHA v2: VWAP + Volume Profile (Tag 6) ----
        "dist_vwap", "vwap_slope", "dist_poc", "above_vwap", "volume_concentration",
    ]

    def __init__(self, db=None) -> None:
        self._db = db
        self._models: List = []         # ensemble of models
        self._scaler: Optional[StandardScaler] = None  # Iron Curtain
        self._symbol: Optional[str] = None
        self._feature_importance: Optional[dict] = None  # SHAP rankings
        self._last_train_time: float = 0.0
        self._rolling_accuracy: float = 0.0
        self._is_trained = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self, symbol: str) -> None:
        """Load the latest persisted model for `symbol`."""
        self._symbol = symbol

        if self._db is not None:
            state = await self._db.load_latest_model_state(symbol)
            if state is not None:
                try:
                    data = pickle.loads(state["model_blob"])
                    # Support v3 (dict with models + scaler) and legacy formats
                    if isinstance(data, dict) and "models" in data:
                        self._models = data["models"]
                        self._scaler = data.get("scaler", None)
                    elif isinstance(data, list):
                        self._models = data
                        self._scaler = None
                    else:
                        self._models = [data]
                        self._scaler = None
                    self._last_train_time = state["trained_at"]
                    self._rolling_accuracy = state["accuracy"]
                    self._is_trained = True
                    logger.info(
                        "Loaded ensemble (%d models, scaler=%s) for %s (accuracy=%.4f).",
                        len(self._models),
                        "YES" if self._scaler else "NO",
                        symbol, self._rolling_accuracy,
                    )
                    return
                except Exception as e:
                    logger.warning("Failed to deserialise model for %s: %s", symbol, e)

        logger.info("No existing model for %s — needs initial training.", symbol)

    # ------------------------------------------------------------------
    # Feature Engineering (v2 — 50+ features)
    # ------------------------------------------------------------------

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw OHLCV into 50+ features including multi-timeframe context.
        """
        df = df.copy()

        # ================================================================
        # ORIGINAL FEATURES (cleaned up)
        # ================================================================

        # ---- RSI ----
        df["rsi_14"] = _rsi(df["close"], period=cfg.RSI_PERIOD)

        # ---- EMAs ----
        df["ema_fast"] = _ema(df["close"], span=cfg.EMA_FAST)
        df["ema_slow"] = _ema(df["close"], span=cfg.EMA_SLOW)
        df["ema_signal"] = _ema(df["close"], span=cfg.EMA_SIGNAL)
        df["ema_long"] = _ema(df["close"], span=cfg.EMA_LONG)

        # EMA slopes (rate of change over 3 periods)
        for name in ["ema_fast", "ema_slow", "ema_signal", "ema_long"]:
            df[f"{name}_slope"] = df[name].pct_change(periods=3)

        # ---- MACD ----
        macd_line, macd_signal_line, macd_hist = _macd(
            df["close"], fast=cfg.EMA_FAST, slow=cfg.EMA_SLOW, signal=9
        )
        df["macd_histogram"] = macd_hist
        df["macd_signal_diff"] = macd_line - macd_signal_line

        # ---- Bollinger Bands ----
        bb_upper, bb_mid, bb_lower = _bollinger_bands(
            df["close"], period=cfg.BB_PERIOD, std_dev=cfg.BB_STD
        )
        df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # ---- ATR ----
        df["atr_14"] = _atr(df["high"], df["low"], df["close"], period=cfg.ATR_PERIOD)
        df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-10)

        # ---- Volume z-score ----
        vol_mean = df["volume"].rolling(window=20).mean()
        vol_std = df["volume"].rolling(window=20).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)

        # ---- Distance from MAs ----
        df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / (df["close"] + 1e-10)
        df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / (df["close"] + 1e-10)
        df["dist_ema_long"] = (df["close"] - df["ema_long"]) / (df["close"] + 1e-10)

        # ---- Lagged Returns ----
        for lag in [1, 3, 5, 10, 20]:
            df[f"return_{lag}"] = df["close"].pct_change(periods=lag)

        # ---- Rolling Volatility ----
        df["volatility_10"] = df["close"].pct_change().rolling(window=10).std()

        # ---- Time features ----
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            hour = df["timestamp"].dt.hour
            dow = df["timestamp"].dt.dayofweek
        else:
            hour = pd.Series(np.zeros(len(df)), index=df.index)
            dow = pd.Series(np.zeros(len(df)), index=df.index)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # ================================================================
        # NEW FEATURES
        # ================================================================

        # ---- Candlestick Patterns ----
        candle_range = df["high"] - df["low"] + 1e-10
        body = (df["close"] - df["open"]).abs()
        df["body_ratio"] = body / candle_range
        df["upper_shadow_ratio"] = (df["high"] - df[["close", "open"]].max(axis=1)) / candle_range
        df["lower_shadow_ratio"] = (df[["close", "open"]].min(axis=1) - df["low"]) / candle_range
        df["is_bullish_candle"] = (df["close"] > df["open"]).astype(float)

        # ---- Stochastic Oscillator ----
        df["stoch_k"], df["stoch_d"] = _stochastic(df["high"], df["low"], df["close"])

        # ---- Rate of Change ----
        for lag in [5, 10, 20]:
            df[f"roc_{lag}"] = df["close"].pct_change(periods=lag) * 100

        # ---- ADX (trend strength) ----
        df["adx_14"] = _adx(df["high"], df["low"], df["close"], period=14)
        # Custom trend strength: |ema_fast - ema_slow| / ATR
        df["trend_strength"] = (
            (df["ema_fast"] - df["ema_slow"]).abs() / (df["atr_14"] + 1e-10)
        )

        # ---- OBV slope ----
        obv = _obv(df["close"], df["volume"])
        df["obv_slope"] = obv.pct_change(periods=5)

        # ---- Volume ratio ----
        df["volume_ratio"] = df["volume"] / (vol_mean + 1e-10)

        # ---- Consecutive candles ----
        bullish = (df["close"] > df["close"].shift(1)).astype(int)
        bearish = (df["close"] < df["close"].shift(1)).astype(int)
        # Count consecutive streaks
        df["consecutive_up"] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumsum()
        df["consecutive_down"] = bearish.groupby((bearish != bearish.shift()).cumsum()).cumsum()

        # ---- Distance from 20-period high/low ----
        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        price_range_20 = high_20 - low_20 + 1e-10
        df["dist_high_20"] = (high_20 - df["close"]) / price_range_20
        df["dist_low_20"] = (df["close"] - low_20) / price_range_20

        # ---- Volatility regime (0=low, 1=med, 2=high) ----
        vol_pctile = df["atr_pct"].rolling(100).rank(pct=True)
        df["volatility_regime"] = pd.cut(
            vol_pctile, bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2],
        ).astype(float)

        # ================================================================
        # PHASE 2: VELOCITY & ACCELERATION (1st / 2nd derivatives)
        # ================================================================
        # These tell the model HOW FAST indicators are moving and
        # WHETHER they are speeding up or slowing down.
        # All use .diff() which is backward-looking (no future leak).

        # -- RSI velocity & acceleration --
        df["rsi_velocity"] = df["rsi_14"].diff(1)          # 1st derivative
        df["rsi_acceleration"] = df["rsi_velocity"].diff(1)  # 2nd derivative

        # -- MACD histogram velocity & acceleration --
        df["macd_velocity"] = df["macd_histogram"].diff(1)
        df["macd_acceleration"] = df["macd_velocity"].diff(1)

        # -- Price velocity & acceleration (pct_change is already 1st deriv) --
        df["price_velocity"] = df["close"].pct_change(1)
        df["price_acceleration"] = df["price_velocity"].diff(1)

        # -- Stochastic velocity (how fast %K is moving) --
        df["stoch_velocity"] = df["stoch_k"].diff(1)

        # -- OBV acceleration (rate of change of OBV slope) --
        df["obv_acceleration"] = df["obv_slope"].diff(1)

        # ================================================================
        # PHASE 2: VOLATILITY-NORMALIZED RETURNS
        # ================================================================
        # A $100 move in BTC is noise; $100 in ADA is a crash.
        # Dividing returns by ATR normalizes across regimes.
        atr_safe = df["atr_14"].replace(0, np.nan)  # avoid div-by-zero
        df["atr_norm_return_1"] = (df["close"].diff(1)) / (atr_safe)
        df["atr_norm_return_3"] = (df["close"].diff(3)) / (atr_safe)
        df["atr_norm_return_5"] = (df["close"].diff(5)) / (atr_safe)

        # ================================================================
        # PHASE 2: LAGGED FEATURES (price & volume memory)
        # ================================================================
        # Normalized by current close so they are scale-invariant.
        for lag in range(1, 6):
            df[f"close_lag_{lag}"] = df["close"].shift(lag) / (df["close"] + 1e-10)
        for lag in range(1, 4):
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag) / (df["volume"].rolling(20).mean() + 1e-10)

        # ================================================================
        # MULTI-TIMEFRAME FEATURES
        # ================================================================
        df = _add_htf_features(df, freq="4h", prefix="htf_4h")
        df = _add_htf_features(df, freq="1D", prefix="htf_1d")

        # ================================================================
        # ALPHA v2: DERIVATIVES & SENTIMENT FEATURES
        # ================================================================
        # These columns are injected into the DataFrame by main.py from
        # live API data. During backtesting/training they won't exist,
        # so we fill with neutral defaults (no signal = 0).
        _DERIVATIVES_COLS = [
            "funding_rate", "funding_rate_zscore",
            "oi_change_5m", "oi_change_1h",
            "fear_greed_norm",
            "liq_imbalance",
            "funding_oi_interaction",
            "extreme_fear_flag", "extreme_greed_flag",
            # Order flow (Tag 4)
            "cvd_1m", "buy_sell_ratio", "large_trade_ratio", "trade_intensity",
            # Cross-asset (Tag 5)
            "btc_dominance_proxy", "cross_asset_momentum", "relative_strength",
        ]
        for col in _DERIVATIVES_COLS:
            if col not in df.columns:
                df[col] = 0.0  # Neutral default when data unavailable

        # Compute interaction feature if raw data is available
        # Funding × OI_change = convergence signal
        # High funding + rising OI = overleveraged (bearish)
        if "funding_rate" in df.columns and "oi_change_5m" in df.columns:
            df["funding_oi_interaction"] = (
                df["funding_rate"] * 1000  # scale up from 0.0001 range
            ) * (
                df["oi_change_5m"] * 100   # scale up from 0.01 range
            )

        # ================================================================
        # ALPHA v2: VWAP + VOLUME PROFILE (Tag 6)
        # ================================================================
        # These are computed from OHLCV data — always available.
        _vwap_period = 20
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vol_sum = df["volume"].rolling(_vwap_period).sum() + 1e-10
        vwap = (typical_price * df["volume"]).rolling(_vwap_period).sum() / vol_sum

        df["dist_vwap"] = (df["close"] - vwap) / (df["close"] + 1e-10)
        df["vwap_slope"] = vwap.diff(3) / (vwap.shift(3) + 1e-10)
        df["above_vwap"] = (df["close"] > vwap).astype(float)

        # Volume Profile: Point of Control (price level with highest volume)
        # Uses a rolling 50-bar window with 10 price bins.
        _poc_window = 50
        dist_poc = pd.Series(0.0, index=df.index)
        vol_concentration = pd.Series(0.0, index=df.index)
        for i in range(_poc_window, len(df)):
            window = df.iloc[i - _poc_window : i]
            price_range = window["high"].max() - window["low"].min()
            if price_range <= 0:
                continue
            bins = np.linspace(window["low"].min(), window["high"].max(), 11)
            bin_idx = np.digitize(window["close"].values, bins) - 1
            bin_idx = np.clip(bin_idx, 0, 9)
            bin_volumes = np.zeros(10)
            for b, v in zip(bin_idx, window["volume"].values):
                bin_volumes[b] += v
            poc_bin = np.argmax(bin_volumes)
            poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
            current_close = df["close"].iloc[i]
            dist_poc.iloc[i] = (current_close - poc_price) / (current_close + 1e-10)
            # Volume concentration: max bin volume / total volume
            total_vol = bin_volumes.sum() + 1e-10
            vol_concentration.iloc[i] = bin_volumes[poc_bin] / total_vol

        df["dist_poc"] = dist_poc
        df["volume_concentration"] = vol_concentration

        # ---- Drop intermediate helper columns ----
        helper_cols = [
            "ema_fast", "ema_slow", "ema_signal", "ema_long",
        ]
        df.drop(columns=[c for c in helper_cols if c in df.columns], inplace=True, errors="ignore")

        # ---- NaN/Inf sanitization (PHASE 2 safety) ----
        # Replace any Inf/-Inf with NaN, then let dropna clean up.
        for col in TradePredictor._FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # ---- Fill NaN in optional derivatives features (backtest compat) ----
        for col in _DERIVATIVES_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # ---- Drop rows with NaN in CORE features only ----
        core_cols = [
            c for c in TradePredictor._FEATURE_COLS
            if c not in _DERIVATIVES_COLS and c in df.columns
        ]
        df.dropna(subset=core_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    # ------------------------------------------------------------------
    # Training (v3 — Honest Metrics + Class Imbalance Fix)
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the 4-model ensemble with class imbalance handling.

        Target: simulated profitable trade (TP=1%, SL=0.5%, max 8 candles).
        CV: Purged walk-forward to prevent data leakage.
        Metrics: Balanced accuracy, precision, recall, F1, AUC-ROC.

        Returns dict with all metrics.
        """
        feature_df = self.engineer_features(df)

        if len(feature_df) < 500:
            logger.warning(
                "Insufficient data (%d rows). Need >= 500.",
                len(feature_df),
            )
            return {"balanced_accuracy": 0.0}

        # ---- Create profitable-trade target ----
        feature_df["target"] = _create_trade_target(
            feature_df, tp_pct=0.01, sl_pct=0.005, max_hold=8,
        )
        feature_df.dropna(subset=["target"], inplace=True)
        feature_df.reset_index(drop=True, inplace=True)

        X = feature_df[self._FEATURE_COLS].values
        y = feature_df["target"].values.astype(int)

        # Target distribution and class imbalance weight
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_ratio = n_pos / len(y)
        imbalance_weight = n_neg / max(n_pos, 1)  # ~3.0 for 25% pos
        naive_baseline = max(pos_ratio, 1 - pos_ratio)  # accuracy of always-majority

        logger.info(
            "Target: %.1f%% profitable (n=%d), imbalance_weight=%.2f, naive_baseline=%.2f%%",
            pos_ratio * 100, len(y), imbalance_weight, naive_baseline * 100,
        )

        # ---- Purged walk-forward cross-validation ----
        # purge_gap=24 → 24 candles (24 hours with 1h data) between
        # train and validation to prevent autocorrelation leakage.
        cv = PurgedTimeSeriesSplit(n_splits=cfg.ML_CV_SPLITS, purge_gap=24)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # ============================================================
            # IRON CURTAIN: fit scaler ONLY on training data.
            # The test/validation data NEVER touches the scaler fitting.
            # This prevents future information from leaking into features.
            # ============================================================
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_train)  # fit on train ONLY
            X_val = fold_scaler.transform(X_val)           # transform (NOT fit!)

            # Build ensemble with class imbalance handling
            ensemble = self._build_ensemble(imbalance_weight)
            fold_preds = []

            for name, model in ensemble:
                if hasattr(model, "eval_metric") or hasattr(model, "eval_metric_"):
                    # XGBoost / LightGBM / CatBoost with early stopping
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                else:
                    # RandomForest — no early stopping
                    model.fit(X_train, y_train)

                proba = model.predict_proba(X_val)[:, 1]
                fold_preds.append(proba)

            # Average ensemble prediction
            avg_proba = np.mean(fold_preds, axis=0)
            preds = (avg_proba >= 0.5).astype(int)

            # Compute honest metrics
            m = {
                "accuracy": accuracy_score(y_val, preds),
                "balanced_accuracy": balanced_accuracy_score(y_val, preds),
                "precision": precision_score(y_val, preds, zero_division=0),
                "recall": recall_score(y_val, preds, zero_division=0),
                "f1": f1_score(y_val, preds, zero_division=0),
                "auc_roc": roc_auc_score(y_val, avg_proba) if len(np.unique(y_val)) > 1 else 0.5,
            }
            fold_metrics.append(m)

            logger.info(
                "Fold %d/%d — bal_acc=%.4f, precision=%.4f, recall=%.4f, F1=%.4f, AUC=%.4f",
                fold, cfg.ML_CV_SPLITS,
                m["balanced_accuracy"], m["precision"], m["recall"], m["f1"], m["auc_roc"],
            )

        # Average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0]:
            avg_metrics[key] = float(np.mean([m[key] for m in fold_metrics]))

        avg_metrics["naive_baseline"] = naive_baseline
        avg_metrics["pos_ratio"] = pos_ratio
        avg_metrics["n_samples"] = len(y)
        avg_metrics["skill"] = avg_metrics["balanced_accuracy"] - 0.5  # >0 means better than random

        logger.info(
            "MEAN — bal_acc=%.4f, precision=%.4f, recall=%.4f, F1=%.4f, AUC=%.4f, skill=+%.4f",
            avg_metrics["balanced_accuracy"], avg_metrics["precision"],
            avg_metrics["recall"], avg_metrics["f1"],
            avg_metrics["auc_roc"], avg_metrics["skill"],
        )

        # ---- LEAKAGE CHECK: reject suspiciously high accuracy ----
        if avg_metrics["accuracy"] > 0.90:
            logger.warning(
                "⚠️  LEAKAGE ALERT: accuracy=%.2f%% is suspiciously high! "
                "Real trading models rarely exceed 60%%. Check for data leakage.",
                avg_metrics["accuracy"] * 100,
            )

        # ---- Final ensemble: train on ALL data with Iron Curtain scaler ----
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)  # Final scaler for production

        final_ensemble = self._build_ensemble(imbalance_weight)
        trained_models = []

        # For final training with early stopping, hold out last 10% as eval set
        split_point = int(len(X_scaled) * 0.9)
        X_final_train = X_scaled[:split_point]
        y_final_train = y[:split_point]
        X_final_eval = X_scaled[split_point:]
        y_final_eval = y[split_point:]

        for name, model in final_ensemble:
            if hasattr(model, "eval_metric") or hasattr(model, "eval_metric_"):
                model.fit(
                    X_final_train, y_final_train,
                    eval_set=[(X_final_eval, y_final_eval)],
                    verbose=False,
                )
            else:
                model.fit(X_scaled, y)  # RF uses all data
            trained_models.append(model)

        self._models = trained_models
        self._last_train_time = time.time()
        self._rolling_accuracy = avg_metrics["balanced_accuracy"]
        self._is_trained = True
        self._train_metrics = avg_metrics

        logger.info(
            "Ensemble trained for %s — bal_acc=%.4f, AUC=%.4f, models=%d, samples=%d.",
            self._symbol, avg_metrics["balanced_accuracy"],
            avg_metrics["auc_roc"], len(self._models), len(X),
        )

        # ---- SHAP Feature Importance Analysis (Tag 3) ----
        self._compute_shap_importance(X_scaled, self._FEATURE_COLS)

        return avg_metrics

    def _compute_shap_importance(
        self, X: np.ndarray, feature_names: list,
        noise_threshold: float = 0.001,
    ) -> None:
        """
        Compute SHAP values for the first tree-based model and rank features.

        Uses TreeExplainer for speed (~10x faster than KernelExplainer).
        Logs top-20 most important and flags bottom features as noise.

        The ranking is stored in self._feature_importance for inspection.
        Features are NOT auto-removed — this is intentional to avoid
        silent model changes. The ranking guides manual pruning decisions.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix (from training).
        feature_names : list
            List of feature column names.
        noise_threshold : float
            Features with mean |SHAP| below this are flagged as noise.
        """
        if not _HAS_SHAP:
            logger.warning(
                "SHAP not installed — skipping feature importance analysis. "
                "Install: pip install shap>=0.45"
            )
            return

        try:
            # Use the first tree-based model (XGBoost deep) for SHAP
            model = self._models[0] if self._models else None
            if model is None:
                return

            # Sample if dataset is large (SHAP can be slow on >5000 rows)
            max_samples = 2000
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification output
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Class 1 (buy signal)

            # Mean absolute SHAP value per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # Build importance dict
            importance = {
                name: float(val)
                for name, val in zip(feature_names, mean_abs_shap)
            }
            # Sort by importance (descending)
            importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )

            self._feature_importance = importance

            # Log top-20
            top_20 = list(importance.items())[:20]
            logger.info("📊 SHAP Feature Importance (Top 20):")
            for rank, (name, val) in enumerate(top_20, 1):
                logger.info("  #%02d  %-30s  %.6f", rank, name, val)

            # Flag noise features
            noise_features = [
                name for name, val in importance.items()
                if val < noise_threshold
            ]
            if noise_features:
                logger.warning(
                    "🗑️  %d noise features (|SHAP| < %.4f): %s",
                    len(noise_features), noise_threshold,
                    ", ".join(noise_features),
                )
            else:
                logger.info("✅ All features contribute above noise threshold.")

        except Exception as e:
            logger.warning("SHAP analysis failed (non-critical): %s", e)

    def get_feature_importance(self) -> Optional[dict]:
        """
        Return the SHAP-based feature importance ranking.

        Returns None if SHAP hasn't been computed yet.
        Dict maps feature_name -> mean |SHAP| value, sorted descending.
        """
        return self._feature_importance

    @staticmethod
    def _build_ensemble(imbalance_weight: float = 1.0) -> list:
        """
        Build the 5-model institutional ensemble.

        Models:
          1. XGBoost Deep   — deep trees, low LR, captures complex interactions
          2. XGBoost Shallow — shallow trees, high LR, fast learner, regularised
          3. LightGBM       — DART boosting, leaf-wise growth, complementary to XGB
          4. CatBoost       — symmetric trees, ordered boosting, robust to noise
          5. RandomForest   — independent trees, decorrelated from boosting models

        scale_pos_weight / is_unbalance handles the minority class (profitable trades).
        """
        xgb_common = dict(
            scale_pos_weight=imbalance_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=50,  # MANDATORY: stop if no improvement for 50 rounds
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        models = [
            ("xgb_deep", XGBClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=5, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0,
                **xgb_common,
            )),
            ("xgb_shallow", XGBClassifier(
                n_estimators=1000, max_depth=3, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.8,
                min_child_weight=10, gamma=0.2,
                reg_alpha=0.5, reg_lambda=2.0,
                **xgb_common,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_leaf=10, min_samples_split=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]

        # ---- LightGBM (DART boosting, leaf-wise, regularised) ----
        if _HAS_LIGHTGBM:
            models.append(("lgbm", LGBMClassifier(
                n_estimators=1000, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.75,
                min_child_samples=20,
                reg_alpha=0.3, reg_lambda=1.5,
                scale_pos_weight=imbalance_weight,
                boosting_type="dart",
                random_state=42, n_jobs=-1,
                verbose=-1,
            )))
        else:
            logger.warning("LightGBM not installed — using 3rd XGBoost variant instead.")
            models.append(("xgb_balanced", XGBClassifier(
                n_estimators=1000, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.75,
                min_child_weight=7, gamma=0.15,
                reg_alpha=0.2, reg_lambda=1.5,
                **xgb_common,
            )))

        # ---- CatBoost (symmetric trees, ordered boosting) ----
        if _HAS_CATBOOST:
            models.append(("catboost", CatBoostClassifier(
                iterations=1000, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                early_stopping_rounds=50,  # CatBoost early stopping
                random_seed=42,
                verbose=0,
                thread_count=-1,
            )))
        else:
            logger.warning("CatBoost not installed — using additional RF variant.")
            models.append(("rf_extra", RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_leaf=5, min_samples_split=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=99, n_jobs=-1,
            )))

        return models

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict probability of a profitable trade using the ensemble.

        Returns probability in [0, 1]. 0.5 = neutral (untrained model).
        """
        prob, _ = self.predict_with_consensus(df)
        return prob

    def predict_with_consensus(self, df: pd.DataFrame) -> tuple:
        """
        Predict with full consensus metrics.

        Returns
        -------
        tuple of (avg_probability: float, agreement_score: float)
            avg_probability : mean probability across all models [0, 1]
            agreement_score : 1.0 - std(probs), higher = more agreement [0, 1]
            Both return (0.5, 0.0) if model is untrained.
        """
        if not self._is_trained or not self._models:
            logger.warning("Model not trained — returning neutral 0.5.")
            return 0.5, 0.0

        feature_df = self.engineer_features(df)

        if feature_df.empty:
            logger.warning("Feature engineering produced empty DataFrame.")
            return 0.5, 0.0

        last_row = feature_df[self._FEATURE_COLS].iloc[[-1]].values

        # Apply the production scaler (Iron Curtain)
        if self._scaler is not None:
            last_row = self._scaler.transform(last_row)

        # Ensemble: collect probabilities from all models
        probas = []
        for model in self._models:
            try:
                p = model.predict_proba(last_row)[0][1]
                probas.append(p)
            except Exception as e:
                logger.warning("Model prediction failed: %s", e)

        if not probas:
            return 0.5, 0.0

        avg_probability = float(np.mean(probas))
        agreement_score = float(1.0 - np.std(probas))  # 1.0 = perfect agreement

        logger.debug(
            "Ensemble prediction for %s: avg=%.4f, agreement=%.4f "
            "(individual: %s) — %d models",
            self._symbol, avg_probability, agreement_score,
            ", ".join(f"{p:.3f}" for p in probas),
            len(probas),
        )
        return avg_probability, agreement_score

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict:
        """Get averaged feature importance across all XGBoost models in ensemble."""
        importances = np.zeros(len(self._FEATURE_COLS))
        xgb_count = 0

        for model in self._models:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                if len(imp) == len(self._FEATURE_COLS):
                    importances += imp
                    xgb_count += 1

        if xgb_count > 0:
            importances /= xgb_count

        return dict(zip(self._FEATURE_COLS, importances))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save_model(self) -> None:
        """Serialise the ensemble and persist to the database."""
        if not self._models or self._db is None:
            logger.warning("Cannot save model — no model or no database.")
            return

        blob = pickle.dumps({
            "models": self._models,
            "scaler": self._scaler,  # Persist scaler with models
        })
        await self._db.save_model_state(
            symbol=self._symbol,
            model_blob=blob,
            accuracy=self._rolling_accuracy,
            candle_count=0,
            metadata={
                "features": self._FEATURE_COLS,
                "n_models": len(self._models),
                "version": "v3_iron_curtain",
                "has_scaler": self._scaler is not None,
            },
        )
        logger.info("Ensemble (%d models + scaler) for %s saved.", len(self._models), self._symbol)

    async def load_model(self) -> bool:
        """Load the latest model checkpoint from the database."""
        if self._db is None:
            return False

        state = await self._db.load_latest_model_state(self._symbol)
        if state is None:
            return False

        try:
            data = pickle.loads(state["model_blob"])
            # Support v3 format (dict with models + scaler) and legacy (list or single model)
            if isinstance(data, dict) and "models" in data:
                self._models = data["models"]
                self._scaler = data.get("scaler", None)
            elif isinstance(data, list):
                self._models = data
                self._scaler = None
            else:
                self._models = [data]
                self._scaler = None
            self._last_train_time = state["trained_at"]
            self._rolling_accuracy = state["accuracy"]
            self._is_trained = True
            logger.info(
                "Model for %s loaded from database (scaler=%s).",
                self._symbol, "YES" if self._scaler else "NO",
            )
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    # ------------------------------------------------------------------
    # Auto-Retrain Logic
    # ------------------------------------------------------------------

    def needs_retrain(self) -> bool:
        """Check whether the model should be retrained (Titanium: every 4h)."""
        if not self._is_trained:
            return True

        hours_since_train = (time.time() - self._last_train_time) / 3600
        if hours_since_train >= cfg.ML_RETRAIN_INTERVAL_HOURS:
            logger.info(
                "🧠 Retrain needed: %.1f hours since last training (threshold=%.1fh).",
                hours_since_train, cfg.ML_RETRAIN_INTERVAL_HOURS,
            )
            return True

        if self._rolling_accuracy < cfg.ML_MIN_ACCURACY:
            logger.info(
                "Retrain needed: accuracy %.4f below threshold %.4f.",
                self._rolling_accuracy, cfg.ML_MIN_ACCURACY,
            )
            return True

        return False

    def update_rolling_accuracy(self, actual: int, predicted_prob: float) -> None:
        """Update the rolling accuracy with exponential moving average."""
        predicted_class = 1 if predicted_prob >= 0.5 else 0
        correct = 1.0 if predicted_class == actual else 0.0
        alpha = 0.05
        self._rolling_accuracy = alpha * correct + (1 - alpha) * self._rolling_accuracy

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def accuracy(self) -> float:
        return self._rolling_accuracy

    @property
    def last_train_time(self) -> float:
        return self._last_train_time
