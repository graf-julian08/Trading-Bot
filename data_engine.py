"""
data_engine.py — Async OHLCV Data Fetcher
==========================================
Handles all communication with crypto/stock exchanges for candle data.

Key features:
  - Paginated historical OHLCV download (for ML training).
  - Live candle polling with configurable interval.
  - Automatic reconnection with exponential backoff on failures.
  - Rate-limit compliance via ccxt's built-in limiter + manual throttle.
  - Returns clean pandas DataFrames ready for indicator computation.

Usage:
    engine = DataEngine()
    await engine.initialise()
    hist_df = await engine.fetch_ohlcv_history("BTC/USDT", limit=5000)
    live_df = await engine.fetch_live_ohlcv("BTC/USDT")
    await engine.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Exponential backoff settings for retry logic.
_BACKOFF_BASE = 2.0       # seconds
_BACKOFF_MAX = 60.0       # max wait between retries
_MAX_RETRIES = 10         # give up after this many consecutive failures

# Maximum candles per single API call (exchange-dependent, 1000 is safe default).
_BATCH_SIZE = 1000


class DataEngine:
    """
    Async OHLCV data engine using ccxt.

    Manages the exchange connection lifecycle and provides methods for
    fetching both historical and live candle data.
    """

    def __init__(self) -> None:
        self._exchange: Optional[ccxt_async.Exchange] = None
        self._initialised = False
        # Heartbeat: tracks when we last successfully received data.
        # Used to detect "zombie connections" where socket looks open
        # but no data flows.
        self._last_data_received: float = time.time()
        self._reconnect_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """
        Create and configure the ccxt async exchange instance.

        Loads markets so symbol validation works immediately.
        """
        exchange_class = getattr(ccxt_async, cfg.EXCHANGE_ID, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {cfg.EXCHANGE_ID}")

        # Build config — only include API keys if actually set.
        # Public endpoints (OHLCV, ticker, orderbook) work without keys.
        exchange_config = {
            "enableRateLimit": True,  # ccxt built-in rate limiter
            "options": {
                "defaultType": "spot",
            },
        }
        if cfg.EXCHANGE_API_KEY:
            exchange_config["apiKey"] = cfg.EXCHANGE_API_KEY
        if cfg.EXCHANGE_API_SECRET:
            exchange_config["secret"] = cfg.EXCHANGE_API_SECRET
        if cfg.EXCHANGE_PASSWORD:
            exchange_config["password"] = cfg.EXCHANGE_PASSWORD

        self._exchange = exchange_class(exchange_config)

        # Use sandbox/testnet if configured.
        if cfg.EXCHANGE_SANDBOX:
            self._exchange.set_sandbox_mode(True)
            logger.info("Exchange sandbox mode ENABLED.")

        # Load markets to validate symbols later.
        await self._retry(self._exchange.load_markets)
        self._initialised = True
        logger.info(
            "DataEngine initialised for %s (%d markets loaded).",
            cfg.EXCHANGE_ID,
            len(self._exchange.markets),
        )

    async def close(self) -> None:
        """Close the exchange connection gracefully."""
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning("Error closing exchange connection: %s", e)
            self._exchange = None
            self._initialised = False
            logger.info("DataEngine closed.")

    async def reconnect(self) -> None:
        """
        Force a hard reconnect: close the current connection and
        reinitialise from scratch.

        Used when:
          - Heartbeat timeout triggers (zombie connection detected)
          - Consecutive retry failures suggest a stale socket
        """
        self._reconnect_count += 1
        logger.warning(
            "🔄 HARD RECONNECT #%d: Forcing new exchange connection ...",
            self._reconnect_count,
        )
        await self.close()
        await asyncio.sleep(2.0)  # Brief cooldown before reconnecting
        await self.initialise()
        self._last_data_received = time.time()
        logger.info(
            "✅ Reconnect #%d complete. Heartbeat reset.",
            self._reconnect_count,
        )

    async def check_heartbeat(self) -> bool:
        """
        Check if data has been received within the heartbeat timeout.

        If the heartbeat is stale (no data for DATA_HEARTBEAT_TIMEOUT_SECONDS),
        forces a hard reconnect.

        Returns
        -------
        bool
            True if heartbeat is healthy, False if reconnect was triggered.
        """
        elapsed = time.time() - self._last_data_received
        if elapsed > cfg.DATA_HEARTBEAT_TIMEOUT_SECONDS:
            logger.warning(
                "💀 ZOMBIE CONNECTION DETECTED: No data for %.1f seconds "
                "(timeout=%.1f). Forcing hard reconnect.",
                elapsed, cfg.DATA_HEARTBEAT_TIMEOUT_SECONDS,
            )
            await self.reconnect()
            return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_ohlcv_history(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Download historical OHLCV candles with automatic pagination.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTC/USDT".
        timeframe : str, optional
            Candle timeframe (default: cfg.TIMEFRAME).
        limit : int, optional
            Total number of candles to fetch (default: cfg.CANDLE_LIMIT).
        since : int, optional
            Unix timestamp in milliseconds to start from.  If None, the
            engine calculates it based on `limit` and `timeframe`.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, open, high, low, close, volume
            Sorted by timestamp ascending, no duplicates.
        """
        self._ensure_ready()
        tf = timeframe or cfg.TIMEFRAME
        total = limit or cfg.CANDLE_LIMIT

        # If no explicit start time, estimate it from the requested candle count.
        if since is None:
            tf_ms = self._timeframe_to_ms(tf)
            since = int(time.time() * 1000) - (total * tf_ms)

        all_candles: List[list] = []
        fetched = 0

        logger.info(
            "Fetching %d historical candles for %s [%s] ...", total, symbol, tf
        )

        while fetched < total:
            batch_limit = min(_BATCH_SIZE, total - fetched)
            candles = await self._retry(
                self._exchange.fetch_ohlcv,
                symbol, tf, since=since, limit=batch_limit,
            )

            if not candles:
                logger.warning("No more candles returned — ending pagination.")
                break

            all_candles.extend(candles)
            fetched += len(candles)

            # Move 'since' pointer forward past the last candle.
            since = candles[-1][0] + 1

            logger.debug("Fetched %d / %d candles ...", fetched, total)

            # Small delay to be polite to the API.
            await asyncio.sleep(0.25)

        df = self._candles_to_dataframe(all_candles)
        logger.info(
            "Historical fetch complete: %d candles for %s.", len(df), symbol
        )
        return df

    async def fetch_live_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch the most recent candles (live polling).

        Parameters
        ----------
        symbol : str
            Trading pair.
        timeframe : str, optional
            Candle timeframe (default: cfg.TIMEFRAME).
        limit : int
            Number of recent candles (default 100, enough for indicators).

        Returns
        -------
        pd.DataFrame
            Same schema as fetch_ohlcv_history.
        """
        self._ensure_ready()
        tf = timeframe or cfg.TIMEFRAME

        candles = await self._retry(
            self._exchange.fetch_ohlcv,
            symbol, tf, limit=limit,
        )
        df = self._candles_to_dataframe(candles)
        return df

    async def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch the current ticker (bid, ask, last, spread, etc.).

        Returns the raw ccxt ticker dict.
        """
        self._ensure_ready()
        ticker = await self._retry(self._exchange.fetch_ticker, symbol)
        return ticker

    async def fetch_balance(self) -> Dict:
        """
        Fetch the account balance.

        Returns the raw ccxt balance dict with 'free', 'used', 'total' keys.
        """
        self._ensure_ready()
        balance = await self._retry(self._exchange.fetch_balance)
        return balance

    async def fetch_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """
        Fetch the order book for spread calculation.

        Returns dict with 'bids' and 'asks' arrays.
        """
        self._ensure_ready()
        book = await self._retry(
            self._exchange.fetch_order_book, symbol, limit
        )
        return book

    # ------------------------------------------------------------------
    # Derivatives Data (Binance Futures — Free, No API Key Required)
    # ------------------------------------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch the current funding rate from Binance Futures (free, no key).

        Negative funding = shorts pay longs = bearish pressure overloaded
        → contrarian LONG signal.

        Returns
        -------
        dict
            {"funding_rate": float, "funding_rate_zscore": float}
            Returns zeros on error (graceful degradation).
        """
        default = {"funding_rate": 0.0, "funding_rate_zscore": 0.0}
        try:
            # Convert "BTC/USDT" → "BTCUSDT"
            binance_symbol = symbol.replace("/", "")
            url = (
                f"https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol={binance_symbol}&limit=30"
            )
            data = await self._http_get(url)
            if not data:
                return default

            # Latest funding rate
            latest_rate = float(data[-1].get("fundingRate", 0.0))

            # Z-score: how extreme is current funding vs recent history?
            rates = [float(d.get("fundingRate", 0.0)) for d in data]
            if len(rates) >= 5:
                mean_r = np.mean(rates)
                std_r = np.std(rates) + 1e-10
                zscore = (latest_rate - mean_r) / std_r
            else:
                zscore = 0.0

            self._last_data_received = time.time()
            logger.debug(
                "Funding rate for %s: %.6f (z=%.2f)",
                symbol, latest_rate, zscore,
            )
            return {
                "funding_rate": latest_rate,
                "funding_rate_zscore": zscore,
            }
        except Exception as e:
            logger.warning("Failed to fetch funding rate for %s: %s", symbol, e)
            return default

    async def fetch_open_interest(self, symbol: str) -> Dict:
        """
        Fetch open interest + short-term OI change from Binance Futures (free).

        Rising OI + falling price = shorts piling in → liquidation risk.
        Falling OI + rising price = shorts closing → sustainable rally.

        Returns
        -------
        dict
            {"oi": float, "oi_change_5m": float, "oi_change_1h": float}
            Returns zeros on error (graceful degradation).
        """
        default = {"oi": 0.0, "oi_change_5m": 0.0, "oi_change_1h": 0.0}
        try:
            binance_symbol = symbol.replace("/", "")

            # Current OI
            url_current = (
                f"https://fapi.binance.com/fapi/v1/openInterest"
                f"?symbol={binance_symbol}"
            )
            current_data = await self._http_get(url_current)
            if not current_data:
                return default
            current_oi = float(current_data.get("openInterest", 0.0))

            # OI history (5m intervals) for delta calculation
            url_hist = (
                f"https://fapi.binance.com/futures/data/openInterestHist"
                f"?symbol={binance_symbol}&period=5m&limit=13"
            )
            hist_data = await self._http_get(url_hist)

            oi_change_5m = 0.0
            oi_change_1h = 0.0

            if hist_data and len(hist_data) >= 2:
                prev_oi = float(hist_data[-2].get("sumOpenInterest", current_oi))
                if prev_oi > 0:
                    oi_change_5m = (current_oi - prev_oi) / prev_oi

            if hist_data and len(hist_data) >= 13:
                oi_1h_ago = float(hist_data[0].get("sumOpenInterest", current_oi))
                if oi_1h_ago > 0:
                    oi_change_1h = (current_oi - oi_1h_ago) / oi_1h_ago

            self._last_data_received = time.time()
            logger.debug(
                "OI for %s: %.2f (Δ5m=%.4f, Δ1h=%.4f)",
                symbol, current_oi, oi_change_5m, oi_change_1h,
            )
            return {
                "oi": current_oi,
                "oi_change_5m": oi_change_5m,
                "oi_change_1h": oi_change_1h,
            }
        except Exception as e:
            logger.warning("Failed to fetch OI for %s: %s", symbol, e)
            return default

    async def fetch_fear_greed_index(self) -> Dict:
        """
        Fetch the Fear & Greed Index from alternative.me (free, no key).

        Extreme Fear (< 20) = historically best buy zones.
        Extreme Greed (> 80) = historically best sell zones.

        Returns
        -------
        dict
            {"value": int, "classification": str}
            Returns neutral on error.
        """
        default = {"value": 50, "classification": "Neutral"}
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            data = await self._http_get(url)
            if not data or "data" not in data or not data["data"]:
                return default

            entry = data["data"][0]
            return {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
            }
        except Exception as e:
            logger.warning("Failed to fetch Fear & Greed Index: %s", e)
            return default

    async def fetch_liquidation_data(self, symbol: str) -> Dict:
        """
        Fetch recent forced liquidations from Binance Futures (free).

        Aggregates liquidation volume above/below current price to identify
        where stop-losses are clustered (liquidation magnets).

        Returns
        -------
        dict
            {"liq_buy_volume": float, "liq_sell_volume": float}
        """
        default = {"liq_buy_volume": 0.0, "liq_sell_volume": 0.0}
        try:
            binance_symbol = symbol.replace("/", "")
            url = (
                f"https://fapi.binance.com/fapi/v1/allForceOrders"
                f"?symbol={binance_symbol}&limit=100"
            )
            data = await self._http_get(url)
            if not data:
                return default

            buy_vol = sum(
                float(d.get("origQty", 0))
                for d in data if d.get("side") == "BUY"
            )
            sell_vol = sum(
                float(d.get("origQty", 0))
                for d in data if d.get("side") == "SELL"
            )
            return {"liq_buy_volume": buy_vol, "liq_sell_volume": sell_vol}
        except Exception as e:
            logger.warning("Failed to fetch liquidation data for %s: %s", symbol, e)
            return default

    # ------------------------------------------------------------------
    # Historical Derivatives Data (for Training — closes serving skew)
    # ------------------------------------------------------------------

    async def fetch_historical_funding_rates(
        self, symbol: str, start_time_ms: int, end_time_ms: int,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Binance Futures.

        Returns a DataFrame with columns: [timestamp, funding_rate]
        Funding rates are published every 8h on Binance.
        """
        binance_symbol = symbol.replace("/", "")
        all_data = []
        current_start = start_time_ms
        chunk_size = 7 * 24 * 3600 * 1000  # 7 days in ms

        logger.info(
            "Fetching historical funding rates for %s ...", symbol,
        )

        while current_start < end_time_ms:
            # Chunking to avoid "startTime invalid" error (>30d range)
            current_end = min(current_start + chunk_size, end_time_ms)
            
            url = (
                f"https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol={binance_symbol}&startTime={current_start}"
                f"&endTime={current_end}&limit=1000"
            )
            data = await self._http_get(url)
            if not data or len(data) == 0:
                current_start = current_end + 1
                continue

            all_data.extend(data)
            # Move past the chunk
            current_start = current_end + 1
            await asyncio.sleep(0.3)  # Rate limit politeness

        if not all_data:
            logger.warning("No historical funding rate data for %s.", symbol)
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(
            df["fundingTime"].astype(int), unit="ms", utc=True,
        )
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].drop_duplicates("timestamp")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            "Fetched %d historical funding rate records for %s.",
            len(df), symbol,
        )
        return df

    async def fetch_historical_open_interest(
        self, symbol: str, start_time_ms: int, end_time_ms: int,
        period: str = "1h",
    ) -> pd.DataFrame:
        """
        Fetch historical open interest from Binance Futures.

        Returns a DataFrame with columns: [timestamp, open_interest]
        """
        binance_symbol = symbol.replace("/", "")
        all_data = []
        chunk_size = 7 * 24 * 3600 * 1000  # 7 days in ms

        # Binance public API only provides ~30 days of OI history.
        # Clamp start_time to avoid wasting minutes on HTTP 400 retries.
        max_history_ms = 30 * 24 * 3600 * 1000  # 30 days
        effective_start = max(start_time_ms, end_time_ms - max_history_ms)
        if effective_start > start_time_ms:
            skipped_days = (effective_start - start_time_ms) / (24 * 3600 * 1000)
            logger.info(
                "OI history limited to ~30 days. Skipping %.0f older days for %s.",
                skipped_days, symbol,
            )
        current_start = effective_start

        logger.info(
            "Fetching historical open interest for %s [%s] ...",
            symbol, period,
        )

        while current_start < end_time_ms:
            # Chunking to avoid "startTime invalid" error (>30d range)
            current_end = min(current_start + chunk_size, end_time_ms)

            url = (
                f"https://fapi.binance.com/futures/data/openInterestHist"
                f"?symbol={binance_symbol}&period={period}"
                f"&startTime={current_start}&endTime={current_end}"
                f"&limit=500"
            )
            data = await self._http_get(url)
            if not data or len(data) == 0:
                current_start = current_end + 1
                continue

            all_data.extend(data)
            current_start = current_end + 1
            await asyncio.sleep(0.3)

        if not all_data:
            logger.warning("No historical OI data for %s.", symbol)
            return pd.DataFrame(columns=["timestamp", "open_interest"])

        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(int), unit="ms", utc=True,
        )
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df = df[["timestamp", "open_interest"]].drop_duplicates("timestamp")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            "Fetched %d historical OI records for %s.", len(df), symbol,
        )
        return df

    async def enrich_training_data(
        self, df: pd.DataFrame, symbol: str,
    ) -> pd.DataFrame:
        """
        Enrich an OHLCV DataFrame with historical derivatives data for training.

        This method closes the "Training-Serving Skew" by fetching the same
        advanced features during training that `main.py` injects at runtime.

        Columns added:
          - funding_rate, funding_rate_zscore
          - oi_change_5m, oi_change_1h
          - fear_greed_norm (static default — not available historically)
          - liq_imbalance (static default — not available historically)
          - extreme_fear_flag, extreme_greed_flag
          - funding_oi_interaction
        """
        if df.empty or "timestamp" not in df.columns:
            return df

        df = df.copy()
        start_ms = int(df["timestamp"].iloc[0].timestamp() * 1000)
        end_ms = int(df["timestamp"].iloc[-1].timestamp() * 1000)

        # ---- Funding Rates ----
        try:
            funding_df = await self.fetch_historical_funding_rates(
                symbol, start_ms, end_ms,
            )
            if not funding_df.empty:
                # Merge: forward-fill funding rates (published every 8h)
                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    funding_df.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                # Compute z-score over a rolling 30-period window
                df["funding_rate"] = df["funding_rate"].fillna(0.0)
                fr_mean = df["funding_rate"].rolling(30, min_periods=1).mean()
                fr_std = df["funding_rate"].rolling(30, min_periods=1).std().fillna(1e-10) + 1e-10
                df["funding_rate_zscore"] = (df["funding_rate"] - fr_mean) / fr_std
            else:
                df["funding_rate"] = 0.0
                df["funding_rate_zscore"] = 0.0
        except Exception as e:
            logger.warning("Failed to fetch historical funding: %s", e)
            df["funding_rate"] = 0.0
            df["funding_rate_zscore"] = 0.0

        # ---- Open Interest ----
        try:
            oi_df = await self.fetch_historical_open_interest(
                symbol, start_ms, end_ms, period="1h",
            )
            if not oi_df.empty:
                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    oi_df.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                df["open_interest"] = df["open_interest"].fillna(method="ffill").fillna(0.0)
                # OI changes (% change over 1 and 12 candles ≈ 5min & 1h equivalent)
                df["oi_change_5m"] = df["open_interest"].pct_change(1).fillna(0.0)
                df["oi_change_1h"] = df["open_interest"].pct_change(1).fillna(0.0)
                df.drop(columns=["open_interest"], inplace=True)
            else:
                df["oi_change_5m"] = 0.0
                df["oi_change_1h"] = 0.0
        except Exception as e:
            logger.warning("Failed to fetch historical OI: %s", e)
            df["oi_change_5m"] = 0.0
            df["oi_change_1h"] = 0.0

        # ---- Interaction feature ----
        df["funding_oi_interaction"] = (
            df["funding_rate"] * 1000
        ) * (
            df["oi_change_5m"] * 100
        )

        # ---- Static defaults for features without free historical APIs ----
        if "fear_greed_norm" not in df.columns:
            df["fear_greed_norm"] = 0.5  # Neutral
        if "liq_imbalance" not in df.columns:
            df["liq_imbalance"] = 0.5  # Balanced
        if "extreme_fear_flag" not in df.columns:
            df["extreme_fear_flag"] = 0.0
        if "extreme_greed_flag" not in df.columns:
            df["extreme_greed_flag"] = 0.0

        # ---- Order flow (not available historically — neutral defaults) ----
        for col in ["cvd_1m", "buy_sell_ratio", "large_trade_ratio", "trade_intensity"]:
            if col not in df.columns:
                df[col] = 0.0
        # buy_sell_ratio neutral = 1.0
        if (df["buy_sell_ratio"] == 0.0).all():
            df["buy_sell_ratio"] = 1.0

        # ---- Cross-asset (not available historically — neutral defaults) ----
        for col in ["btc_dominance_proxy", "cross_asset_momentum", "relative_strength"]:
            if col not in df.columns:
                df[col] = 0.0

        logger.info(
            "Training data enriched: %d rows, funding=%s, OI=%s",
            len(df),
            "YES" if "funding_rate" in df.columns and (df["funding_rate"] != 0).any() else "NO",
            "YES" if "oi_change_5m" in df.columns and (df["oi_change_5m"] != 0).any() else "NO",
        )

        df.reset_index(drop=True, inplace=True)
        return df

    async def _http_get(self, url: str) -> Any:
        """
        Perform a simple async HTTP GET with retry.

        Used for direct Binance Futures API calls that don't go through ccxt.
        Returns parsed JSON or None on failure.
        """
        import aiohttp as _aiohttp

        for attempt in range(1, 4):
            try:
                async with _aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        logger.warning(
                            "HTTP %d from %s (attempt %d/3)",
                            resp.status, url.split("?")[0], attempt,
                        )
            except Exception as e:
                logger.warning(
                    "HTTP GET failed (attempt %d/3): %s", attempt, e,
                )
            await asyncio.sleep(1.0 * attempt)
        return None

    @property
    def exchange(self) -> ccxt_async.Exchange:
        """Direct access to the underlying ccxt exchange (for order placement)."""
        self._ensure_ready()
        return self._exchange

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Raise if the engine hasn't been initialised."""
        if not self._initialised or self._exchange is None:
            raise RuntimeError("DataEngine not initialised — call .initialise() first.")

    async def _retry(self, func, *args, **kwargs):
        """
        Execute an async function with exponential backoff retry.

        Handles network errors, exchange errors, and rate limits
        transparently.  After _MAX_RETRIES consecutive failures the
        exception is re-raised to the caller.
        """
        last_exception = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = await func(*args, **kwargs)
                # Stamp heartbeat on every successful data receipt.
                self._last_data_received = time.time()
                return result
            except (
                ccxt_async.NetworkError,
                ccxt_async.ExchangeNotAvailable,
                ccxt_async.RequestTimeout,
                ccxt_async.DDoSProtection,
            ) as e:
                last_exception = e
                wait = min(_BACKOFF_BASE * (2 ** (attempt - 1)), _BACKOFF_MAX)
                logger.warning(
                    "Retry %d/%d for %s: %s — waiting %.1fs",
                    attempt, _MAX_RETRIES, func.__name__, e, wait,
                )
                await asyncio.sleep(wait)

                # If we've failed 3+ times in a row, try a hard reconnect
                # before continuing retries.
                if attempt == 3 and self._initialised:
                    logger.warning(
                        "🔄 3 consecutive failures — attempting hard reconnect ..."
                    )
                    try:
                        await self.reconnect()
                    except Exception as re:
                        logger.error("Reconnect failed: %s", re)

            except ccxt_async.RateLimitExceeded as e:
                last_exception = e
                # Rate limit hit — wait longer.
                wait = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)
                logger.warning(
                    "Rate limit hit on %s — backing off %.1fs",
                    func.__name__, wait,
                )
                await asyncio.sleep(wait)
            except ccxt_async.ExchangeError as e:
                # Non-transient exchange errors should not be retried.
                logger.error("Exchange error (non-retryable): %s", e)
                raise

        # Exhausted retries.
        logger.error(
            "All %d retries exhausted for %s.", _MAX_RETRIES, func.__name__
        )
        raise last_exception

    @staticmethod
    def _candles_to_dataframe(candles: List[list]) -> pd.DataFrame:
        """
        Convert raw ccxt OHLCV candle arrays into a clean DataFrame.

        Each candle is [timestamp_ms, open, high, low, close, volume].
        """
        if not candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Convert millisecond timestamp to pandas datetime (UTC).
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Drop duplicates and sort.
        df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Ensure numeric types.
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN prices (shouldn't happen, but defensive).
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        return df

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        """
        Convert a ccxt-style timeframe string to milliseconds.

        Examples: '1m' -> 60_000, '1h' -> 3_600_000, '1d' -> 86_400_000
        """
        multipliers = {
            "s": 1_000,
            "m": 60_000,
            "h": 3_600_000,
            "d": 86_400_000,
            "w": 604_800_000,
            "M": 2_592_000_000,  # ~30 days
        }
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        return amount * multipliers.get(unit, 60_000)
