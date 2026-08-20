"""
ws_manager.py — Binance WebSocket Connection Manager
=====================================================
Event-driven real-time data feed using Binance's free public WebSocket API.

Streams:
  1. **kline_1m** — 1-minute candles for signal generation (sub-second updates).
  2. **bookTicker** — Real-time best bid/ask for execution decisions.
  3. **ticker** — 24h rolling price/volume for monitoring.

Architecture:
  - `ConnectionManager` handles auto-reconnect, pong responses, and
    stream multiplexing via Binance's combined stream endpoint.
  - Callbacks are registered per event type so the main loop is event-driven
    (no more 60s polling).
  - Falls back to REST gracefully if the WS dies.

Latency goal: <200ms reaction time (vs. 60,000ms before).

Usage:
    ws = ConnectionManager(symbols=["BTC/USDT", "ETH/USDT"])
    ws.on_kline(my_kline_handler)
    ws.on_book_ticker(my_book_handler)
    await ws.start()
    ...
    await ws.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import numpy as np
import aiohttp

from config import cfg

logger = logging.getLogger(__name__)

# ============================================================================
# Type Aliases
# ============================================================================

# Callback signature: async def handler(data: dict) -> None
AsyncCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]

# ============================================================================
# Binance WebSocket Endpoints
# ============================================================================

_BINANCE_WS_BASE = "wss://stream.binance.com:9443"
_BINANCE_WS_COMBINED = f"{_BINANCE_WS_BASE}/stream?streams="

# ============================================================================
# Connection Manager
# ============================================================================


class ConnectionManager:
    """
    Manages a persistent Binance WebSocket connection with:
      - Combined stream multiplexing (one connection for all streams).
      - Automatic reconnection with exponential backoff.
      - Pong response handling (Binance sends pings every 3 min).
      - Event-driven callbacks for kline, bookTicker, and ticker data.
      - Thread-safe data snapshot access for the execution engine.
    """

    # Reconnect settings
    _RECONNECT_BASE = 1.0       # initial wait (seconds)
    _RECONNECT_MAX = 30.0       # max backoff
    _RECONNECT_MULTIPLIER = 2.0

    # Binance sends a ping every ~3 minutes; we send a pong.
    # If no message arrives for 5 minutes, assume dead connection.
    _HEARTBEAT_TIMEOUT = 300.0  # 5 minutes

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self._symbols = symbols or cfg.TRADING_PAIRS
        self._running = False
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None

        # Callbacks
        self._kline_callbacks: List[AsyncCallback] = []
        self._book_ticker_callbacks: List[AsyncCallback] = []
        self._ticker_callbacks: List[AsyncCallback] = []
        self._agg_trade_callbacks: List[AsyncCallback] = []
        self._raw_callbacks: List[AsyncCallback] = []

        # Latest snapshots (thread-safe via asyncio single-thread model)
        self._best_bid: Dict[str, float] = {}
        self._best_ask: Dict[str, float] = {}
        self._best_bid_qty: Dict[str, float] = {}
        self._best_ask_qty: Dict[str, float] = {}
        self._last_price: Dict[str, float] = {}
        self._last_kline: Dict[str, Dict[str, Any]] = {}

        # ---- Alpha v2: Order Flow (aggTrade accumulator) ----
        # Rolling 1-minute buckets for buy/sell volume tracking.
        self._trade_flow: Dict[str, dict] = {}  # {symbol: flow_state}
        self._trade_sizes: Dict[str, deque] = {}  # for dynamic large-trade threshold
        for sym in self._symbols:
            self._trade_flow[sym] = {
                "buy_vol": 0.0,
                "sell_vol": 0.0,
                "trade_count": 0,
                "large_trade_count": 0,
                "window_start": time.time(),
            }
            self._trade_sizes[sym] = deque(maxlen=500)  # recent trade sizes

        # Connection metrics
        self._connected = False
        self._last_message_time: float = 0.0
        self._reconnect_count = 0
        self._total_messages = 0

    # ------------------------------------------------------------------
    # Callback Registration
    # ------------------------------------------------------------------

    def on_kline(self, callback: AsyncCallback) -> None:
        """Register a callback for 1-minute kline updates."""
        self._kline_callbacks.append(callback)

    def on_book_ticker(self, callback: AsyncCallback) -> None:
        """Register a callback for best bid/ask updates."""
        self._book_ticker_callbacks.append(callback)

    def on_ticker(self, callback: AsyncCallback) -> None:
        """Register a callback for 24h ticker updates."""
        self._ticker_callbacks.append(callback)

    def on_agg_trade(self, callback: AsyncCallback) -> None:
        """Register a callback for aggregated trade events."""
        self._agg_trade_callbacks.append(callback)

    def on_raw(self, callback: AsyncCallback) -> None:
        """Register a callback for ALL raw messages (for debugging)."""
        self._raw_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Data Snapshots (non-blocking reads)
    # ------------------------------------------------------------------

    def get_best_bid(self, symbol: str) -> Optional[float]:
        """Get the latest best bid price for a symbol."""
        return self._best_bid.get(self._normalise_symbol(symbol))

    def get_best_ask(self, symbol: str) -> Optional[float]:
        """Get the latest best ask price for a symbol."""
        return self._best_ask.get(self._normalise_symbol(symbol))

    def get_best_bid_qty(self, symbol: str) -> Optional[float]:
        """Get the latest best bid quantity for a symbol."""
        return self._best_bid_qty.get(self._normalise_symbol(symbol))

    def get_best_ask_qty(self, symbol: str) -> Optional[float]:
        """Get the latest best ask quantity for a symbol."""
        return self._best_ask_qty.get(self._normalise_symbol(symbol))

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get the current spread as a fraction of mid price."""
        sym = self._normalise_symbol(symbol)
        bid = self._best_bid.get(sym)
        ask = self._best_ask.get(sym)
        if bid and ask and bid > 0:
            mid = (bid + ask) / 2
            return (ask - bid) / mid
        return None

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get the latest trade price for a symbol."""
        return self._last_price.get(self._normalise_symbol(symbol))

    def get_last_kline(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest 1m kline data for a symbol."""
        return self._last_kline.get(self._normalise_symbol(symbol))

    def get_trade_flow(self, symbol: str) -> Dict[str, float]:
        """
        Get order flow metrics for a symbol.

        Returns
        -------
        dict with keys:
            cvd_1m: Cumulative Volume Delta (buy - sell) in current window
            buy_sell_ratio: taker buy / taker sell volume
            large_trade_ratio: fraction of volume from large trades
            trade_intensity: trades per second in current window
        """
        sym = self._normalise_symbol(symbol)
        flow = self._trade_flow.get(sym)
        if not flow:
            return {
                "cvd_1m": 0.0, "buy_sell_ratio": 1.0,
                "large_trade_ratio": 0.0, "trade_intensity": 0.0,
            }

        buy_v = flow["buy_vol"]
        sell_v = flow["sell_vol"]
        total_v = buy_v + sell_v + 1e-10
        elapsed = max(time.time() - flow["window_start"], 1.0)

        return {
            "cvd_1m": buy_v - sell_v,
            "buy_sell_ratio": buy_v / (sell_v + 1e-10),
            "large_trade_ratio": flow["large_trade_count"] / max(flow["trade_count"], 1),
            "trade_intensity": flow["trade_count"] / elapsed,
        }

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is currently connected and alive."""
        return self._connected and self._ws is not None and not self._ws.closed

    @property
    def last_message_age(self) -> float:
        """Seconds since the last message was received."""
        if self._last_message_time == 0.0:
            return float("inf")
        return time.time() - self._last_message_time

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket connection in a background task."""
        if self._running:
            logger.warning("ConnectionManager already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_forever())
        logger.info(
            "🔌 WebSocket ConnectionManager started for %d symbols: %s",
            len(self._symbols), ", ".join(self._symbols),
        )

    async def stop(self) -> None:
        """Gracefully shut down the WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._connected = False
        logger.info("🔌 WebSocket ConnectionManager stopped.")

    # ------------------------------------------------------------------
    # Core Connection Loop
    # ------------------------------------------------------------------

    async def _run_forever(self) -> None:
        """Main connection loop with auto-reconnect."""
        backoff = self._RECONNECT_BASE

        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                logger.info("WS loop cancelled — shutting down.")
                break
            except Exception as exc:
                logger.error("WS connection error: %s", exc)

            if not self._running:
                break

            # Exponential backoff for reconnection
            self._connected = False
            self._reconnect_count += 1
            logger.warning(
                "🔄 WS reconnecting in %.1fs (attempt #%d)...",
                backoff, self._reconnect_count,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * self._RECONNECT_MULTIPLIER, self._RECONNECT_MAX)

        self._connected = False

    async def _connect_and_listen(self) -> None:
        """Establish a WS connection and process messages until disconnect."""
        url = self._build_stream_url()
        logger.info("🔌 Connecting to Binance WS: %s", url[:120] + "...")

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                url,
                heartbeat=20.0,    # aiohttp sends ping every 20s
                timeout=30.0,
            )
            self._connected = True
            self._reconnect_count = 0
            logger.info("✅ WebSocket connected — streaming real-time data.")

            async for msg in self._ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._last_message_time = time.time()
                    self._total_messages += 1
                    await self._handle_message(msg.data)

                elif msg.type == aiohttp.WSMsgType.PING:
                    await self._ws.pong(msg.data)
                    self._last_message_time = time.time()

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    logger.warning("WS message type: %s — will reconnect.", msg.type)
                    break
        finally:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session and not self._session.closed:
                await self._session.close()
            self._connected = False

    # ------------------------------------------------------------------
    # Stream URL Builder
    # ------------------------------------------------------------------

    def _build_stream_url(self) -> str:
        """
        Build the Binance combined stream URL.

        Streams per symbol:
          - <symbol>@kline_1m       — 1-minute candles
          - <symbol>@bookTicker     — best bid/ask
          - <symbol>@miniTicker     — 24h price + volume (lighter than full ticker)
        """
        streams = []
        for symbol in self._symbols:
            s = self._to_binance_symbol(symbol)
            streams.append(f"{s}@kline_1m")
            streams.append(f"{s}@bookTicker")
            streams.append(f"{s}@miniTicker")
            streams.append(f"{s}@aggTrade")  # Alpha v2: Order Flow
        return _BINANCE_WS_COMBINED + "/".join(streams)

    # ------------------------------------------------------------------
    # Message Dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch a WebSocket message to the appropriate handler."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("WS received non-JSON message: %s", raw[:200])
            return

        # Fire raw callbacks (for debugging / logging)
        for cb in self._raw_callbacks:
            try:
                await cb(payload)
            except Exception as exc:
                logger.error("Raw callback error: %s", exc)

        # Binance combined stream format: {"stream": "...", "data": {...}}
        stream = payload.get("stream", "")
        data = payload.get("data", payload)  # fallback for single streams

        if "@kline_" in stream:
            await self._on_kline_event(data)
        elif "@bookTicker" in stream:
            await self._on_book_ticker_event(data)
        elif "@miniTicker" in stream or "@ticker" in stream:
            await self._on_ticker_event(data)
        elif "@aggTrade" in stream:
            await self._on_agg_trade_event(data)

    async def _on_kline_event(self, data: Dict[str, Any]) -> None:
        """Process a kline/candlestick event."""
        k = data.get("k", {})
        symbol = self._from_binance_symbol(data.get("s", ""))

        kline_data = {
            "symbol": symbol,
            "interval": k.get("i"),
            "open_time": k.get("t"),
            "close_time": k.get("T"),
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "is_closed": k.get("x", False),   # True when candle is finalised
            "trades": k.get("n", 0),
        }

        self._last_kline[symbol] = kline_data
        self._last_price[symbol] = kline_data["close"]

        for cb in self._kline_callbacks:
            try:
                await cb(kline_data)
            except Exception as exc:
                logger.error("Kline callback error: %s", exc)

    async def _on_book_ticker_event(self, data: Dict[str, Any]) -> None:
        """Process a best bid/ask update."""
        symbol = self._from_binance_symbol(data.get("s", ""))

        bid = float(data.get("b", 0))
        ask = float(data.get("a", 0))
        bid_qty = float(data.get("B", 0))
        ask_qty = float(data.get("A", 0))

        self._best_bid[symbol] = bid
        self._best_ask[symbol] = ask
        self._best_bid_qty[symbol] = bid_qty
        self._best_ask_qty[symbol] = ask_qty

        book_data = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "spread": (ask - bid) / ((ask + bid) / 2) if (ask + bid) > 0 else 0,
            "timestamp": time.time(),
        }

        for cb in self._book_ticker_callbacks:
            try:
                await cb(book_data)
            except Exception as exc:
                logger.error("BookTicker callback error: %s", exc)

    async def _on_ticker_event(self, data: Dict[str, Any]) -> None:
        """Process a 24h mini-ticker event."""
        symbol = self._from_binance_symbol(data.get("s", ""))

        ticker_data = {
            "symbol": symbol,
            "close": float(data.get("c", 0)),
            "open": float(data.get("o", 0)),
            "high": float(data.get("h", 0)),
            "low": float(data.get("l", 0)),
            "volume": float(data.get("v", 0)),
            "quote_volume": float(data.get("q", 0)),
        }

        self._last_price[symbol] = ticker_data["close"]

        for cb in self._ticker_callbacks:
            try:
                await cb(ticker_data)
            except Exception as exc:
                logger.error("Ticker callback error: %s", exc)

    async def _on_agg_trade_event(self, data: Dict[str, Any]) -> None:
        """
        Process an aggregated trade event.

        Fields from Binance:
          p: price, q: quantity, m: is_buyer_maker (True = seller aggressor)
        """
        symbol = self._from_binance_symbol(data.get("s", ""))
        qty = float(data.get("q", 0))
        is_buyer_maker = data.get("m", False)

        flow = self._trade_flow.get(symbol)
        if flow is None:
            return

        # Reset window every 60 seconds
        now = time.time()
        if now - flow["window_start"] > 60.0:
            flow["buy_vol"] = 0.0
            flow["sell_vol"] = 0.0
            flow["trade_count"] = 0
            flow["large_trade_count"] = 0
            flow["window_start"] = now

        # is_buyer_maker=True means the buyer placed a limit order and
        # the seller was the taker (aggressor). So the trade is a SELL.
        if is_buyer_maker:
            flow["sell_vol"] += qty
        else:
            flow["buy_vol"] += qty
        flow["trade_count"] += 1

        # Track trade sizes for dynamic large-trade threshold
        sizes = self._trade_sizes.get(symbol)
        if sizes is not None:
            sizes.append(qty)
            # Large trade = above 95th percentile of recent trades
            if len(sizes) >= 50:
                threshold = float(np.percentile(list(sizes), 95))
                if qty >= threshold:
                    flow["large_trade_count"] += 1

        # Fire callbacks
        trade_data = {
            "symbol": symbol,
            "price": float(data.get("p", 0)),
            "quantity": qty,
            "is_sell": is_buyer_maker,
            "timestamp": now,
        }

        for cb in self._agg_trade_callbacks:
            try:
                await cb(trade_data)
            except Exception as exc:
                logger.error("AggTrade callback error: %s", exc)

    # ------------------------------------------------------------------
    # Symbol Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_binance_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' → 'btcusdt' (Binance WS format)."""
        return symbol.replace("/", "").lower()

    @staticmethod
    def _from_binance_symbol(symbol: str) -> str:
        """Convert 'BTCUSDT' → 'BTC/USDT' (ccxt format)."""
        s = symbol.upper()
        # Common quote currencies (order matters — check longer ones first)
        for quote in ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"):
            if s.endswith(quote):
                base = s[: -len(quote)]
                return f"{base}/{quote}"
        return s  # fallback — return as-is

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """Normalise symbol to 'BTC/USDT' format regardless of input."""
        if "/" in symbol:
            return symbol.upper()
        return ConnectionManager._from_binance_symbol(symbol)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return connection status for health checks."""
        return {
            "connected": self.is_connected,
            "running": self._running,
            "reconnect_count": self._reconnect_count,
            "total_messages": self._total_messages,
            "last_message_age_s": round(self.last_message_age, 1),
            "symbols": self._symbols,
            "tracked_prices": {
                sym: self._last_price.get(sym, "N/A")
                for sym in self._symbols
            },
        }
