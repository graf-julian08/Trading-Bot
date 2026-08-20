"""
state_manager.py — Local Order State Machine
=============================================
Maintains a local, in-memory ledger of all orders and positions,
independent of the exchange API.

Key invariants:
  1. The bot ALWAYS knows what it owns — even during API outages.
  2. Order status is updated via WebSocket `executionReport` events
     (sub-second latency), NOT via REST polling.
  3. On startup (or reconnect), the local state is reconciled against
     the exchange via a single REST call. After that, WebSocket-only.

Architecture:
  - `OrderState`: Enum of order lifecycle states.
  - `TrackedOrder`: Dataclass holding all order metadata.
  - `OrderManager`: The state machine that tracks everything.

Usage:
    mgr = OrderManager()
    mgr.register_order("ord-123", "BTC/USDT", "buy", 0.1, 50000.0)
    mgr.on_execution_report({...})  # from WS stream
    print(mgr.get_open_positions())
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from config import cfg

logger = logging.getLogger(__name__)

# ============================================================================
# Order State Enum
# ============================================================================


class OrderState(enum.Enum):
    """Order lifecycle states, matching Binance's execution report states."""
    PENDING = "PENDING"          # Submitted but not confirmed
    NEW = "NEW"                  # Accepted by exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"            # Fully executed
    CANCELLED = "CANCELLED"      # Cancelled by user or exchange
    REJECTED = "REJECTED"        # Rejected by exchange
    EXPIRED = "EXPIRED"          # Time-in-force expired
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Tracked Order Dataclass
# ============================================================================


@dataclass
class TrackedOrder:
    """Immutable-ish record of a single order in the local ledger."""
    order_id: str
    symbol: str
    side: str                     # "buy" or "sell"
    order_type: str               # "limit", "market", "limit_ioc"
    requested_amount: float
    requested_price: Optional[float]  # None for market orders
    state: OrderState = OrderState.PENDING
    filled_amount: float = 0.0
    avg_fill_price: float = 0.0
    cumulative_cost: float = 0.0
    fee: float = 0.0
    trade_id: Optional[int] = None  # DB trade ID (set after logging)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    exchange_order_id: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.state in (
            OrderState.PENDING,
            OrderState.NEW,
            OrderState.PARTIALLY_FILLED,
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        )

    @property
    def remaining_amount(self) -> float:
        return max(0.0, self.requested_amount - self.filled_amount)


# ============================================================================
# Position Tracker
# ============================================================================


@dataclass
class Position:
    """Aggregated position for a symbol (sum of all fills)."""
    symbol: str
    side: str                     # "long" or "short" (or "flat")
    amount: float = 0.0
    avg_entry_price: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    entry_time: float = 0.0
    trade_id: Optional[int] = None  # DB trade ID


# ============================================================================
# Order Manager (State Machine)
# ============================================================================

# Callback for state transitions: async def handler(order: TrackedOrder) -> None
OrderCallback = Callable[[TrackedOrder], Coroutine[Any, Any, None]]


class OrderManager:
    """
    Local order state machine.

    Tracks every order's lifecycle without depending on REST API calls.
    Feeds from WebSocket executionReport events for real-time updates.
    """

    def __init__(self) -> None:
        # Order ledger: order_id -> TrackedOrder
        self._orders: Dict[str, TrackedOrder] = {}

        # Active positions: symbol -> Position
        self._positions: Dict[str, Position] = {}

        # Callbacks for state transitions
        self._on_fill_callbacks: List[OrderCallback] = []
        self._on_cancel_callbacks: List[OrderCallback] = []

        # Reconciliation flag
        self._reconciled = False
        self._last_reconciliation_time: float = 0.0

    # ------------------------------------------------------------------
    # Callback Registration
    # ------------------------------------------------------------------

    def on_fill(self, callback: OrderCallback) -> None:
        """Register callback for when an order is fully filled."""
        self._on_fill_callbacks.append(callback)

    def on_cancel(self, callback: OrderCallback) -> None:
        """Register callback for when an order is cancelled/rejected."""
        self._on_cancel_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Order Registration (before or after placement)
    # ------------------------------------------------------------------

    def register_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        trade_id: Optional[int] = None,
    ) -> TrackedOrder:
        """
        Register a new order in the local ledger.

        Call this BEFORE placing the order on the exchange, so the local
        state always has the order even if the API call fails/timeouts.
        """
        order = TrackedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side.lower(),
            order_type=order_type,
            requested_amount=amount,
            requested_price=price,
            trade_id=trade_id,
        )
        self._orders[order_id] = order
        logger.info(
            "📋 Registered order %s: %s %s %.8f %s @ %s",
            order_id, side, amount, amount, symbol,
            f"{price:.8g}" if price else "MARKET",
        )
        return order

    # ------------------------------------------------------------------
    # WebSocket Event Processing
    # ------------------------------------------------------------------

    async def on_execution_report(self, data: Dict[str, Any]) -> None:
        """
        Process a Binance executionReport event from the WebSocket.

        Binance executionReport fields:
          - i: orderId
          - s: symbol (BTCUSDT)
          - S: side (BUY/SELL)
          - o: order type (LIMIT, MARKET, etc.)
          - X: order status (NEW, PARTIALLY_FILLED, FILLED, CANCELED, etc.)
          - l: last filled quantity (this execution)
          - L: last filled price (this execution)
          - z: cumulative filled quantity
          - Z: cumulative quote quantity (cost)
          - n: commission amount
          - T: transaction time
        """
        order_id = str(data.get("i", ""))
        status_str = data.get("X", "").upper()
        symbol = self._from_binance_symbol(data.get("s", ""))
        side = data.get("S", "").lower()

        # Map Binance status to our enum
        status_map = {
            "NEW": OrderState.NEW,
            "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
            "FILLED": OrderState.FILLED,
            "CANCELED": OrderState.CANCELLED,
            "CANCELLED": OrderState.CANCELLED,
            "REJECTED": OrderState.REJECTED,
            "EXPIRED": OrderState.EXPIRED,
        }
        new_state = status_map.get(status_str, OrderState.UNKNOWN)

        # Find or create the order in our ledger
        order = self._orders.get(order_id)
        if order is None:
            # Order was placed outside our system (manual trade, or pre-startup)
            order = TrackedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=data.get("o", "unknown").lower(),
                requested_amount=float(data.get("q", 0)),
                requested_price=float(data.get("p", 0)) or None,
            )
            self._orders[order_id] = order
            logger.info("📋 Auto-registered unknown order %s", order_id)

        # Update state
        old_state = order.state
        order.state = new_state
        order.filled_amount = float(data.get("z", order.filled_amount))
        order.cumulative_cost = float(data.get("Z", order.cumulative_cost))
        order.fee += float(data.get("n", 0))
        order.updated_at = time.time()

        if order.filled_amount > 0:
            order.avg_fill_price = order.cumulative_cost / order.filled_amount

        # Log state transitions
        if old_state != new_state:
            logger.info(
                "📋 Order %s: %s → %s (filled=%.8f/%s, avg_price=%.8g)",
                order_id, old_state.value, new_state.value,
                order.filled_amount, order.requested_amount,
                order.avg_fill_price,
            )

        # Update position tracking
        if new_state == OrderState.FILLED:
            self._update_position_on_fill(order)
            for cb in self._on_fill_callbacks:
                try:
                    await cb(order)
                except Exception as exc:
                    logger.error("Fill callback error: %s", exc)

        elif new_state in (OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED):
            for cb in self._on_cancel_callbacks:
                try:
                    await cb(order)
                except Exception as exc:
                    logger.error("Cancel callback error: %s", exc)

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def _update_position_on_fill(self, order: TrackedOrder) -> None:
        """Update the position ledger when an order is fully filled."""
        symbol = order.symbol
        pos = self._positions.get(symbol)

        if order.side == "buy":
            if pos is None or pos.side == "flat":
                # Opening a new long position
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side="long",
                    amount=order.filled_amount,
                    avg_entry_price=order.avg_fill_price,
                    entry_time=order.updated_at,
                    trade_id=order.trade_id,
                )
            elif pos.side == "long":
                # Adding to long position
                total = pos.amount + order.filled_amount
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.amount +
                     order.avg_fill_price * order.filled_amount) / total
                )
                pos.amount = total
            elif pos.side == "short":
                # Reducing/closing short position
                pos.amount -= order.filled_amount
                if pos.amount <= 0:
                    pnl = (pos.avg_entry_price - order.avg_fill_price) * order.filled_amount
                    pos.realised_pnl += pnl
                    pos.side = "flat"
                    pos.amount = 0.0

        elif order.side == "sell":
            if pos is None or pos.side == "flat":
                # Opening a new short position (or just exiting completely)
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side="flat",
                    amount=0.0,
                    avg_entry_price=0.0,
                    entry_time=order.updated_at,
                    trade_id=order.trade_id,
                )
            elif pos.side == "long":
                # Reducing/closing long position
                close_amount = min(order.filled_amount, pos.amount)
                pnl = (order.avg_fill_price - pos.avg_entry_price) * close_amount
                pos.realised_pnl += pnl
                pos.amount -= close_amount
                if pos.amount <= 1e-10:
                    pos.side = "flat"
                    pos.amount = 0.0

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get the current position for a symbol."""
        return self._positions.get(symbol)

    def get_open_positions(self) -> Dict[str, Position]:
        """Get all non-flat positions."""
        return {
            sym: pos for sym, pos in self._positions.items()
            if pos.side != "flat" and pos.amount > 0
        }

    def has_position(self, symbol: str) -> bool:
        """Check if we have an active position in this symbol."""
        pos = self._positions.get(symbol)
        return pos is not None and pos.side != "flat" and pos.amount > 0

    def get_position_amount(self, symbol: str) -> float:
        """Get the current position size for a symbol (0 if flat)."""
        pos = self._positions.get(symbol)
        if pos and pos.side != "flat":
            return pos.amount
        return 0.0

    # ------------------------------------------------------------------
    # Order Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        """Retrieve a tracked order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        """Get all orders that are still open (NEW or PARTIALLY_FILLED)."""
        result = [o for o in self._orders.values() if o.is_open]
        if symbol:
            result = [o for o in result if o.symbol == symbol]
        return result

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        """Get orders in PENDING state (submitted but not yet confirmed)."""
        result = [
            o for o in self._orders.values()
            if o.state == OrderState.PENDING
        ]
        if symbol:
            result = [o for o in result if o.symbol == symbol]
        return result

    # ------------------------------------------------------------------
    # REST Reconciliation (startup / reconnect)
    # ------------------------------------------------------------------

    async def reconcile_with_exchange(self, exchange) -> None:
        """
        One-time REST sync to align local state with exchange reality.

        Called on startup and after WebSocket reconnects. After this,
        the local state machine relies exclusively on WS events.

        Parameters
        ----------
        exchange : ccxt exchange instance
            Used to fetch open orders and account balances.
        """
        logger.info("🔄 Reconciling local state with exchange...")

        try:
            # Fetch all open orders from exchange
            for symbol in cfg.TRADING_PAIRS:
                try:
                    open_orders = await exchange.fetch_open_orders(symbol)
                    for raw_order in open_orders:
                        oid = str(raw_order.get("id", ""))
                        if oid not in self._orders:
                            # Register unknown order
                            side = raw_order.get("side", "unknown")
                            filled = float(raw_order.get("filled", 0))
                            amount = float(raw_order.get("amount", 0))

                            order = TrackedOrder(
                                order_id=oid,
                                symbol=symbol,
                                side=side,
                                order_type=raw_order.get("type", "unknown"),
                                requested_amount=amount,
                                requested_price=raw_order.get("price"),
                                state=OrderState.NEW if filled == 0 else OrderState.PARTIALLY_FILLED,
                                filled_amount=filled,
                                avg_fill_price=float(raw_order.get("average", 0)),
                                exchange_order_id=oid,
                            )
                            self._orders[oid] = order
                            logger.info(
                                "📋 Reconciled order %s: %s %s %.8f %s",
                                oid, side, amount, amount, symbol,
                            )
                except Exception as exc:
                    logger.warning("Failed to reconcile orders for %s: %s", symbol, exc)

            self._reconciled = True
            self._last_reconciliation_time = time.time()
            logger.info(
                "✅ Reconciliation complete — %d orders tracked, %d positions.",
                len(self._orders),
                len(self.get_open_positions()),
            )

        except Exception as exc:
            logger.error("❌ Reconciliation failed: %s", exc)
            # Non-fatal — we continue with what we have

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_orders(self, max_age_hours: float = 24.0) -> int:
        """Remove terminal orders older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = [
            oid for oid, order in self._orders.items()
            if order.is_terminal and order.updated_at < cutoff
        ]
        for oid in to_remove:
            del self._orders[oid]
        if to_remove:
            logger.info("🧹 Cleaned up %d old orders.", len(to_remove))
        return len(to_remove)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return state manager health for monitoring."""
        return {
            "reconciled": self._reconciled,
            "last_reconciliation_age_s": (
                round(time.time() - self._last_reconciliation_time, 1)
                if self._last_reconciliation_time > 0 else "never"
            ),
            "total_orders_tracked": len(self._orders),
            "open_orders": len(self.get_open_orders()),
            "open_positions": {
                sym: {
                    "side": pos.side,
                    "amount": pos.amount,
                    "avg_entry": pos.avg_entry_price,
                    "realised_pnl": pos.realised_pnl,
                }
                for sym, pos in self.get_open_positions().items()
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_binance_symbol(symbol: str) -> str:
        """Convert 'BTCUSDT' → 'BTC/USDT'."""
        s = symbol.upper()
        for quote in ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"):
            if s.endswith(quote):
                base = s[: -len(quote)]
                return f"{base}/{quote}"
        return s
