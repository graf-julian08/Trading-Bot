"""
tests_phase1.py — Hostile Self-Audit for Phase 1 Critical Fixes
=================================================================
Tests the 3 "wallet-draining" bug fixes:
  1. Circuit Breaker is alive (monitor_open_positions returns data)
  2. Paper Balance tracks dynamically (not static 10k)
  3. PnL subtracts BOTH entry and exit fees

Run:
    cd /path/to/Trading-Bot
    python -m pytest tests_phase1.py -v
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock infrastructure — we don't need a real DB or exchange
# ---------------------------------------------------------------------------

class MockDatabase:
    """Minimal mock that stores trades in-memory."""

    def __init__(self):
        self._trades: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    async def initialise(self):
        pass

    async def log_trade(self, symbol, side, order_type, price, amount,
                        cost, fee=0.0, stop_loss=None, take_profit=None,
                        ml_probability=0.0, metadata=None) -> int:
        tid = self._next_id
        self._next_id += 1
        self._trades[tid] = {
            "id": tid, "symbol": symbol, "side": side,
            "order_type": order_type, "price": price, "amount": amount,
            "cost": cost, "fee": fee,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "status": "open", "pnl": 0.0,
            "ml_probability": ml_probability,
        }
        return tid

    async def close_trade(self, trade_id, exit_price, pnl, fee=0.0):
        if trade_id in self._trades:
            self._trades[trade_id]["status"] = "closed"
            self._trades[trade_id]["pnl"] = pnl
            self._trades[trade_id]["fee"] = fee

    async def get_open_trades(self, symbol=None) -> List[Dict]:
        trades = [t for t in self._trades.values() if t["status"] == "open"]
        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol]
        return trades

    async def get_trade_by_id(self, trade_id) -> Optional[Dict]:
        return self._trades.get(trade_id)

    async def count_open_trades(self) -> int:
        return sum(1 for t in self._trades.values() if t["status"] == "open")


class MockNotifier:
    """Swallows all notification calls silently."""

    async def send_trade_opened(self, **kw): return True
    async def send_trade_closed(self, **kw): return True
    async def send_kill_switch_alert(self, **kw): return True
    async def send_error_alert(self, msg): return True


class MockDataEngine:
    """Returns controllable ticker prices."""

    def __init__(self, price: float = 50000.0):
        self._price = price

    def set_price(self, price: float):
        self._price = price

    async def fetch_ticker(self, symbol):
        return {"last": self._price, "bid": self._price - 1, "ask": self._price + 1}

    async def fetch_order_book(self, symbol, limit=10):
        return {
            "bids": [[self._price - 1, 10]],
            "asks": [[self._price + 1, 10]],
        }

    async def fetch_balance(self):
        return {"free": {"USDT": 100000, "BTC": 1.0}, "total": {"USDT": 100000}}

    @property
    def exchange(self):
        return self


# Patch cfg.PAPER_TRADE to True for all tests
import config
original_cfg = config.cfg


# ===========================================================================
# Test 1: Circuit Breaker — monitor_open_positions returns closed trades
# ===========================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_returns_closed_trades():
    """
    CRITICAL: monitor_open_positions() MUST return a list of dicts
    when trades are closed, NOT None.  This is what feeds the
    circuit breaker in main.py.
    """
    from execution_engine import ExecutionEngine

    db = MockDatabase()
    data = MockDataEngine(price=50000.0)
    notifier = MockNotifier()

    engine = ExecutionEngine(data_engine=data, db=db, notifier=notifier)

    # Open a trade manually in the DB with a stop-loss at 49000
    trade_id = await db.log_trade(
        symbol="BTC/USDT", side="buy", order_type="limit",
        price=50000.0, amount=0.1, cost=5000.0, fee=5.0,  # $5 entry fee
        stop_loss=49000.0, take_profit=52000.0,
    )

    assert trade_id == 1
    assert await db.count_open_trades() == 1

    # Simulate price crashing to 48500 (below stop-loss of 49000)
    data.set_price(48500.0)

    # Call monitor — this should close the trade and RETURN the result
    closed = await engine.monitor_open_positions()

    # CRITICAL ASSERTIONS
    assert isinstance(closed, list), \
        "monitor_open_positions must return a LIST, not None"
    assert len(closed) == 1, \
        f"Expected 1 closed trade, got {len(closed)}"

    result = closed[0]
    assert result["trade_id"] == trade_id
    assert result["symbol"] == "BTC/USDT"
    assert result["side"] == "buy"
    assert result["reason"] == "stop_loss"
    assert result["pnl"] < 0, \
        f"Stop-loss trade must have negative PnL, got {result['pnl']}"

    # Verify the trade is actually closed in the DB
    assert await db.count_open_trades() == 0


@pytest.mark.asyncio
async def test_circuit_breaker_empty_when_no_closes():
    """
    When no trades are closed, monitor_open_positions MUST return
    an empty list [], NOT None.
    """
    from execution_engine import ExecutionEngine

    db = MockDatabase()
    data = MockDataEngine(price=50000.0)
    notifier = MockNotifier()

    engine = ExecutionEngine(data_engine=data, db=db, notifier=notifier)

    # Open a trade with SL at 49000, price is still 50000 — no trigger
    await db.log_trade(
        symbol="BTC/USDT", side="buy", order_type="limit",
        price=50000.0, amount=0.1, cost=5000.0, fee=5.0,
        stop_loss=49000.0, take_profit=52000.0,
    )

    closed = await engine.monitor_open_positions()

    assert isinstance(closed, list), \
        "Must return a list even when no trades are closed"
    assert len(closed) == 0


# ===========================================================================
# Test 2: Paper Balance — dynamic equity tracking
# ===========================================================================

def test_paper_balance_debits_on_open():
    """
    When a paper trade is opened, the balance MUST decrease by
    cost + fee.  It must NOT stay at 10,000 forever.
    """
    # Import here to get the class
    sys.path.insert(0, ".")
    from main import PaperBalanceTracker

    tracker = PaperBalanceTracker(initial_balance=10_000.0)
    assert tracker.balance == 10_000.0

    # Simulate opening a trade: buy 0.1 BTC at $50,000 = $5,000 cost + $5 fee
    tracker.on_trade_opened(cost=5000.0, fee=5.0)

    assert tracker.balance == pytest.approx(10_000.0 - 5000.0 - 5.0)
    assert tracker.balance == pytest.approx(4995.0)


def test_paper_balance_credits_on_close_profit():
    """
    When a profitable paper trade is closed, the balance must reflect
    the actual profit.
    """
    from main import PaperBalanceTracker

    tracker = PaperBalanceTracker(initial_balance=10_000.0)

    # Open: buy 0.1 BTC at $50,000 = $5,000 cost + $5 fee
    tracker.on_trade_opened(cost=5000.0, fee=5.0)
    assert tracker.balance == pytest.approx(4995.0)

    # Close: sell 0.1 BTC at $51,000 = $5,100 proceeds - $5.1 fee
    tracker.on_trade_closed(
        side="buy",
        entry_price=50000.0,
        exit_price=51000.0,
        amount=0.1,
        entry_cost=5000.0,
        exit_fee=5.1,
    )

    # Expected: 4995.0 + (51000 * 0.1 - 5.1) = 4995.0 + 5094.9 = 10089.9
    assert tracker.balance == pytest.approx(10089.9)
    assert tracker.balance > 10_000.0, \
        "Profitable trade must increase balance above initial"


def test_paper_balance_credits_on_close_loss():
    """
    When a losing paper trade is closed, the balance must decrease
    AND be below initial capital.
    """
    from main import PaperBalanceTracker

    tracker = PaperBalanceTracker(initial_balance=10_000.0)

    # Open: buy 0.1 BTC at $50,000 = $5,000 cost + $5 fee
    tracker.on_trade_opened(cost=5000.0, fee=5.0)

    # Close at loss: sell 0.1 BTC at $49,000 = $4,900 proceeds - $4.9 fee
    tracker.on_trade_closed(
        side="buy",
        entry_price=50000.0,
        exit_price=49000.0,
        amount=0.1,
        entry_cost=5000.0,
        exit_fee=4.9,
    )

    # Expected: 4995.0 + (49000 * 0.1 - 4.9) = 4995.0 + 4895.1 = 9890.1
    assert tracker.balance == pytest.approx(9890.1)
    assert tracker.balance < 10_000.0, \
        "Losing trade must decrease balance below initial"


def test_paper_balance_kills_switch_possible():
    """
    The kill switch requires equity to drop >= 5%.
    With static 10k this was impossible.
    Verify it's now possible with dynamic tracking.
    """
    from main import PaperBalanceTracker

    tracker = PaperBalanceTracker(initial_balance=10_000.0)

    # Simulate multiple large losses
    for _ in range(5):
        tracker.on_trade_opened(cost=2000.0, fee=2.0)
        tracker.on_trade_closed(
            side="buy",
            entry_price=50000.0,
            exit_price=47500.0,  # -5% per trade on the position
            amount=0.04,
            entry_cost=2000.0,
            exit_fee=1.9,
        )

    # Balance should be significantly below initial
    pct_change = (tracker.balance - 10_000.0) / 10_000.0
    assert pct_change < -0.05, \
        f"Balance should drop >5% for kill switch, actual: {pct_change:.2%}"


# ===========================================================================
# Test 3: PnL subtracts BOTH entry and exit fees (no phantom profits)
# ===========================================================================

@pytest.mark.asyncio
async def test_pnl_subtracts_both_fees():
    """
    CRITICAL: PnL = (exit - entry) × amount - entry_fee - exit_fee
    Previously only exit_fee was subtracted, creating phantom profits.
    """
    from execution_engine import ExecutionEngine

    db = MockDatabase()
    data = MockDataEngine(price=50000.0)
    notifier = MockNotifier()

    engine = ExecutionEngine(data_engine=data, db=db, notifier=notifier)

    # Open trade with $5.00 entry fee
    entry_fee = 5.0
    trade_id = await db.log_trade(
        symbol="BTC/USDT", side="buy", order_type="limit",
        price=50000.0, amount=0.1, cost=5000.0, fee=entry_fee,
        stop_loss=49000.0, take_profit=52000.0,
    )

    # Close at exactly the same price (should be a LOSS due to fees)
    data.set_price(50000.0)
    result = await engine.close_trade(trade_id, exit_price=50000.0, reason="manual")

    assert result is not None
    pnl = result["pnl"]

    # exit_fee from paper simulation: 50000 * 0.1 * 0.001 = 5.0 (0.1% fee)
    expected_exit_fee = 50000.0 * 0.1 * 0.001  # 5.0
    expected_total_fees = entry_fee + expected_exit_fee  # 10.0
    # Price didn't move, so gross PnL = 0.  Net PnL = -total_fees
    expected_pnl = 0.0 - expected_total_fees  # -10.0

    assert pnl == pytest.approx(expected_pnl, abs=0.01), \
        f"PnL should be {expected_pnl} (entry_fee={entry_fee} + exit_fee={expected_exit_fee}), got {pnl}"
    assert pnl < 0, \
        "Flat trade MUST show a loss due to fees — this was the phantom profit bug"


@pytest.mark.asyncio
async def test_pnl_profit_still_works():
    """
    Verify that a genuinely profitable trade still shows profit
    (just reduced by fees).
    """
    from execution_engine import ExecutionEngine

    db = MockDatabase()
    data = MockDataEngine(price=50000.0)
    notifier = MockNotifier()

    engine = ExecutionEngine(data_engine=data, db=db, notifier=notifier)

    entry_fee = 5.0
    trade_id = await db.log_trade(
        symbol="BTC/USDT", side="buy", order_type="limit",
        price=50000.0, amount=0.1, cost=5000.0, fee=entry_fee,
        stop_loss=49000.0, take_profit=52000.0,
    )

    # Close at $51,000 — a $100 gross profit on 0.1 BTC
    data.set_price(51000.0)
    result = await engine.close_trade(trade_id, exit_price=51000.0, reason="take_profit")

    assert result is not None
    pnl = result["pnl"]

    # Gross PnL: (51000 - 50000) * 0.1 = 100.0
    # Exit fee: 51000 * 0.1 * 0.001 = 5.1  (paper sim)
    # Total fees: 5.0 (entry) + 5.1 (exit) = 10.1
    # Net PnL: 100.0 - 10.1 = 89.9
    gross = (51000 - 50000) * 0.1
    exit_fee_expected = 51000 * 0.1 * 0.001
    expected_net = gross - entry_fee - exit_fee_expected

    assert pnl == pytest.approx(expected_net, abs=0.01), \
        f"Expected net PnL {expected_net}, got {pnl}"
    assert pnl > 0, "A $100 gross profit minus $10 fees should still be positive"


@pytest.mark.asyncio
async def test_close_trade_returns_complete_dict():
    """
    Verify close_trade() returns a complete dict with all required
    fields for the circuit breaker and paper balance tracker.
    """
    from execution_engine import ExecutionEngine

    db = MockDatabase()
    data = MockDataEngine(price=50000.0)
    notifier = MockNotifier()

    engine = ExecutionEngine(data_engine=data, db=db, notifier=notifier)

    trade_id = await db.log_trade(
        symbol="ETH/USDT", side="buy", order_type="limit",
        price=3000.0, amount=1.0, cost=3000.0, fee=3.0,
        stop_loss=2900.0, take_profit=3200.0,
    )

    data.set_price(3200.0)
    result = await engine.close_trade(trade_id, exit_price=3200.0, reason="take_profit")

    # Verify all required fields exist
    required_keys = [
        "trade_id", "symbol", "side", "entry_price", "exit_price",
        "amount", "pnl", "pnl_pct", "reason", "entry_fee", "exit_fee",
    ]
    for key in required_keys:
        assert key in result, f"Missing required key '{key}' in close_trade result"

    assert result["trade_id"] == trade_id
    assert result["symbol"] == "ETH/USDT"
    assert result["reason"] == "take_profit"
    assert result["entry_fee"] == 3.0
    assert result["exit_fee"] > 0
