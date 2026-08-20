"""
tests_iron_dome.py — Iron Dome Safety Verification Suite
=========================================================
Phase 1 of Operation Alpha & Omega.

Tests the 3 critical safety fixes identified in the Hell Week Audit:
  1. Fat-Finger Clamp: extreme config values are clamped to safe ranges
  2. Dead-Man's Switch: failed exits trigger emergency loop + Telegram alert
  3. Partial Fill Logic: order['filled'] is handled correctly (even when 0)

Run with:
    python -m pytest tests_iron_dome.py -v

If ANY of these tests fail, the bot MUST NOT go live.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Test 1: Fat-Finger Clamp — Config validation
# ===========================================================================


class TestFatFingerClamp:
    """
    Verify that extreme configuration values are silently clamped to
    safe ranges by Config._clamp_risk_parameters().

    If ANY of these fail, a single .env typo can nuke the account.
    """

    def _make_config(self, overrides: dict):
        """
        Create a Config instance with specific field overrides.
        We patch the .env check so it doesn't crash in CI.
        """
        import dataclasses
        from pathlib import Path
        # Ensure .env exists for the check
        env_path = Path(__file__).resolve().parent / ".env"
        env_existed = env_path.exists()
        if not env_existed:
            env_path.write_text("PAPER_TRADE=true\n")

        try:
            from config import Config
            # Build kwargs from dataclass field defaults
            kwargs = {}
            for f in dataclasses.fields(Config):
                if f.default is not dataclasses.MISSING:
                    kwargs[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    kwargs[f.name] = f.default_factory()
                # else: field has no default — skip (will be overridden)

            kwargs.update(overrides)
            kwargs["PAPER_TRADE"] = True  # Never allow live in tests
            return Config(**kwargs)
        finally:
            if not env_existed:
                env_path.unlink(missing_ok=True)

    # ---- MAX_RISK_PER_TRADE ----

    def test_risk_per_trade_clamped_high(self):
        """
        User sets MAX_RISK_PER_TRADE = 100 (10,000%).
        Must be clamped to 0.05 (5%).
        """
        cfg = self._make_config({"MAX_RISK_PER_TRADE": 100.0})
        assert cfg.MAX_RISK_PER_TRADE == 0.05, (
            f"MAX_RISK_PER_TRADE should be 0.05, got {cfg.MAX_RISK_PER_TRADE}"
        )

    def test_risk_per_trade_clamped_low(self):
        """
        User sets MAX_RISK_PER_TRADE = 0 (0%).
        Must be clamped to 0.001 (0.1%).
        """
        cfg = self._make_config({"MAX_RISK_PER_TRADE": 0.0})
        assert cfg.MAX_RISK_PER_TRADE == 0.001

    def test_risk_per_trade_normal_value_not_clamped(self):
        """
        User sets MAX_RISK_PER_TRADE = 0.02 (2%) — a normal value.
        Must NOT be clamped.
        """
        cfg = self._make_config({"MAX_RISK_PER_TRADE": 0.02})
        assert cfg.MAX_RISK_PER_TRADE == 0.02

    # ---- STOP_LOSS_PCT ----

    def test_stop_loss_clamped_to_max_20_percent(self):
        """STOP_LOSS_PCT = 0.50 → must clamp to 0.20."""
        cfg = self._make_config({"STOP_LOSS_PCT": 0.50})
        assert cfg.STOP_LOSS_PCT == 0.20

    def test_stop_loss_clamped_to_min(self):
        """STOP_LOSS_PCT = 0.0 → must clamp to 0.001."""
        cfg = self._make_config({"STOP_LOSS_PCT": 0.0})
        assert cfg.STOP_LOSS_PCT == 0.001

    # ---- TAKE_PROFIT_PCT ----

    def test_take_profit_clamped_to_max_50_percent(self):
        """TAKE_PROFIT_PCT = 1.0 → must clamp to 0.50."""
        cfg = self._make_config({"TAKE_PROFIT_PCT": 1.0})
        assert cfg.TAKE_PROFIT_PCT == 0.50

    # ---- DAILY_DRAWDOWN_LIMIT ----

    def test_daily_drawdown_must_be_negative(self):
        """DAILY_DRAWDOWN_LIMIT = 0.10 (positive!). Must clamp to -0.01."""
        cfg = self._make_config({"DAILY_DRAWDOWN_LIMIT": 0.10})
        assert cfg.DAILY_DRAWDOWN_LIMIT == -0.01

    def test_daily_drawdown_extreme_clamp(self):
        """DAILY_DRAWDOWN_LIMIT = -0.90 → clamped to -0.50."""
        cfg = self._make_config({"DAILY_DRAWDOWN_LIMIT": -0.90})
        assert cfg.DAILY_DRAWDOWN_LIMIT == -0.50

    # ---- CIRCUIT_BREAKER_LOSSES ----

    def test_circuit_breaker_min_1(self):
        """CIRCUIT_BREAKER_LOSSES = 0 → clamped to 1."""
        cfg = self._make_config({"CIRCUIT_BREAKER_LOSSES": 0})
        assert cfg.CIRCUIT_BREAKER_LOSSES == 1

    def test_circuit_breaker_max_20(self):
        """CIRCUIT_BREAKER_LOSSES = 999 → clamped to 20."""
        cfg = self._make_config({"CIRCUIT_BREAKER_LOSSES": 999})
        assert cfg.CIRCUIT_BREAKER_LOSSES == 20

    # ---- MAX_OPEN_POSITIONS ----

    def test_max_open_positions_max_10(self):
        """MAX_OPEN_POSITIONS = 100 → clamped to 10."""
        cfg = self._make_config({"MAX_OPEN_POSITIONS": 100})
        assert cfg.MAX_OPEN_POSITIONS == 10


# ===========================================================================
# Test 2: Dead-Man's Switch — Emergency exit escalation
# ===========================================================================


class TestDeadMansSwitch:
    """
    Verify the emergency exit loop fires when all normal exits fail,
    and sends a Telegram alert when ALL retries are exhausted.
    """

    def _make_engine(self, notifier=None):
        """Create an ExecutionEngine with mocked dependencies."""
        from execution_engine import ExecutionEngine

        mock_data = MagicMock()
        mock_db = AsyncMock()
        mock_notifier = notifier or AsyncMock()
        return ExecutionEngine(mock_data, mock_db, mock_notifier), mock_notifier

    @pytest.mark.asyncio
    async def test_emergency_loop_succeeds_on_third_retry(self):
        """
        _place_order fails twice, succeeds on 3rd attempt.
        The emergency loop should return the successful result.
        """
        engine, notifier = self._make_engine()

        # First 2 calls fail, 3rd succeeds
        call_count = 0
        async def mock_place_order(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None
            return {"id": "test123", "type": "market", "price": 50000.0,
                    "amount": 0.1, "cost": 5000.0, "fee": 5.0}

        engine._place_order = mock_place_order

        result = await engine._emergency_exit_loop(
            symbol="BTC/USDT", side="sell", amount=0.1,
            trade_id=42, entry_price=50000.0, original_side="buy",
        )

        assert result is not None
        assert result["id"] == "test123"
        assert call_count == 3
        # Should NOT have sent a Telegram alert (succeeded before exhaustion)
        notifier.send_emergency_exit_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_loop_exhausted_sends_telegram(self):
        """
        All 12 retries fail. Must send a Telegram CRITICAL alert.
        """
        engine, notifier = self._make_engine()

        # Override max retries to 3 for faster tests
        engine._EMERGENCY_MAX_RETRIES = 3
        engine._EMERGENCY_RETRY_DELAY = 0.01  # 10ms instead of 5s

        # All calls fail
        async def always_fail(**kwargs):
            return None

        engine._place_order = always_fail

        result = await engine._emergency_exit_loop(
            symbol="BTC/USDT", side="sell", amount=0.1,
            trade_id=42, entry_price=50000.0, original_side="buy",
        )

        assert result is None
        # MUST have sent the alert
        notifier.send_emergency_exit_alert.assert_called_once()
        call_args = notifier.send_emergency_exit_alert.call_args
        assert call_args.kwargs["symbol"] == "BTC/USDT"
        assert call_args.kwargs["trade_id"] == 42
        assert call_args.kwargs["attempts"] == 3

    @pytest.mark.asyncio
    async def test_close_trade_triggers_emergency_on_double_failure(self):
        """
        close_trade: limit fails → market fallback fails →
        emergency loop must be invoked.
        """
        engine, notifier = self._make_engine()

        # Simulate DB returning an open trade
        engine._db.get_trade_by_id.return_value = {
            "id": 1, "symbol": "BTC/USDT", "side": "buy",
            "amount": 0.1, "price": 50000.0, "fee": 5.0,
            "status": "open", "ml_probability": 0.8,
        }

        # Track calls to _place_order
        place_order_calls = []
        async def mock_place_order(symbol, side, amount, price=None, order_type=None):
            place_order_calls.append(order_type or ("limit" if price else "market"))
            return None  # ALL attempts fail

        engine._place_order = mock_place_order

        # Emergency loop should also fail
        emergency_called = False
        async def mock_emergency(**kwargs):
            nonlocal emergency_called
            emergency_called = True
            return None  # All retries exhausted

        engine._emergency_exit_loop = mock_emergency

        result = await engine.close_trade(1, 49000.0, "stop_loss")

        # The emergency loop should have been called
        assert emergency_called, "Emergency exit loop was NOT triggered!"


# ===========================================================================
# Test 3: Partial Fill Logic
# ===========================================================================


class TestPartialFillLogic:
    """
    Verify that _place_order correctly handles partial fills and
    the edge case where filled=0 (falsy in Python).
    """

    def _make_engine(self):
        from execution_engine import ExecutionEngine

        mock_data = MagicMock()
        mock_db = AsyncMock()
        mock_notifier = AsyncMock()
        engine = ExecutionEngine(mock_data, mock_db, mock_notifier)
        return engine

    @pytest.mark.asyncio
    async def test_partial_fill_uses_filled_amount(self):
        """
        Exchange returns filled=0.5 for an order of amount=1.0.
        Engine must record 0.5, not 1.0.
        """
        engine = self._make_engine()

        # Mock the exchange to return a partial fill
        mock_exchange = AsyncMock()
        mock_exchange.create_market_order.return_value = {
            "id": "ord-001",
            "type": "market",
            "average": 50000.0,
            "price": 50000.0,
            "filled": 0.5,      # Only half filled!
            "amount": 1.0,      # Full intended amount
            "cost": 25000.0,
            "fee": {"cost": 25.0},
        }
        engine._data.exchange = mock_exchange

        # Disable paper mode for this test
        with patch("execution_engine.cfg") as mock_cfg:
            mock_cfg.PAPER_TRADE = False

            result = await engine._place_order(
                symbol="BTC/USDT", side="buy", amount=1.0,
                price=None, order_type="market",
            )

        assert result is not None
        assert result["amount"] == 0.5, (
            f"Expected filled amount 0.5, got {result['amount']}"
        )

    @pytest.mark.asyncio
    async def test_filled_zero_falls_back_to_amount(self):
        """
        Exchange returns filled=0 (unfilled order).
        Engine must fall back to order['amount'], NOT use 0.
        """
        engine = self._make_engine()

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order.return_value = {
            "id": "ord-002",
            "type": "market",
            "average": 50000.0,
            "price": 50000.0,
            "filled": 0,        # Zero — falsy in Python!
            "amount": 1.0,
            "cost": 50000.0,
            "fee": {"cost": 50.0},
        }
        engine._data.exchange = mock_exchange

        with patch("execution_engine.cfg") as mock_cfg:
            mock_cfg.PAPER_TRADE = False

            result = await engine._place_order(
                symbol="BTC/USDT", side="buy", amount=1.0,
                price=None, order_type="market",
            )

        assert result is not None
        assert result["amount"] == 1.0, (
            f"Expected fallback amount 1.0 when filled=0, got {result['amount']}"
        )

    @pytest.mark.asyncio
    async def test_filled_none_falls_back_to_amount(self):
        """
        Exchange doesn't include 'filled' key at all.
        Engine must fall back to order['amount'].
        """
        engine = self._make_engine()

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order.return_value = {
            "id": "ord-003",
            "type": "market",
            "average": 50000.0,
            "amount": 1.0,
            "cost": 50000.0,
            "fee": {"cost": 50.0},
            # No 'filled' key!
        }
        engine._data.exchange = mock_exchange

        with patch("execution_engine.cfg") as mock_cfg:
            mock_cfg.PAPER_TRADE = False

            result = await engine._place_order(
                symbol="BTC/USDT", side="buy", amount=1.0,
                price=None, order_type="market",
            )

        assert result is not None
        assert result["amount"] == 1.0

    @pytest.mark.asyncio
    async def test_full_fill_uses_filled(self):
        """
        Exchange returns filled=1.0, amount=1.0.
        Engine must use 1.0 (from filled).
        """
        engine = self._make_engine()

        mock_exchange = AsyncMock()
        mock_exchange.create_market_order.return_value = {
            "id": "ord-004",
            "type": "market",
            "average": 50000.0,
            "filled": 1.0,
            "amount": 1.0,
            "cost": 50000.0,
            "fee": {"cost": 50.0},
        }
        engine._data.exchange = mock_exchange

        with patch("execution_engine.cfg") as mock_cfg:
            mock_cfg.PAPER_TRADE = False

            result = await engine._place_order(
                symbol="BTC/USDT", side="buy", amount=1.0,
                price=None, order_type="market",
            )

        assert result is not None
        assert result["amount"] == 1.0


# ===========================================================================
# Test 4: Telegram Emergency Alert — Message format
# ===========================================================================


class TestTelegramEmergencyAlert:
    """Verify the emergency alert method exists and formats correctly."""

    @pytest.mark.asyncio
    async def test_emergency_alert_sends_message(self):
        """send_emergency_exit_alert must call _send_raw with trade details."""
        from notifications import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test_token",
            chat_id="test_chat",
            enabled=True,
        )

        # Mock the raw send
        notifier._send_raw = AsyncMock(return_value=True)

        result = await notifier.send_emergency_exit_alert(
            symbol="BTC/USDT",
            side="buy",
            amount=0.5,
            entry_price=50000.0,
            trade_id=42,
            attempts=12,
        )

        assert result is True
        notifier._send_raw.assert_called_once()
        sent_text = notifier._send_raw.call_args[0][0]
        assert "MANUAL INTERVENTION" in sent_text
        assert "BTC/USDT" in sent_text
        assert "42" in sent_text
        assert "12" in sent_text
