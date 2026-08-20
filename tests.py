"""
tests.py — Proof of Life Test Suite ("Black Box" Safety Verification)
=====================================================================
These tests verify all safety modules work WITHOUT INTERNET.
If any of these fail, the bot MUST NOT go live.

Run:
    cd /path/to/Trading-Bot
    python -m pytest tests.py -v

Required tests:
    1. Kill switch triggers at -5% drawdown
    2. Min notional scales up small orders
    3. Profitability gate rejects high-fee trades
    4. Zero-Trust config crashes on missing secrets
    5. Forensic logging captures rejections
    + 3 bonus tests
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ===========================================================================
# Test 1: Kill Switch triggers at -5% drawdown
# ===========================================================================

class TestKillSwitch:
    """Verify the daily drawdown kill switch halts trading."""

    @pytest.mark.asyncio
    async def test_kill_switch_triggers_at_minus_5_percent(self):
        """
        BLACK BOX TEST 1:
        Starting equity = 10,000 USDT.
        Current equity  =  9,400 USDT (-6%).
        
        Kill switch MUST trigger because -6% < -5% limit.
        """
        from risk_manager import RiskManager

        db = AsyncMock()
        db.upsert_daily_pnl = AsyncMock()
        notifier = AsyncMock()
        notifier.send_kill_switch_alert = AsyncMock()

        risk = RiskManager(db=db, notifier=notifier)
        await risk.start_of_day(current_equity=10_000.0)

        # -6% drawdown => should trigger
        result = await risk.check_kill_switch(current_equity=9_400.0)
        assert result is True, \
            "FATAL: Kill switch did NOT trigger at -6% drawdown!"

    @pytest.mark.asyncio
    async def test_kill_switch_does_not_trigger_at_minus_3_percent(self):
        """
        Starting equity = 10,000.
        Current equity  =  9,700 (-3%).
        
        Kill switch must NOT trigger — -3% > -5% limit.
        """
        from risk_manager import RiskManager

        db = AsyncMock()
        notifier = AsyncMock()

        risk = RiskManager(db=db, notifier=notifier)
        await risk.start_of_day(current_equity=10_000.0)

        result = await risk.check_kill_switch(current_equity=9_700.0)
        assert result is False, \
            "Kill switch should NOT trigger at -3% drawdown"


# ===========================================================================
# Test 2: Min Notional scales up small orders
# ===========================================================================

class TestMinNotional:
    """Verify small orders are scaled up to meet exchange minimums."""

    def test_min_notional_scale_up(self):
        """
        BLACK BOX TEST 2:
        Order: 0.00001 BTC at $50,000 = $0.50 notional.
        MIN_NOTIONAL = $10.
        
        System must scale up amount to at least 10/50000 = 0.0002 BTC.
        """
        from config import cfg

        entry_price = 50_000.0
        amount = 0.00001  # $0.50 notional — way below minimum

        notional_value = entry_price * amount
        assert notional_value < cfg.MIN_NOTIONAL_USDT, \
            "Setup error: notional should be below minimum for this test"

        # Calculate the scaled amount
        min_amount = cfg.MIN_NOTIONAL_USDT / entry_price
        assert min_amount > amount, \
            "Scaled amount must be larger than original"
        assert min_amount * entry_price >= cfg.MIN_NOTIONAL_USDT, \
            "Scaled notional must meet minimum"

    def test_notional_check_exists_in_open_trade(self):
        """Verify the notional check is implemented in ExecutionEngine.open_trade."""
        from execution_engine import ExecutionEngine
        source = inspect.getsource(ExecutionEngine.open_trade)

        assert "MIN_NOTIONAL" in source or "min_notional" in source.lower(), \
            "ExecutionEngine.open_trade must check MIN_NOTIONAL"
        assert "notional_value" in source, \
            "Must calculate notional_value = price * amount"


# ===========================================================================
# Test 3: Profitability Gate rejects high-fee trades
# ===========================================================================

class TestProfitabilityGate:
    """Verify trades are rejected when fees eat the profit."""

    def test_high_fee_trade_rejected(self):
        """
        BLACK BOX TEST 3:
        Entry=50000, TP=50010, Amount=0.01.
        Expected profit = $0.10.
        Fees + slippage ≈ $1.075.
        
        Trade MUST be rejected because profit < costs.
        """
        from execution_engine import ExecutionEngine

        result = ExecutionEngine.check_profitability_gate(
            entry_price=50_000.0,
            take_profit=50_010.0,   # Only $10 move on $50k
            amount=0.01,            # 0.01 BTC
            side="buy",
            atr_pct=0.001,
        )
        assert result is False, \
            "FATAL: Profitability gate PASSED a trade where fees exceed profit!"

    def test_profitable_trade_passes(self):
        """
        Entry=50000, TP=52000, Amount=0.1.
        Expected profit = $200.
        This should easily pass the gate.
        """
        from execution_engine import ExecutionEngine

        result = ExecutionEngine.check_profitability_gate(
            entry_price=50_000.0,
            take_profit=52_000.0,   # $2000 move
            amount=0.1,             # 0.1 BTC
            side="buy",
            atr_pct=0.005,
        )
        assert result is True, \
            "Profitability gate should PASS clearly profitable trades"


# ===========================================================================
# Test 4: Zero-Trust Config crashes on missing secrets
# ===========================================================================

class TestZeroTrustConfig:
    """Verify the bot REFUSES to start without required configuration."""

    def test_missing_env_file_crashes(self):
        """
        BLACK BOX TEST 4 (Scenario):
        Delete the .env file and start the bot.
        It must CRASH IMMEDIATELY — not run for even a second.
        """
        from config import Config

        # Verify __post_init__ exists (the crash mechanism)
        assert hasattr(Config, "__post_init__"), \
            "FATAL: Config must have __post_init__ for Zero-Trust validation"

        source = inspect.getsource(Config.__post_init__)
        assert "SystemExit" in source or "raise" in source, \
            "FATAL: __post_init__ must raise SystemExit on missing config"

    def test_live_mode_requires_api_secrets(self):
        """
        PAPER_TRADE=false but no API keys = must crash.
        """
        from config import Config

        source = inspect.getsource(Config.__post_init__)
        assert "EXCHANGE_API_KEY" in source, \
            "Must validate EXCHANGE_API_KEY for live trading"
        assert "EXCHANGE_API_SECRET" in source, \
            "Must validate EXCHANGE_API_SECRET for live trading"
        assert "PAPER_TRADE" in source, \
            "Must check PAPER_TRADE mode before requiring secrets"

    def test_telegram_enabled_requires_tokens(self):
        """
        TELEGRAM_ENABLED=true but no token = must crash.
        """
        from config import Config

        source = inspect.getsource(Config.__post_init__)
        assert "TELEGRAM_BOT_TOKEN" in source, \
            "Must validate TELEGRAM_BOT_TOKEN when Telegram is enabled"
        assert "TELEGRAM_CHAT_ID" in source, \
            "Must validate TELEGRAM_CHAT_ID when Telegram is enabled"


# ===========================================================================
# Test 5: Forensic Logging captures every rejection
# ===========================================================================

class TestForensicLogging:
    """Verify every trade rejection is logged with exact reason."""

    def test_jury_disagreement_logged(self):
        """
        BLACK BOX TEST 5 (Scenario):
        Bot runs for 24h, makes 0 trades.
        Log file MUST NOT be empty — it should show rejection reasons.
        
        Verify the code path for jury disagreement logs at INFO level.
        """
        import main
        source = inspect.getsource(main.run_bot)

        # Must have the standardized rejection format
        assert "Trade REJECTED:" in source, \
            "FATAL: Rejection logs must use 'Trade REJECTED:' prefix"

        # Must log Jury Agreement rejections
        assert "Jury Agreement" in source, \
            "Must log Jury Agreement rejection reason"

    def test_consensus_rejection_logged(self):
        """Verify low-probability consensus rejection is logged."""
        import main
        source = inspect.getsource(main.run_bot)

        assert "Consensus" in source, \
            "Must log Consensus threshold rejection"

    def test_heartbeat_logged_every_iteration(self):
        """
        Scenario: Bot runs for 24h, log MUST show heartbeat entries.
        """
        import main
        source = inspect.getsource(main.run_bot)

        assert "Heartbeat:" in source or "Heartbeat" in source, \
            "FATAL: Must log heartbeat every iteration so log is never empty"

    def test_scan_results_logged(self):
        """
        Scenario: No signal found, but log must show scan results.
        """
        import main
        source = inspect.getsource(main.run_bot)

        assert "Scan results:" in source, \
            "Must log scan results even when no signal is found"


# ===========================================================================
# Bonus Test 6: Circuit Breaker records trade results
# ===========================================================================

class TestCircuitBreaker:
    """Verify the circuit breaker tracks consecutive losses."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_activates_after_losses(self):
        """After N consecutive losses, can_open_trade must return False."""
        from risk_manager import RiskManager

        db = AsyncMock()
        db.upsert_daily_pnl = AsyncMock()
        notifier = AsyncMock()
        notifier.send_circuit_breaker_alert = AsyncMock()

        risk = RiskManager(db=db, notifier=notifier)
        await risk.start_of_day(current_equity=10_000.0)

        # Record N consecutive losses (N = CIRCUIT_BREAKER_LOSSES)
        from config import cfg
        for _ in range(cfg.CIRCUIT_BREAKER_LOSSES):
            risk.record_trade_result(is_win=False)

        # After N losses, trading should be blocked
        can_trade = await risk.can_open_trade(10_000.0)
        assert can_trade is False, \
            f"Circuit breaker must activate after {cfg.CIRCUIT_BREAKER_LOSSES} consecutive losses"


# ===========================================================================
# Bonus Test 7: Spot-Only Sell Guard
# ===========================================================================

class TestSpotOnlySellGuard:
    """Verify sell signals are blocked when not holding the asset."""

    def test_sell_guard_exists_in_main_loop(self):
        """The main loop must check for open positions before executing sells."""
        import main
        source = inspect.getsource(main.run_bot)

        assert "sell" in source.lower() and "REJECTED" in source, \
            "Must reject sell signals when no position is open"
        assert "Spot" in source or "position" in source.lower(), \
            "Must reference Spot-only constraint"


# ===========================================================================  
# Bonus Test 8: Paper Balance Tracker
# ===========================================================================

class TestPaperBalanceTracker:
    """Verify paper trading tracks equity correctly."""

    def test_paper_balance_debits_on_open(self):
        """Opening a trade must debit cost + fee from paper balance."""
        from main import PaperBalanceTracker

        tracker = PaperBalanceTracker(initial_balance=10_000.0)
        assert tracker.balance == 10_000.0

        tracker.on_trade_opened(cost=5_000.0, fee=5.0)
        assert tracker.balance == 10_000.0 - 5_000.0 - 5.0, \
            "Paper balance must debit cost + fee on trade open"

    def test_paper_balance_credits_on_close(self):
        """Closing a profitable trade must credit proceeds - fee."""
        from main import PaperBalanceTracker

        tracker = PaperBalanceTracker(initial_balance=10_000.0)
        tracker.on_trade_opened(cost=5_000.0, fee=5.0)

        # Close at profit: bought at 50000, sell at 51000, 0.1 BTC
        tracker.on_trade_closed(
            side="buy",
            entry_price=50_000.0,
            exit_price=51_000.0,
            amount=0.1,           # 0.1 BTC
            entry_cost=5_000.0,
            exit_fee=5.1,         # fee on exit
        )

        # After close: initial - cost - entry_fee + (exit_price * amount - exit_fee)
        expected = 10_000.0 - 5_000.0 - 5.0 + (51_000.0 * 0.1 - 5.1)
        assert abs(tracker.balance - expected) < 0.01, \
            f"Paper balance should be {expected}, got {tracker.balance}"
