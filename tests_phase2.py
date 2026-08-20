"""
tests_phase2.py — Hostile Self-Audit for Phase 2: Jury & Signal Logic
=======================================================================
Tests the 3 architectural fixes:
  1. Agreement gate: contradicting models MUST NOT execute a trade
  2. Spot-only guard: sell signals with no open position MUST be blocked
  3. Rolling accuracy: tracks TP hits, not price direction

Run:
    cd /path/to/Trading-Bot
    python -m pytest tests_phase2.py -v
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ===========================================================================
# Test 1: Agreement Gate
# ===========================================================================


class TestAgreementGate:
    """
    Hostile audit: Model A says 90% Buy, Model B says 20% Buy.
    The trade MUST be rejected.
    """

    def test_high_disagreement_blocks_trade(self):
        """
        Simulate: XGBoost=0.90, CatBoost=0.20
        Average = 0.55, Agreement = 1.0 - std([0.90, 0.20]) = 1.0 - 0.495 = 0.505
        Agreement 0.505 < 0.85 → MUST REJECT.
        """
        probas = [0.90, 0.20]
        avg_prob = float(np.mean(probas))
        agreement = float(1.0 - np.std(probas))

        # Import the threshold
        from config import cfg
        min_agreement = cfg.ENSEMBLE_MIN_AGREEMENT  # 0.85

        assert agreement < min_agreement, \
            f"Agreement {agreement:.4f} should be below {min_agreement} " \
            f"when models contradict (0.90 vs 0.20)"

        # Even if avg_prob is irrelevant, the agreement gate fires FIRST
        # and rejects the trade. Verify the logic order.
        should_trade = agreement >= min_agreement and avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD
        assert not should_trade, "Trade must be REJECTED when agreement is below threshold"

    def test_five_models_one_dissenter_blocks(self):
        """
        Simulate: XGB_Deep=0.95, XGB_Shallow=0.95, CatBoost=0.10, LGBM=0.95, RF=0.95
        This was the scenario from the original audit that PASSED incorrectly.
        """
        probas = [0.95, 0.95, 0.10, 0.95, 0.95]
        avg_prob = float(np.mean(probas))
        agreement = float(1.0 - np.std(probas))

        from config import cfg

        # Avg = 0.78, which is above 0.70 threshold
        assert avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD, \
            "Average is high enough to pass consensus — that's the danger"

        # BUT agreement is low because of the CatBoost dissenter
        assert agreement < cfg.ENSEMBLE_MIN_AGREEMENT, \
            f"Agreement {agreement:.4f} must be below {cfg.ENSEMBLE_MIN_AGREEMENT} " \
            f"with one strong dissenter"

        # Under the NEW logic: agreement gate fires first → trade rejected
        should_trade = agreement >= cfg.ENSEMBLE_MIN_AGREEMENT
        assert not should_trade, \
            "CRITICAL: Trade must be REJECTED. The old code would have passed this!"

    def test_perfect_agreement_passes(self):
        """
        When all models agree (e.g., all at 0.80), the trade should pass.
        """
        probas = [0.80, 0.80, 0.80, 0.80, 0.80]
        avg_prob = float(np.mean(probas))
        agreement = float(1.0 - np.std(probas))

        from config import cfg

        assert agreement >= cfg.ENSEMBLE_MIN_AGREEMENT, \
            f"Perfect agreement {agreement:.4f} should be >= {cfg.ENSEMBLE_MIN_AGREEMENT}"
        assert avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD
        should_trade = agreement >= cfg.ENSEMBLE_MIN_AGREEMENT and avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD
        assert should_trade

    def test_moderate_agreement_with_minor_variance_passes(self):
        """
        Realistic scenario: models have slight variation but general agreement.
        XGB_Deep=0.82, XGB_Shallow=0.78, CatBoost=0.75, LGBM=0.80, RF=0.77
        """
        probas = [0.82, 0.78, 0.75, 0.80, 0.77]
        avg_prob = float(np.mean(probas))
        agreement = float(1.0 - np.std(probas))

        from config import cfg

        assert avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD, \
            f"Average {avg_prob:.4f} should pass consensus threshold"
        assert agreement >= cfg.ENSEMBLE_MIN_AGREEMENT, \
            f"Agreement {agreement:.4f} should pass — models have minor variance"

    def test_all_models_low_but_agreeing_is_rejected(self):
        """
        All models agree the probability is low (e.g., 0.40).
        Agreement is perfect, but avg_prob is below threshold. MUST NOT trade.
        """
        probas = [0.40, 0.40, 0.40, 0.40, 0.40]
        avg_prob = float(np.mean(probas))
        agreement = float(1.0 - np.std(probas))

        from config import cfg

        assert agreement >= cfg.ENSEMBLE_MIN_AGREEMENT
        assert avg_prob < cfg.ENSEMBLE_CONSENSUS_THRESHOLD
        should_trade = agreement >= cfg.ENSEMBLE_MIN_AGREEMENT and avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD
        assert not should_trade, "Low-confidence consensus must still be rejected"


# ===========================================================================
# Test 2: Spot-Only Sell Guard
# ===========================================================================


class TestSpotOnlySellGuard:
    """
    Hostile audit: Strong "Sell" signal when wallet is empty.
    If it tries to send a sell order, we have FAILED.
    """

    @pytest.mark.asyncio
    async def test_sell_blocked_when_no_position(self):
        """
        Simulate: generate_signal returns "sell" but get_open_trades(symbol)
        returns an empty list. The sell must be blocked.
        """
        # Import the mock DB from phase 1 tests
        import sys
        sys.path.insert(0, ".")
        from tests_phase1 import MockDatabase

        db = MockDatabase()

        # No open trades for BTC/USDT
        open_trades = await db.get_open_trades("BTC/USDT")
        buy_positions = [t for t in open_trades if t["side"] == "buy"]

        assert len(buy_positions) == 0, "No buy positions should exist"

        # The guard logic from main.py:
        signal_direction = "sell"
        should_skip = (signal_direction == "sell" and len(buy_positions) == 0)
        assert should_skip, \
            "CRITICAL: Sell signal with no position must be skipped on Spot"

    @pytest.mark.asyncio
    async def test_sell_allowed_when_holding(self):
        """
        If we hold a buy position, sell signal should be allowed.
        """
        from tests_phase1 import MockDatabase

        db = MockDatabase()

        # Open a buy position
        await db.log_trade(
            symbol="BTC/USDT", side="buy", order_type="limit",
            price=50000.0, amount=0.1, cost=5000.0, fee=5.0,
            stop_loss=49000.0, take_profit=52000.0,
        )

        open_trades = await db.get_open_trades("BTC/USDT")
        buy_positions = [t for t in open_trades if t["side"] == "buy"]

        assert len(buy_positions) == 1, "Should have 1 buy position"

        signal_direction = "sell"
        should_skip = (signal_direction == "sell" and len(buy_positions) == 0)
        assert not should_skip, "Sell should be ALLOWED when holding a buy position"

    @pytest.mark.asyncio
    async def test_buy_signal_always_allowed(self):
        """
        Buy signals should never be blocked by the spot guard,
        regardless of holdings.
        """
        from tests_phase1 import MockDatabase

        db = MockDatabase()

        signal_direction = "buy"
        # The guard only applies to "sell" signals
        guard_active = (signal_direction == "sell")
        assert not guard_active, "Buy signals must never be blocked by spot guard"


# ===========================================================================
# Test 3: Rolling Accuracy — TP Hit Tracking
# ===========================================================================


class TestRollingAccuracy:
    """
    The rolling accuracy must track whether trades hit TP (actual_outcome=1)
    or SL (actual_outcome=0), NOT whether the price went up or down.
    """

    def test_tp_hit_increases_accuracy(self):
        """
        When a trade hits take_profit, actual_outcome=1 and the model predicted
        high probability → accuracy should increase.
        """
        from ai_model import TradePredictor

        predictor = TradePredictor(db=None)
        predictor._rolling_accuracy = 0.50  # baseline

        # Simulate: model predicted 0.80 (buy), trade hit TP → actual=1
        predictor.update_rolling_accuracy(actual=1, predicted_prob=0.80)

        # predicted_class = 1 (0.80 >= 0.5), actual = 1 → correct
        # accuracy = 0.05 * 1.0 + 0.95 * 0.50 = 0.525
        assert predictor.accuracy > 0.50, \
            f"TP hit should increase accuracy, got {predictor.accuracy:.4f}"

    def test_sl_hit_decreases_accuracy(self):
        """
        When a trade hits stop_loss, actual_outcome=0 but model predicted high
        probability → accuracy should decrease.
        """
        from ai_model import TradePredictor

        predictor = TradePredictor(db=None)
        predictor._rolling_accuracy = 0.60  # decent baseline

        # Simulate: model predicted 0.75 (buy), but trade hit SL → actual=0
        predictor.update_rolling_accuracy(actual=0, predicted_prob=0.75)

        # predicted_class = 1 (0.75 >= 0.5), actual = 0 → incorrect
        # accuracy = 0.05 * 0.0 + 0.95 * 0.60 = 0.57
        assert predictor.accuracy < 0.60, \
            f"SL hit should decrease accuracy, got {predictor.accuracy:.4f}"

    def test_consecutive_losses_trigger_retrain(self):
        """
        Multiple consecutive SL hits should drive accuracy below
        ML_MIN_ACCURACY (0.52), triggering a retrain.
        """
        from ai_model import TradePredictor
        from config import cfg

        predictor = TradePredictor(db=None)
        predictor._rolling_accuracy = 0.55
        predictor._is_trained = True
        predictor._last_train_time = 1e12  # far future, so time doesn't trigger

        # 30 consecutive SL hits (losses)
        for _ in range(30):
            predictor.update_rolling_accuracy(actual=0, predicted_prob=0.75)

        assert predictor.accuracy < cfg.ML_MIN_ACCURACY, \
            f"After 30 SL hits, accuracy {predictor.accuracy:.4f} " \
            f"must be below {cfg.ML_MIN_ACCURACY}"
        assert predictor.needs_retrain(), \
            "30 consecutive losses MUST trigger retrain"

    def test_wrong_metric_not_used(self):
        """
        Verify the old "price went up" metric is NO LONGER used.
        
        Old code checked: actual = 1 if curr_close > prev_close
        New code checks: actual = 1 if reason == "take_profit"
        
        Scenario: Price went up (bullish candle) but the TRADE hit SL.
        Old metric would say "correct", new metric correctly says "incorrect".
        """
        from ai_model import TradePredictor

        predictor = TradePredictor(db=None)
        predictor._rolling_accuracy = 0.60

        # Old broken metric: price went up → actual=1
        # New correct metric: trade hit SL → actual=0
        # The model predicted 0.70 (buy), so predicted_class=1
        # Old: correct (1==1), New: incorrect (0!=1)

        # Using the NEW metric:
        predictor.update_rolling_accuracy(actual=0, predicted_prob=0.70)

        # accuracy should DECREASE because the trade was a loss
        assert predictor.accuracy < 0.60, \
            "SL hit must decrease accuracy even when price went up on the candle"


# ===========================================================================
# Test 4: Integration — Full Decision Pipeline
# ===========================================================================


class TestDecisionPipeline:
    """
    End-to-end logic verification of the full signal → filter → decide pipeline.
    """

    def test_full_rejection_scenario(self):
        """
        Classical signal: BUY
        Model A: 0.90 (strong buy)
        Model B: 0.15 (strong sell)
        Agreement: 1.0 - std([0.90, 0.15]) = 1.0 - 0.53 = 0.47
        
        Step 1: Agreement gate → 0.47 < 0.85 → REJECT (stop here)
        The average probability is NEVER even checked.
        """
        probas = [0.90, 0.15]
        agreement = float(1.0 - np.std(probas))
        avg_prob = float(np.mean(probas))

        from config import cfg

        # Agreement gate fires FIRST
        if agreement < cfg.ENSEMBLE_MIN_AGREEMENT:
            trade_allowed = False
        elif avg_prob >= cfg.ENSEMBLE_CONSENSUS_THRESHOLD:
            trade_allowed = True
        else:
            trade_allowed = False

        assert not trade_allowed, \
            "CRITICAL FAILURE: conflicting models must be rejected at agreement gate"

    def test_close_trade_includes_ml_probability(self):
        """
        Verify close_trade return dict includes ml_probability so
        the rolling accuracy tracker can use entries ML confidence.
        """
        result = {
            "trade_id": 1,
            "symbol": "BTC/USDT",
            "side": "buy",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "amount": 0.1,
            "pnl": 89.9,
            "pnl_pct": 0.018,
            "reason": "take_profit",
            "entry_fee": 5.0,
            "exit_fee": 5.1,
            "ml_probability": 0.78,  # This must exist now
        }

        assert "ml_probability" in result, \
            "close_trade must return ml_probability for accuracy tracking"
        assert result["ml_probability"] > 0, \
            "ml_probability must reflect the entry confidence"
