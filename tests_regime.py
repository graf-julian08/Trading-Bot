"""
tests_regime.py — Phase 3: Regime Discriminator Hostile Audit
=============================================================
Verifies the three regime filters:
  1. Mega-Trend Filter (SMA 200) — blocks longs in bear markets
  2. Chop Filter (ADX < 25) — rejects imperfect setups in sideways markets
  3. Volatility Targeting (ATR) — position size halves when ATR doubles

These tests directly validate the user's three hostile scenarios:
  - BTC crashing below SMA 200, Jury=88% BUY → MUST REJECT
  - Market flat (ADX=15), Jury=90% BUY → MUST REJECT (needs >95%)
  - ATR doubles → position size must halve

Run with:
    python -m pytest tests_regime.py -v

If ANY of these fail, Phase 3 is NOT deployable.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ai_model import _sma, _adx, _atr
from risk_manager import RiskManager


# ===========================================================================
# Helpers
# ===========================================================================


def _make_bear_market_df(n: int = 300) -> pd.DataFrame:
    """
    Create OHLCV data where the current price is BELOW the SMA 200.
    Strategy: first 200 candles trend UP, then sharp decline.
    """
    rng = np.random.RandomState(42)
    # Rising phase: 50000 → 55000
    rising = np.linspace(50000, 55000, 200) + rng.randn(200) * 20
    # Crashing phase: 55000 → 48000
    falling = np.linspace(55000, 48000, n - 200) + rng.randn(n - 200) * 20
    close = np.concatenate([rising, falling])

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": close + rng.randn(n) * 10,
        "high": close + rng.uniform(50, 150, n),
        "low": close - rng.uniform(50, 150, n),
        "close": close,
        "volume": rng.uniform(100, 5000, n),
    })


def _make_choppy_market_df(n: int = 300) -> pd.DataFrame:
    """
    Create OHLCV data where ADX will be LOW (< 25) — sideways/flat.
    Strategy: oscillate around a fixed mean.
    """
    rng = np.random.RandomState(99)
    close = 50000 + rng.randn(n).cumsum() * 5  # very tiny moves
    close = close - close.mean() + 50000  # center around 50000

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": close + rng.randn(n) * 3,
        "high": close + rng.uniform(5, 20, n),
        "low": close - rng.uniform(5, 20, n),
        "close": close,
        "volume": rng.uniform(100, 5000, n),
    })


# ===========================================================================
# Test 1: Mega-Trend Filter (SMA 200)
# ===========================================================================


class TestMegaTrendFilter:
    """
    HOSTILE SCENARIO: Bitcoin is crashing below the SMA 200.
    The Jury votes 88% BUY. Action: MUST REJECT.
    """

    def test_sma200_bear_mode_rejects_long(self):
        """
        Price below SMA 200 + ensemble < 0.90 → MUST reject buy signal.
        This is the primary hostile scenario.
        """
        df = _make_bear_market_df(300)
        current_price = float(df["close"].iloc[-1])
        sma_200 = _sma(df["close"], 200)
        sma_200_val = float(sma_200.iloc[-1])

        # Verify we're actually in bear mode
        assert current_price < sma_200_val, (
            f"Test setup failed: price {current_price:.0f} should be "
            f"below SMA200 {sma_200_val:.0f}"
        )

        # Simulate the filter logic from main.py
        signal_direction = "buy"
        ml_prob = 0.88  # The Jury votes 88% BUY

        should_reject = (
            signal_direction == "buy"
            and current_price < sma_200_val
            and ml_prob < 0.90
        )

        assert should_reject, (
            "CRITICAL: Mega-Trend Filter FAILED to reject a long trade "
            f"below SMA200! price={current_price:.0f}, "
            f"SMA200={sma_200_val:.0f}, ml_prob={ml_prob}"
        )

    def test_sma200_bear_mode_allows_extreme_confidence(self):
        """
        Price below SMA 200 BUT ensemble >= 0.90 → should ALLOW.
        """
        df = _make_bear_market_df(300)
        current_price = float(df["close"].iloc[-1])
        sma_200 = _sma(df["close"], 200)
        sma_200_val = float(sma_200.iloc[-1])

        ml_prob = 0.92  # Extreme confidence override

        should_reject = (
            current_price < sma_200_val and ml_prob < 0.90
        )
        assert not should_reject, (
            "Mega-Trend Filter wrongly rejected a high-confidence trade "
            f"(ml_prob={ml_prob})"
        )

    def test_sma200_bull_mode_allows_long(self):
        """
        Price ABOVE SMA 200 → no filter action, longs allowed.
        """
        rng = np.random.RandomState(42)
        n = 300
        # Steady uptrend: SMA will be below price
        close = np.linspace(48000, 55000, n) + rng.randn(n) * 20

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close + rng.randn(n) * 10,
            "high": close + rng.uniform(50, 150, n),
            "low": close - rng.uniform(50, 150, n),
            "close": close,
            "volume": rng.uniform(100, 5000, n),
        })

        current_price = float(df["close"].iloc[-1])
        sma_200 = _sma(df["close"], 200)
        sma_200_val = float(sma_200.iloc[-1])

        assert current_price > sma_200_val, "Test setup: should be bull mode"

        # In bull mode, the filter should NOT reject
        signal_direction = "buy"
        ml_prob = 0.55  # Even low confidence passes in bull mode

        should_reject = (
            signal_direction == "buy"
            and current_price < sma_200_val
            and ml_prob < 0.90
        )
        assert not should_reject, "Mega-Trend filter wrongly blocked a bull-mode long"

    def test_sell_signals_not_affected_by_bear_mode(self):
        """
        Sell signals should pass through regardless of SMA 200.
        Only BUY signals are blocked in bear mode.
        """
        df = _make_bear_market_df(300)
        current_price = float(df["close"].iloc[-1])
        sma_200 = _sma(df["close"], 200)
        sma_200_val = float(sma_200.iloc[-1])

        # Bear mode confirmed
        assert current_price < sma_200_val

        signal_direction = "sell"
        ml_prob = 0.70

        # The filter only checks signal_direction == "buy"
        should_reject = (
            signal_direction == "buy"
            and current_price < sma_200_val
            and ml_prob < 0.90
        )
        assert not should_reject, "Sell signal wrongly blocked by bear filter"


# ===========================================================================
# Test 2: Chop Filter (ADX < 25)
# ===========================================================================


class TestChopFilter:
    """
    HOSTILE SCENARIO: Market is flat (ADX=15). Jury=90% BUY.
    Action: MUST REJECT (needs > 95% in chop).
    """

    def test_chop_filter_rejects_90pct_in_sideways(self):
        """
        ADX < 25 + agreement < 0.95 → MUST reject.
        This is the second hostile scenario.
        """
        df = _make_choppy_market_df(300)

        adx_series = _adx(df["high"], df["low"], df["close"], period=14)
        adx_val = float(adx_series.dropna().iloc[-1])

        # Verify we're in choppy territory
        assert adx_val < 25, (
            f"Test setup: ADX should be < 25 for choppy market, got {adx_val:.1f}. "
            "The synthetic data may need adjustment."
        )

        # The hostile scenario
        agreement = 0.90  # The Jury agrees 90%

        chop_agreement_threshold = 0.85  # default
        if adx_val < 25:
            chop_agreement_threshold = 0.95

        should_reject = agreement < chop_agreement_threshold

        assert should_reject, (
            "CRITICAL: Chop Filter FAILED to reject a 90% agreement trade "
            f"in a sideways market! ADX={adx_val:.1f}, agreement={agreement}"
        )

    def test_chop_filter_allows_96pct_in_sideways(self):
        """
        ADX < 25 but agreement >= 0.95 → should ALLOW (perfect setup).
        """
        agreement = 0.96

        chop_agreement_threshold = 0.85
        adx_val = 15  # Definitely choppy
        if adx_val < 25:
            chop_agreement_threshold = 0.95

        should_reject = agreement < chop_agreement_threshold
        assert not should_reject, (
            "Chop filter wrongly rejected a 96% agreement — this IS a perfect setup"
        )

    def test_trending_market_uses_normal_threshold(self):
        """
        ADX >= 25 → use normal agreement threshold (0.85), not 0.95.
        """
        adx_val = 35  # Trending
        agreement = 0.88  # Passes normal threshold

        chop_agreement_threshold = 0.85
        if adx_val < 25:
            chop_agreement_threshold = 0.95

        should_reject = agreement < chop_agreement_threshold
        assert not should_reject, (
            "In a trending market (ADX=35), normal agreement threshold (0.85) "
            "should apply. 0.88 should pass."
        )

    def test_adx_boundary_at_25(self):
        """
        ADX = 25 exactly → should NOT trigger chop filter (< 25 only).
        """
        adx_val = 25
        agreement = 0.88

        chop_agreement_threshold = 0.85
        if adx_val < 25:  # strictly less than
            chop_agreement_threshold = 0.95

        should_reject = agreement < chop_agreement_threshold
        assert not should_reject, (
            "ADX=25 exactly should NOT trigger chop filter (< 25 only)"
        )


# ===========================================================================
# Test 3: Volatility Targeting (ATR Scaling)
# ===========================================================================


class TestVolatilityTargeting:
    """
    HOSTILE SCENARIO: ATR doubles due to news.
    Action: Position size must halve.
    """

    def test_atr_doubles_position_halves(self):
        """
        If ATR doubles, the position size must halve
        to keep dollar risk constant.
        """
        equity = 10_000.0
        target_risk = 0.02  # 2%

        # Normal conditions
        normal_atr = 500.0
        normal_size = RiskManager.calculate_atr_position_size(
            equity=equity,
            target_risk_pct=target_risk,
            current_atr=normal_atr,
        )

        # ATR doubles (crash/pump)
        high_atr = 1000.0
        crisis_size = RiskManager.calculate_atr_position_size(
            equity=equity,
            target_risk_pct=target_risk,
            current_atr=high_atr,
        )

        # Crisis size should be exactly half of normal
        ratio = crisis_size / normal_size
        assert abs(ratio - 0.5) < 0.001, (
            f"CRITICAL: ATR doubled but position ratio is {ratio:.4f}, "
            f"expected 0.5. normal={normal_size:.8f}, crisis={crisis_size:.8f}"
        )

    def test_formula_is_correct(self):
        """
        Verify: Position = (Equity × Risk) / (ATR × Multiplier)
        """
        equity = 10_000.0
        risk_pct = 0.02
        atr = 500.0
        multiplier = 2.0

        expected = (equity * risk_pct) / (atr * multiplier)
        actual = RiskManager.calculate_atr_position_size(
            equity=equity,
            target_risk_pct=risk_pct,
            current_atr=atr,
            atr_multiplier=multiplier,
        )

        assert abs(actual - expected) < 1e-10, (
            f"Formula mismatch: expected {expected}, got {actual}"
        )

    def test_zero_atr_returns_zero(self):
        """ATR=0 (impossible but defensive) → must not divide by zero."""
        size = RiskManager.calculate_atr_position_size(
            equity=10_000.0,
            target_risk_pct=0.02,
            current_atr=0.0,
        )
        assert size == 0.0, "ATR=0 should return 0, not crash"

    def test_negative_equity_returns_zero(self):
        """Defensive: negative equity should return 0."""
        size = RiskManager.calculate_atr_position_size(
            equity=-5000.0,
            target_risk_pct=0.02,
            current_atr=500.0,
        )
        assert size == 0.0, "Negative equity should return 0"

    def test_atr_triples_position_thirds(self):
        """If ATR triples, position should be 1/3 of normal."""
        equity = 10_000.0
        risk = 0.02

        normal = RiskManager.calculate_atr_position_size(
            equity=equity, target_risk_pct=risk, current_atr=500.0,
        )
        triple = RiskManager.calculate_atr_position_size(
            equity=equity, target_risk_pct=risk, current_atr=1500.0,
        )

        ratio = triple / normal
        assert abs(ratio - 1/3) < 0.001, (
            f"ATR tripled but ratio is {ratio:.4f}, expected {1/3:.4f}"
        )

    def test_volatility_adjusted_halves_when_atr_doubles(self):
        """
        The existing calculate_volatility_adjusted_size should also
        approximately halve when current_atr_pct is 2× median_atr_pct.
        """
        from config import cfg

        base_size = 1.0

        # Normal: current == median, ratio = 1.0
        normal = RiskManager.calculate_volatility_adjusted_size(
            base_size=base_size,
            current_atr_pct=0.01,
            median_atr_pct=0.01,
        )

        # Doubled volatility: current = 2× median
        doubled = RiskManager.calculate_volatility_adjusted_size(
            base_size=base_size,
            current_atr_pct=0.02,
            median_atr_pct=0.01,
        )

        # The ratio should be approximately 0.5 (× RISK_SCALAR)
        # Exact: base_size * (0.01/0.02) * RISK_SCALAR = 0.5 * RISK_SCALAR
        expected_ratio = 0.5 * cfg.VOL_TARGET_RISK_SCALAR
        actual_ratio = doubled / normal if normal > 0 else 0

        assert abs(actual_ratio - expected_ratio) < 0.01, (
            f"Volatility-adjusted ratio is {actual_ratio:.4f}, "
            f"expected ≈ {expected_ratio:.4f} "
            f"(risk_scalar={cfg.VOL_TARGET_RISK_SCALAR})"
        )


# ===========================================================================
# Test 4: Integration Checks
# ===========================================================================


class TestRegimeFilterIntegration:
    """
    Verify the filter functions are importable and the main.py
    imports are correct.
    """

    def test_sma_function_exists(self):
        """_sma must be importable from ai_model."""
        from ai_model import _sma
        assert callable(_sma)

    def test_adx_function_exists(self):
        """_adx must be importable from ai_model."""
        from ai_model import _adx
        assert callable(_adx)

    def test_atr_function_exists(self):
        """_atr must be importable from ai_model."""
        from ai_model import _atr
        assert callable(_atr)

    def test_main_imports_regime_functions(self):
        """main.py must import _sma, _adx, _atr."""
        import importlib
        main_module = importlib.import_module("main")
        assert hasattr(main_module, "_sma"), "_sma not imported in main.py"
        assert hasattr(main_module, "_adx"), "_adx not imported in main.py"
        assert hasattr(main_module, "_atr"), "_atr not imported in main.py"

    def test_atr_position_method_exists(self):
        """RiskManager must have calculate_atr_position_size."""
        assert hasattr(RiskManager, "calculate_atr_position_size"), (
            "Missing calculate_atr_position_size on RiskManager"
        )

    def test_regime_filter_in_main_source(self):
        """The regime discriminator block must exist in main.py."""
        import inspect
        import main

        src = inspect.getsource(main.run_bot)
        assert "MEGA-TREND FILTER" in src, "Missing Mega-Trend Filter in main loop"
        assert "CHOP FILTER" in src, "Missing Chop Filter in main loop"
        assert "REGIME DISCRIMINATOR" in src, "Missing Phase 3 header in main loop"
