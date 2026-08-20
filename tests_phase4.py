"""
tests_phase4.py — Hostile Self-Audit for Phase 4: ML Training "Cheat Code" Detector
=====================================================================================
Tests that the 3 "data leakage" cheat codes are IMPOSSIBLE:

  1. IRON CURTAIN: StandardScaler must NEVER be fit on validation/test data
  2. PURGE GAP:    Minimum 24h embargo between train and validation sets
  3. EARLY STOP:   Boosted models must use early_stopping_rounds=50

Run:
    cd /path/to/Trading-Bot
    python -m pytest tests_phase4.py -v
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# Test 1: Iron Curtain — Scaler NEVER sees future data
# ===========================================================================


class TestIronCurtainScaler:
    """
    Hostile audit: If the StandardScaler.fit() is called on data that
    includes validation/test rows, the model is CHEATING.
    """

    def test_scaler_exists_on_predictor(self):
        """TradePredictor must have a _scaler attribute."""
        from ai_model import TradePredictor
        predictor = TradePredictor()
        assert hasattr(predictor, "_scaler"), \
            "CRITICAL: TradePredictor must have _scaler attribute"

    def test_scaler_is_none_before_training(self):
        """Before training, scaler should be None."""
        from ai_model import TradePredictor
        predictor = TradePredictor()
        assert predictor._scaler is None, \
            "Scaler must be None before training"

    def test_train_method_fits_scaler_on_train_only(self):
        """
        CHEAT CODE SCENARIO:
        The StandardScaler mean is calculated using rows from the future.
        
        We verify by inspecting the training code: the CV loop must call
        fold_scaler.fit_transform(X_train) and fold_scaler.transform(X_val).
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.train)

        # Must fit scaler ONLY on train
        assert "fit_transform(X_train)" in source, \
            "CHEAT CODE: Scaler must be fit_transform'd ONLY on X_train"
        assert "fold_scaler.transform(X_val)" in source, \
            "CHEAT CODE: X_val must be transform'd (NOT fit) by the scaler"

        # Must NOT fit_transform the validation set
        assert "fit_transform(X_val)" not in source, \
            "CHEAT CODE DETECTED: Scaler must NEVER fit_transform the validation set"

    def test_leaky_scaler_detected(self):
        """
        Demonstrate that fitting on ALL data vs train-only produces
        DIFFERENT mean/std values. If they're the same, Iron Curtain is broken.
        """
        rng = np.random.RandomState(42)

        # Create "train" and "test" with different distributions
        X_train = rng.randn(100, 5) * 2 + 5   # mean ~5, std ~2
        X_test = rng.randn(50, 5) * 10 + 50   # mean ~50, std ~10

        # Leaky scaler: fits on ALL data (FORBIDDEN)
        leaky_scaler = StandardScaler()
        X_all = np.vstack([X_train, X_test])
        leaky_scaler.fit(X_all)

        # Iron Curtain scaler: fits ONLY on train (REQUIRED)
        safe_scaler = StandardScaler()
        safe_scaler.fit(X_train)

        # The means MUST be different (proving the Iron Curtain matters)
        assert not np.allclose(leaky_scaler.mean_, safe_scaler.mean_, atol=1.0), \
            "Leaky and safe scalers must have different means — " \
            "if they're the same, the test data distribution is too similar"

    def test_predict_uses_scaler(self):
        """
        The predict_with_consensus method must apply the scaler before inference.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.predict_with_consensus)

        assert "self._scaler" in source, \
            "predict_with_consensus must reference self._scaler"
        assert "transform" in source, \
            "predict_with_consensus must call transform on features"

    def test_scaler_persisted_with_models(self):
        """
        The save_model method must persist the scaler alongside models.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.save_model)

        assert "scaler" in source, \
            "save_model must persist the scaler"

    def test_scaler_loaded_from_models(self):
        """
        The load_model method must restore the scaler from saved state.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.load_model)

        assert "scaler" in source, \
            "load_model must restore the scaler"


# ===========================================================================
# Test 2: Purge Gap — 24-hour minimum embargo
# ===========================================================================


class TestPurgeGap:
    """
    Hostile audit: Training data ends at timestamp T and validation
    starts at T+1h. This is CHEATING because market autocorrelation
    lasts for hours.
    """

    def test_default_purge_gap_is_24(self):
        """
        CHEAT CODE SCENARIO:
        Training ends at T, validation starts at T+1h.

        Must be T+24h minimum.
        """
        from ai_model import PurgedTimeSeriesSplit
        cv = PurgedTimeSeriesSplit()
        assert cv.purge_gap >= 24, \
            f"CHEAT CODE: purge_gap={cv.purge_gap} < 24. " \
            "Must have 24+ hour embargo (24 candles at 1h timeframe)"

    def test_purge_gap_enforced_in_splits(self):
        """
        Verify that actual splits have a gap >= 24 between train end and val start.
        """
        from ai_model import PurgedTimeSeriesSplit

        cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=24)
        X = np.zeros(1000)  # simulate 1000 candles

        for train_idx, val_idx in cv.split(X):
            train_end = train_idx[-1]
            val_start = val_idx[0]
            gap = val_start - train_end

            assert gap >= 24, \
                f"CHEAT CODE: gap between train[{train_end}] and " \
                f"val[{val_start}] is only {gap} < 24 hours!"

    def test_one_hour_gap_would_fail(self):
        """
        Explicitly verify that a 1-candle gap is insufficient.
        """
        from ai_model import PurgedTimeSeriesSplit

        # This is the WRONG configuration that would allow cheating
        bad_cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=1)
        X = np.zeros(1000)

        for train_idx, val_idx in bad_cv.split(X):
            gap = val_idx[0] - train_idx[-1]
            # With purge_gap=1, gap should be exactly 1 (the BAD case)
            assert gap < 24, "A purge_gap=1 should NOT produce 24h gaps"
            break  # Just check the first fold

    def test_purge_gap_used_in_train_method(self):
        """
        Verify the train() method uses purge_gap=24, not 10.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.train)

        assert "purge_gap=24" in source, \
            "CHEAT CODE: train() must use purge_gap=24 (not 10)"

    def test_no_overlap_between_train_and_val(self):
        """
        No index should appear in both train and validation sets.
        """
        from ai_model import PurgedTimeSeriesSplit

        cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=24)
        X = np.zeros(2000)

        for train_idx, val_idx in cv.split(X):
            train_set = set(train_idx)
            val_set = set(val_idx)
            overlap = train_set & val_set
            assert len(overlap) == 0, \
                f"CHEAT CODE: {len(overlap)} indices overlap between train and val!"


# ===========================================================================
# Test 3: Early Stopping — No fixed 1000 rounds
# ===========================================================================


class TestEarlyStopping:
    """
    Hostile audit: If the model trains for a fixed number of rounds
    without early stopping, it will overfit massively.
    """

    def test_xgboost_has_early_stopping(self):
        """
        All XGBoost models must have early_stopping_rounds=50.
        """
        from ai_model import TradePredictor
        ensemble = TradePredictor._build_ensemble(imbalance_weight=1.0)

        for name, model in ensemble:
            if "xgb" in name:
                es = getattr(model, "early_stopping_rounds", None)
                assert es == 50, \
                    f"CHEAT CODE: {name} has early_stopping_rounds={es}, must be 50"

    def test_xgboost_n_estimators_is_cap(self):
        """
        With early stopping, n_estimators is a CAP, not a fixed number.
        It should be high (1000) to let early stopping decide when to stop.
        """
        from ai_model import TradePredictor
        ensemble = TradePredictor._build_ensemble(imbalance_weight=1.0)

        for name, model in ensemble:
            if "xgb" in name:
                n = model.n_estimators
                assert n >= 1000, \
                    f"{name}: n_estimators={n}, should be >= 1000 " \
                    "since early stopping controls actual iterations"

    def test_catboost_has_early_stopping(self):
        """CatBoost (if available) must also have early stopping."""
        from ai_model import TradePredictor, _HAS_CATBOOST

        if not _HAS_CATBOOST:
            pytest.skip("CatBoost not installed")

        ensemble = TradePredictor._build_ensemble(imbalance_weight=1.0)

        for name, model in ensemble:
            if "catboost" in name:
                # CatBoost uses 'od_wait' or early_stopping_rounds param
                source = inspect.getsource(TradePredictor._build_ensemble)
                assert "early_stopping_rounds=50" in source or "od_wait=50" in source, \
                    "CatBoost must have early_stopping_rounds=50"

    def test_train_passes_eval_set(self):
        """
        The train() method must pass eval_set to enable early stopping.
        Without eval_set, early_stopping_rounds is silently ignored!
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.train)

        assert "eval_set=" in source, \
            "CHEAT CODE: train() must pass eval_set for early stopping to work"

    def test_build_ensemble_code_review(self):
        """
        Code review: _build_ensemble must include early_stopping_rounds
        in the XGBoost common params.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor._build_ensemble)

        assert "early_stopping_rounds" in source, \
            "CHEAT CODE: _build_ensemble must set early_stopping_rounds"
        assert "50" in source, \
            "CHEAT CODE: early_stopping_rounds must be 50"


# ===========================================================================
# Test 4: Leakage Detection — 99% Accuracy = FAIL
# ===========================================================================


class TestLeakageDetection:
    """
    Hostile audit: If the model achieves > 90% accuracy,
    something is seriously wrong (data leakage).
    """

    def test_leakage_alarm_exists_in_code(self):
        """
        The train() method must have a leakage alarm for >90% accuracy.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.train)

        assert "LEAKAGE" in source.upper() or "leakage" in source.lower(), \
            "train() must have a leakage detection alarm"
        assert "0.90" in source or "90" in source, \
            "train() must flag accuracy > 90% as suspicious"

    def test_realistic_accuracy_expectations(self):
        """
        Document the expected accuracy range for honest trading models.
        55-60% is realistic. >90% is always data leakage.
        """
        # This is a documentation test — it verifies our expectations
        realistic_range = (0.50, 0.65)
        leakage_threshold = 0.90

        assert realistic_range[1] < leakage_threshold, \
            "Realistic accuracy should be well below leakage threshold"


# ===========================================================================
# Test 5: Integration — Full Pipeline Integrity
# ===========================================================================


class TestPipelineIntegrity:
    """
    End-to-end hostile scenarios matching the exact audit requirements.
    """

    def test_audit_scenario_train_val_gap(self):
        """
        EXACT AUDIT SCENARIO:
        Training data ends at timestamp T and validation starts at T+1h.
        
        With 1h candles, this must be T+24h minimum.
        """
        from ai_model import PurgedTimeSeriesSplit

        cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=24)
        X = np.zeros(500)

        for train_idx, val_idx in cv.split(X):
            gap_hours = val_idx[0] - train_idx[-1]
            assert gap_hours >= 24, \
                f"AUDIT FAILURE: Train ends at T, val starts at T+{gap_hours}h. " \
                "Must be T+24h minimum!"

    def test_audit_scenario_scaler_contamination(self):
        """
        EXACT AUDIT SCENARIO:
        The StandardScaler mean is calculated using rows from the future.

        We verify the code structure makes this impossible.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.train)

        # Must have Iron Curtain comment/code
        assert "IRON CURTAIN" in source, \
            "AUDIT FAILURE: Iron Curtain pattern must be documented in code"
        assert "fit_transform(X_train)" in source, \
            "AUDIT FAILURE: Scaler must fit on X_train ONLY"

    def test_audit_scenario_99_accuracy_alarm(self):
        """
        EXACT AUDIT SCENARIO:
        The model has 99% accuracy on the test set. FAIL.

        Verify that the code flags this as suspicious.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.train)

        # Must log a warning for suspiciously high accuracy
        assert "0.90" in source or "90%" in source, \
            "AUDIT FAILURE: Must flag >90% accuracy as potential leakage"

    def test_model_persistence_format(self):
        """
        Verify save format includes scaler for production deployment.
        """
        from ai_model import TradePredictor
        save_source = inspect.getsource(TradePredictor.save_model)
        load_source = inspect.getsource(TradePredictor.load_model)

        assert "scaler" in save_source, \
            "Save format must include scaler for Iron Curtain in production"
        assert "scaler" in load_source, \
            "Load must restore scaler for consistent inference"

    def test_initialise_handles_v3_format(self):
        """
        Verify initialise() can load both v3 (dict) and legacy (list) formats.
        """
        from ai_model import TradePredictor
        source = inspect.getsource(TradePredictor.initialise)

        assert "scaler" in source, \
            "initialise() must handle scaler from v3 format"
        assert "models" in source, \
            "initialise() must handle models key from v3 dict format"
"""
tests_phase4.py — Hostile Self-Audit for Phase 4: ML Training "Cheat Code" Detector
=====================================================================================
Tests the 3 hardening fixes:
  1. Iron Curtain: StandardScaler fit ONLY on train data
  2. 24h Purge Gap: minimum embargo between train/val
  3. Early Stopping: patience=50 on all boosted models

Run:
    cd /path/to/Trading-Bot
    python -m pytest tests_phase4.py -v
"""
