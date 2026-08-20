"""
optimizer.py — Optuna Hyperparameter Optimizer ("The Solver")
==============================================================
Automatically finds the optimal trading parameters by running 
thousands of trials with different configurations, evaluating each 
against a realistic backtest (with fees + slippage).

Optimises:
  - Strategy: TP%, SL%, max hold candles
  - ML model: learning rates, max depths, n_estimators, regularisation
  - Indicators: RSI period, EMA periods
  - Execution: ML confidence threshold

Objective: Maximise Sharpe Ratio (risk-adjusted returns, not raw PnL).

Usage:
    python optimizer.py                              # 200 trials, BTC/USDT
    python optimizer.py --pairs BTC/USDT ETH/USDT    # Multiple pairs
    python optimizer.py --trials 1000 --candles 30000 # Deep search
    python optimizer.py --jobs -1                     # All CPU cores
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Suppress noisy warnings during optimisation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: Optuna not installed. Run: pip install optuna>=3.6")
    sys.exit(1)

from config import cfg
from database import Database
from data_engine import DataEngine
from ai_model import TradePredictor, _create_trade_target

logger = logging.getLogger("optimizer")

# ============================================================================
# Constants
# ============================================================================

_RESULTS_FILE = Path(__file__).parent / "optimizer_results.json"
_MIN_CANDLES = 2000
_MIN_TEST_TRADES = 20  # Need at least this many trades for a meaningful result


# ============================================================================
# Backtest Engine (parameterised for Optuna)
# ============================================================================

def _backtest_with_params(
    predictor: TradePredictor,
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    max_hold: int,
    threshold: float,
    trading_fee: float = 0.001,
    slippage: float = 0.0005,
    test_frac: float = 0.2,
) -> Dict[str, float]:
    """
    Run a realistic backtest with the given parameters.

    Returns dict with: sharpe, total_pnl, win_rate, max_drawdown,
    profit_factor, n_trades. Returns empty dict on failure.
    """
    feature_df = predictor.engineer_features(df)
    if len(feature_df) < 500:
        return {}

    split_idx = int(len(feature_df) * (1 - test_frac))
    test_df = feature_df.iloc[split_idx:].copy()

    if len(test_df) < 100:
        return {}

    # Create actual trade outcomes with the trial's TP/SL/max_hold
    test_df["actual_target"] = _create_trade_target(
        test_df, tp_pct=tp_pct, sl_pct=sl_pct, max_hold=max_hold,
    )
    test_df = test_df.dropna(subset=["actual_target"]).copy()

    if len(test_df) < 50:
        return {}

    # Get ensemble predictions
    X = test_df[TradePredictor._FEATURE_COLS].values
    probas = []
    for model in predictor._models:
        try:
            p = model.predict_proba(X)[:, 1]
            probas.append(p)
        except Exception:
            pass

    if not probas:
        return {}

    avg_proba = np.mean(probas, axis=0)

    # Simulate trading
    equity = 10000.0
    peak_equity = equity
    max_dd = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_returns = []
    cooldown = 0

    for i in range(len(avg_proba)):
        if cooldown > 0:
            cooldown -= 1
            continue

        if avg_proba[i] >= threshold:
            actual = test_df["actual_target"].values[i]
            cost_pct = 2 * trading_fee + slippage

            if actual == 1:
                net = tp_pct - cost_pct
                wins += 1
                gross_profit += tp_pct
            else:
                net = -(sl_pct + cost_pct)
                losses += 1
                gross_loss += sl_pct

            equity *= (1 + net)
            trade_returns.append(net)
            cooldown = max_hold

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_dd:
            max_dd = dd

    n_trades = wins + losses
    if n_trades < _MIN_TEST_TRADES:
        return {}

    total_pnl = (equity / 10000.0) - 1.0
    profit_factor = gross_profit / max(gross_loss, 0.0001)

    if len(trade_returns) > 1 and np.std(trade_returns) > 0:
        sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(
            252 * 24 / max_hold
        )
    else:
        sharpe = 0.0

    return {
        "sharpe": sharpe,
        "total_pnl": total_pnl,
        "win_rate": wins / n_trades,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
    }


# ============================================================================
# Optuna Objective
# ============================================================================

def create_objective(df: pd.DataFrame, symbol: str, db):
    """
    Factory: creates an Optuna objective function that captures the data.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial: suggest params → train → backtest → return Sharpe.
        """
        # ---- Strategy parameters ----
        tp_pct = trial.suggest_float("tp_pct", 0.005, 0.03, step=0.001)
        sl_pct = trial.suggest_float("sl_pct", 0.002, 0.015, step=0.001)
        max_hold = trial.suggest_int("max_hold", 4, 16)
        threshold = trial.suggest_float("threshold", 0.50, 0.80, step=0.05)

        # ---- PHASE 2: Expanded XGBoost parameters ----
        xgb_lr = trial.suggest_float("xgb_lr", 0.005, 0.15, log=True)
        xgb_depth = trial.suggest_int("xgb_depth", 3, 10)
        xgb_estimators = trial.suggest_int("xgb_estimators", 200, 1200, step=50)
        xgb_subsample = trial.suggest_float("xgb_subsample", 0.5, 0.95)
        xgb_colsample = trial.suggest_float("xgb_colsample", 0.4, 0.9)
        xgb_reg_alpha = trial.suggest_float("xgb_reg_alpha", 0.01, 5.0, log=True)
        xgb_reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.5, 10.0)
        xgb_min_child_weight = trial.suggest_int("xgb_min_child_weight", 1, 20)

        # ---- PHASE 2: CatBoost parameters ----
        cat_depth = trial.suggest_int("cat_depth", 4, 10)
        cat_lr = trial.suggest_float("cat_lr", 0.01, 0.15, log=True)
        cat_l2_reg = trial.suggest_float("cat_l2_reg", 1.0, 10.0)

        # ---- PHASE 2: LightGBM parameters ----
        lgb_lr = trial.suggest_float("lgb_lr", 0.005, 0.15, log=True)
        lgb_depth = trial.suggest_int("lgb_depth", 3, 10)
        lgb_leaves = trial.suggest_int("lgb_leaves", 20, 150)
        lgb_reg_alpha = trial.suggest_float("lgb_reg_alpha", 0.01, 5.0, log=True)
        lgb_reg_lambda = trial.suggest_float("lgb_reg_lambda", 0.5, 10.0)

        # ---- Build a temporary predictor with trial params ----
        predictor = TradePredictor(db=None)
        predictor._symbol = symbol

        # Create target with trial TP/SL
        feature_df = predictor.engineer_features(df)
        if len(feature_df) < 500:
            return -999.0

        feature_df["target"] = _create_trade_target(
            feature_df, tp_pct=tp_pct, sl_pct=sl_pct, max_hold=max_hold,
        )
        feature_df.dropna(subset=["target"], inplace=True)
        feature_df.reset_index(drop=True, inplace=True)

        if len(feature_df) < 500:
            return -999.0

        X = feature_df[TradePredictor._FEATURE_COLS].values
        y = feature_df["target"].values.astype(int)

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos < 50 or n_neg < 50:
            return -999.0

        imbalance_weight = n_neg / max(n_pos, 1)

        # Train on first 80%, test on last 20%
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        try:
            from xgboost import XGBClassifier

            models = []

            # ---- XGBoost (expanded search space) ----
            xgb = XGBClassifier(
                n_estimators=xgb_estimators,
                max_depth=xgb_depth,
                learning_rate=xgb_lr,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                reg_alpha=xgb_reg_alpha,
                reg_lambda=xgb_reg_lambda,
                min_child_weight=xgb_min_child_weight,
                scale_pos_weight=imbalance_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            models.append(xgb)

            # ---- CatBoost (PHASE 2) ----
            try:
                from catboost import CatBoostClassifier
                cat = CatBoostClassifier(
                    depth=cat_depth,
                    learning_rate=cat_lr,
                    iterations=600,
                    l2_leaf_reg=cat_l2_reg,
                    auto_class_weights="Balanced",
                    eval_metric="AUC",
                    random_seed=42,
                    verbose=0,
                )
                cat.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
                models.append(cat)
            except ImportError:
                pass

            # ---- LightGBM (PHASE 2) ----
            try:
                from lightgbm import LGBMClassifier
                lgb = LGBMClassifier(
                    n_estimators=600,
                    max_depth=lgb_depth,
                    num_leaves=lgb_leaves,
                    learning_rate=lgb_lr,
                    reg_alpha=lgb_reg_alpha,
                    reg_lambda=lgb_reg_lambda,
                    scale_pos_weight=imbalance_weight,
                    random_state=42,
                    n_jobs=1,
                    verbose=-1,
                )
                lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])
                models.append(lgb)
            except ImportError:
                pass

            if not models:
                return -999.0

            predictor._models = models
            predictor._is_trained = True

        except Exception as e:
            logger.warning("Trial %d training failed: %s", trial.number, e)
            return -999.0

        # Backtest with trial params
        result = _backtest_with_params(
            predictor, df,
            tp_pct=tp_pct, sl_pct=sl_pct, max_hold=max_hold,
            threshold=threshold,
        )

        if not result:
            return -999.0

        sharpe = result["sharpe"]

        # Log progress
        trial.set_user_attr("win_rate", result["win_rate"])
        trial.set_user_attr("total_pnl", result["total_pnl"])
        trial.set_user_attr("max_drawdown", result["max_drawdown"])
        trial.set_user_attr("profit_factor", result["profit_factor"])
        trial.set_user_attr("n_trades", result["n_trades"])

        return sharpe

    return objective


# ============================================================================
# Study Runner
# ============================================================================

def create_study(
    df: pd.DataFrame,
    symbol: str,
    db,
    n_trials: int = 200,
    n_jobs: int = 1,
    timeout: Optional[int] = None,
) -> optuna.Study:
    """
    Create and run an Optuna hyperparameter optimisation study.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data.
    symbol : str
        Trading pair.
    db : database.Database
        For model persistence (not used in trials, but available).
    n_trials : int
        Number of optimisation trials.
    n_jobs : int
        Parallel workers (-1 = all CPU cores).
    timeout : int, optional
        Maximum seconds for optimisation.

    Returns
    -------
    optuna.Study
        The completed study with best parameters.
    """
    # Suppress Optuna's internal logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(
        multivariate=True,
        seed=42,
        n_startup_trials=max(20, n_trials // 10),
    )

    study = optuna.create_study(
        study_name=f"trading_bot_{symbol.replace('/', '_')}",
        direction="maximize",  # Maximise Sharpe ratio
        sampler=sampler,
    )

    objective = create_objective(df, symbol, db)

    print(f"\n  🔬 Starting Optuna optimisation for {symbol}")
    print(f"     Trials: {n_trials} | Workers: {n_jobs} | Objective: Sharpe Ratio")
    print(f"     Search space: TP%, SL%, max_hold, threshold, XGB params")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        timeout=timeout,
        show_progress_bar=True,
    )

    return study


# ============================================================================
# Results & Reporting
# ============================================================================

def print_optimisation_report(study: optuna.Study, symbol: str) -> Dict[str, Any]:
    """
    Print the optimisation results and return the best parameters.
    """
    best = study.best_trial

    print(f"\n  {'=' * 60}")
    print(f"  🏆 OPTIMISATION RESULTS: {symbol}")
    print(f"  {'=' * 60}")
    print(f"  Best Sharpe Ratio : {best.value:.4f}")
    print(f"  Trials completed  : {len(study.trials)}")

    win_rate = best.user_attrs.get("win_rate", 0)
    total_pnl = best.user_attrs.get("total_pnl", 0)
    max_dd = best.user_attrs.get("max_drawdown", 0)
    pf = best.user_attrs.get("profit_factor", 0)
    n_trades = best.user_attrs.get("n_trades", 0)

    print(f"\n  📊 Best Trial Performance:")
    print(f"     Win Rate       : {win_rate:.1%}")
    print(f"     Total P&L      : {total_pnl:+.2%}")
    print(f"     Max Drawdown   : {max_dd:.2%}")
    print(f"     Profit Factor  : {pf:.2f}x")
    print(f"     Trades         : {n_trades}")

    print(f"\n  ⚙️  Best Parameters:")
    params = best.params
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            print(f"     {key:20s} : {value:.6f}")
        else:
            print(f"     {key:20s} : {value}")

    # ---- Actionable config suggestions ----
    print(f"\n  📝 Suggested .env Configuration:")
    print(f"     STOP_LOSS_PCT={params.get('sl_pct', 0.005)}")
    print(f"     TAKE_PROFIT_PCT={params.get('tp_pct', 0.01)}")
    print(f"     ML_THRESHOLD={params.get('threshold', 0.65)}")

    result = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_sharpe": best.value,
        "best_params": params,
        "performance": {
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "max_drawdown": max_dd,
            "profit_factor": pf,
            "n_trades": n_trades,
        },
        "n_trials": len(study.trials),
    }

    return result


def save_results(results: list) -> None:
    """Save optimisation results to JSON file."""
    existing = []
    if _RESULTS_FILE.exists():
        try:
            existing = json.loads(_RESULTS_FILE.read_text())
        except Exception:
            pass

    existing.extend(results)
    _RESULTS_FILE.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\n  💾 Results saved to {_RESULTS_FILE}")


# ============================================================================
# Main
# ============================================================================

async def run_optimisation(
    pairs: list,
    timeframe: str,
    candle_count: int,
    n_trials: int,
    n_jobs: int,
    timeout: Optional[int],
) -> None:
    """Async entry point for the optimisation pipeline."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print(f"\n{'=' * 64}")
    print(f"  🧬 OPTUNA HYPERPARAMETER OPTIMIZER — \"The Solver\"")
    print(f"{'=' * 64}")
    print(f"  Pairs       : {', '.join(pairs)}")
    print(f"  Timeframe   : {timeframe}")
    print(f"  Candles     : {candle_count:,}")
    print(f"  Trials      : {n_trials}")
    print(f"  Workers     : {n_jobs if n_jobs > 0 else 'all CPUs'}")
    if timeout:
        print(f"  Timeout     : {timeout}s")
    print(f"  Objective   : Maximise Sharpe Ratio")
    print(f"  Started     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    db = Database()
    await db.initialise()
    data_engine = DataEngine()
    await data_engine.initialise()

    all_results = []

    for symbol in pairs:
        print(f"\n  ⬇️  Downloading {candle_count:,} candles for {symbol} [{timeframe}] ...")
        df = await data_engine.fetch_ohlcv_history(
            symbol=symbol, timeframe=timeframe, limit=candle_count,
        )

        if len(df) < _MIN_CANDLES:
            print(f"  ❌ Only {len(df)} candles — need >= {_MIN_CANDLES}. Skipping.")
            continue

        # ---- Enrich with historical derivatives data (closes serving skew) ----
        if cfg.DERIVATIVES_DATA_ENABLED:
            print(f"  🧬 Enriching with historical funding/OI ...")
            df = await data_engine.enrich_training_data(df, symbol)

        print(f"  📊 Got {len(df):,} candles ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")

        t0 = time.time()
        study = create_study(
            df=df, symbol=symbol, db=db,
            n_trials=n_trials, n_jobs=n_jobs, timeout=timeout,
        )
        elapsed = time.time() - t0

        result = print_optimisation_report(study, symbol)
        result["elapsed_seconds"] = elapsed
        all_results.append(result)

        print(f"\n  ⏱️  Optimisation took {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Save all results
    if all_results:
        save_results(all_results)

    # ---- Training Plan ----
    print(f"\n{'=' * 64}")
    print(f"  📋 1-WEEK PERFECTION TRAINING PLAN")
    print(f"{'=' * 64}")
    print(f"""
  Day 1-2: Initial Optimisation (you are here)
  ─────────────────────────────────────────────
  Run the optimizer with increasing data:
    python optimizer.py --trials 500 --candles 20000 --jobs -1
    python optimizer.py --trials 500 --candles 30000 --jobs -1
  
  Day 3-4: Deep Search with Best Settings
  ────────────────────────────────────────
  Take the best params from Day 1-2 and do targeted search:
    python optimizer.py --trials 1000 --candles 40000 --jobs -1
  
  Review optimizer_results.json for the best Sharpe ratio.
  Update your .env with the suggested parameters.
  
  Day 5: Train the Full Ensemble
  ──────────────────────────────
  With optimised params now in .env, train the full 5-model ensemble:
    python train.py --candles 40000
  
  Day 6: Validation (Paper Trading)
  ─────────────────────────────────
  Start the bot in paper mode with a dry run:
    python main.py --dry-run
  
  Then let it run for 24h in paper mode:
    python main.py
  
  Day 7: Review & Decision
  ────────────────────────
  Check trading_bot.log and the database for:
    - Number of trades taken
    - Win rate
    - Sharpe ratio
    - Max drawdown
  
  If paper results match backtest results (±20%), the model is
  validated and ready for live deployment.
  
  ⚠️  CRITICAL: Never go live without at least 48h of paper trading.
""")

    await data_engine.close()
    await db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimizer for AI Trading Bot",
    )
    parser.add_argument("--pairs", nargs="+", default=cfg.TRADING_PAIRS,
                        help="Trading pairs to optimise")
    parser.add_argument("--timeframe", default=cfg.TIMEFRAME,
                        help="Candle timeframe")
    parser.add_argument("--candles", type=int, default=max(cfg.ML_TRAINING_CANDLES, 20000),
                        help="Number of historical candles")
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of Optuna trials")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel workers (-1 = all CPUs)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max seconds for optimisation")
    args = parser.parse_args()

    try:
        asyncio.run(run_optimisation(
            pairs=args.pairs,
            timeframe=args.timeframe,
            candle_count=args.candles,
            n_trials=args.trials,
            n_jobs=args.jobs,
            timeout=args.timeout,
        ))
    except KeyboardInterrupt:
        print("\nOptimisation interrupted.")


if __name__ == "__main__":
    main()
