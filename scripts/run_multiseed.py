"""Multi-seed robustness protocol for all 8 non-baseline models.

Trains Linear Regression, Random Forest, HistGradientBoosting, ExtraTrees,
MLP, CatBoost, LightGBM (tuned), and XGBoost (tuned) across 5 random seeds
each, evaluates on the fixed 2025 chronological test set, and reports
mean +/- std for R2 / MAE / RMSE / MAPE.

Mirrors exactly the code in notebooks/04_model_training_and_evaluation_revised.ipynb,
section "2.9 Multi-Seed Robustness Protocol":
  - cells for Linear Regression / Random Forest / HistGradientBoosting /
    ExtraTrees / MLP (already run in this sandbox, see
    models/diagnostics/multiseed_results.csv, models/diagnostics/multiseed_summary.csv)
  - cells for CatBoost and for XGBoost/LightGBM (tuned) (NOT run in this
    sandbox: no network, catboost/lightgbm/xgboost not installed here; the
    notebook cells are written but left with empty outputs and a "CHUA CHAY"
    marker). Their hyperparameters below are hardcoded copies of
    config.CATBOOST_PARAMS / config.XGB_TUNED_PARAMS / config.LGBM_TUNED_PARAMS
    at the time of writing -- if those change in config.py, update both this
    file and the two notebook cells together so all three stay in sync.

CatBoost/LightGBM/XGBoost require `pip install catboost lightgbm xgboost`.
Run this file on a machine that has network + those three libraries to
produce the full 8-model table; it will overwrite
models/diagnostics/multiseed_results.csv and multiseed_summary.csv with all
8 models included.

Linear Regression has no random_state parameter (OLS is deterministic), so
it is run once and reported with std = 0.0 for transparency rather than
omitted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demandforecast import config, metrics, splits  # noqa: E402

SEEDS = [42, 7, 123, 2024, 99]

# Copied from config.py so this script's CatBoost/LightGBM/XGBoost branches
# match the notebook 2.9 cells exactly (which are self-contained and do not
# import demandforecast.config -- see notebook markdown note in that section).
CATBOOST_PARAMS = dict(
    iterations=1000, learning_rate=0.05, depth=8, subsample=0.8,
    colsample_bylevel=0.8, random_state=42, early_stopping_rounds=50, verbose=False,
)
XGB_TUNED_PARAMS = dict(
    objective="reg:squarederror", n_estimators=1000, learning_rate=0.03, max_depth=6,
    subsample=0.9, colsample_bytree=0.7, min_child_weight=5, random_state=42,
    n_jobs=-1, early_stopping_rounds=50,
)
LGBM_TUNED_PARAMS = dict(
    objective="regression", n_estimators=1000, learning_rate=0.03, max_depth=6,
    num_leaves=63, subsample=0.9, colsample_bytree=0.7, min_child_samples=30,
    random_state=42, n_jobs=-1, verbosity=-1,
)

ALL_MODEL_NAMES = [
    "Linear Regression", "Random Forest", "HistGradientBoosting", "ExtraTrees", "MLP",
    "CatBoost", "XGBoost (tuned)", "LightGBM (tuned)",
]


def build_model(name: str, seed: int):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        params = dict(config.RF_PARAMS)
        params["random_state"] = seed
        return RandomForestRegressor(**params)
    if name == "HistGradientBoosting":
        params = dict(config.HISTGB_PARAMS)
        params["random_state"] = seed
        return HistGradientBoostingRegressor(**params)
    if name == "ExtraTrees":
        params = dict(config.EXTRATREES_PARAMS)
        params["random_state"] = seed
        return ExtraTreesRegressor(**params)
    if name == "MLP":
        params = dict(config.MLP_PARAMS)
        params["random_state"] = seed
        return make_pipeline(StandardScaler(), MLPRegressor(**params))
    if name == "CatBoost":
        from catboost import CatBoostRegressor  # requires: pip install catboost

        params = dict(CATBOOST_PARAMS)
        params["random_state"] = seed
        return CatBoostRegressor(**params)
    if name == "XGBoost (tuned)":
        import xgboost as xgb  # requires: pip install xgboost

        params = dict(XGB_TUNED_PARAMS)
        params["random_state"] = seed
        return xgb.XGBRegressor(**params)
    if name == "LightGBM (tuned)":
        import lightgbm as lgb  # requires: pip install lightgbm

        params = dict(LGBM_TUNED_PARAMS)
        params["random_state"] = seed
        return lgb.LGBMRegressor(**params)
    raise ValueError(name)


def _needs_valid_set(name: str) -> bool:
    """CatBoost/XGBoost/LightGBM here use early stopping against the 2024
    validation split, exactly like their single-seed training in notebook 04
    section 2.2/2.3/2.4 and in models.py. The five sklearn-only models do not
    use early stopping."""
    return name in {"CatBoost", "XGBoost (tuned)", "LightGBM (tuned)"}


def main() -> None:
    df = splits.load_feature_matrix(config.FEATURES_CSV)
    split = splits.chronological_split(df)
    X_train, y_train = splits.xy(split.train, split)
    X_valid, y_valid = splits.xy(split.valid, split)
    X_test, y_test = splits.xy(split.test, split)

    rows = []
    skipped = []
    t0 = time.time()
    for name in ALL_MODEL_NAMES:
        seeds_for_model = [SEEDS[0]] if name == "Linear Regression" else SEEDS
        for seed in seeds_for_model:
            try:
                model = build_model(name, seed)
            except ImportError as exc:
                print(f"{name:22s} SKIPPED (library not installed: {exc})")
                skipped.append(name)
                break  # no point retrying remaining seeds for this model
            if _needs_valid_set(name):
                if name == "CatBoost":
                    model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                              use_best_model=True, verbose=False)
                else:
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model.fit(X_train, y_train)
            yhat = model.predict(X_test)
            row = metrics.ml_error(name, y_test.to_numpy(), np.asarray(yhat))
            row["Seed"] = seed
            rows.append(row)
            print(f"{name:22s} seed={seed:5d}  MAPE={row['MAPE (%)'][0]:.4f}%  "
                  f"RMSE={row['RMSE'][0]:.1f}  R2={row['R2'][0]:.4f}  "
                  f"[{time.time()-t0:.0f}s elapsed]")

    if not rows:
        print("Nothing ran.")
        return

    detail = pd.concat(rows, ignore_index=True)
    out_dir = ROOT / "models" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "multiseed_results.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby("Model Name")
        .agg(
            n_seeds=("Seed", "nunique"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_mean=("MAPE (%)", "mean"),
            MAPE_std=("MAPE (%)", "std"),
        )
        .fillna(0.0)
        .reset_index()
    )
    summary_path = out_dir / "multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Multi-seed summary (mean +/- std over seeds) ===")
    for _, r in summary.sort_values("MAPE_mean").iterrows():
        print(
            f"{r['Model Name']:22s} n={int(r['n_seeds'])}  "
            f"MAPE={r['MAPE_mean']:.2f}+/-{r['MAPE_std']:.2f}%  "
            f"RMSE={r['RMSE_mean']:.1f}+/-{r['RMSE_std']:.1f}  "
            f"R2={r['R2_mean']:.4f}+/-{r['R2_std']:.4f}"
        )
    print(f"\nWrote {detail_path}\nWrote {summary_path}")
    if skipped:
        print(f"\nSkipped (library not installed in this environment): {', '.join(skipped)}")
        print("Install with: pip install catboost lightgbm xgboost")


if __name__ == "__main__":
    main()
