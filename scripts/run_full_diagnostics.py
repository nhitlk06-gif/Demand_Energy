#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from demandforecast import config, models  
from demandforecast.features import (  
    add_cyclical_calendar_features,
    add_lag_features,
    add_rolling_features,
    drop_leaky_exogenous_columns,
    load_clean_series,
    select_features_by_mutual_information,
    trim_and_drop_administrative_columns,
)
from demandforecast.forecast import ForecastArtifacts, evaluate_recursive_horizon 
from demandforecast.metrics import diebold_mariano_test, ml_error, prediction_interval_coverage  
from demandforecast.splits import chronological_split, xy  
from demandforecast.train import train_quantile_models  

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_full_diagnostics")

OUT_DIR = config.MODELS_DIR / "diagnostics"



# 1. Three-config comparison (review Table 0 / item 8's "3 cau hinh")
def build_three_configs() -> dict[str, pd.DataFrame]:
    raw = load_clean_series(config.CLEANED_CSV)

    configs: dict[str, pd.DataFrame] = {}

    # with_short_lags: temporarily lag ND at 1 and 2 as well
    df = raw.copy()
    for h in [1, 2] + config.SHORT_TERM_LAGS:
        df[f"{config.PRIMARY_TARGET}_LAG_{h}"] = df[config.PRIMARY_TARGET].shift(h)
    df[f"{config.PRIMARY_TARGET}_LAG_{config.WEEKLY_LAG}"] = df[config.PRIMARY_TARGET].shift(config.WEEKLY_LAG)
    df = add_rolling_features(df)
    df = add_cyclical_calendar_features(df)
    # contemporaneous exogenous columns stay in this config too
    df_trimmed = trim_and_drop_administrative_columns(df)
    kept, _, _ = select_features_by_mutual_information(df_trimmed)
    configs["with_short_lags"] = df_trimmed[list(config.TARGET_COLUMNS) + kept].dropna()

    # contemporaneous_exogenous: today's AR fix, leak fix NOT applied 
    df = add_lag_features(raw.copy())
    df = add_rolling_features(df)
    df = add_cyclical_calendar_features(df)
    df_trimmed = trim_and_drop_administrative_columns(df)
    kept, _, _ = select_features_by_mutual_information(df_trimmed)
    configs["contemporaneous_exogenous"] = df_trimmed[list(config.TARGET_COLUMNS) + kept].dropna()

    # leak_removed: both fixes applied (scripts/run_features.py) 
    df = add_lag_features(raw.copy())
    df = add_rolling_features(df)
    df = add_cyclical_calendar_features(df)
    df = drop_leaky_exogenous_columns(df)
    df_trimmed = trim_and_drop_administrative_columns(df)
    kept, _, _ = select_features_by_mutual_information(df_trimmed)
    configs["leak_removed"] = df_trimmed[list(config.TARGET_COLUMNS) + kept].dropna()

    return configs


def run_three_config_comparison() -> pd.DataFrame:
    logger.info("1/6: three-config comparison (LightGBM) ===")
    try:
        import lightgbm as lgb 
    except ImportError:
        logger.warning("lightgbm not installed - skipping three-config comparison. `pip install lightgbm`.")
        return pd.DataFrame()

    rows = []
    for name, df in build_three_configs().items():
        split = chronological_split(df)
        X_train, y_train = xy(split.train, split)
        X_valid, y_valid = xy(split.valid, split)
        X_test, y_test = xy(split.test, split)

        model = models.lightgbm_model(tuned=True)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        yhat = model.predict(X_test)

        err = ml_error(f"LightGBM Tuned - {name}", y_test, yhat)
        err["n_features"] = len(split.feature_columns)
        rows.append(err)
        logger.info("%s: MAPE=%.2f%% R2=%.4f (%d features)", name, err["MAPE (%)"][0], err["R2"][0], len(split.feature_columns))

    result = pd.concat(rows, ignore_index=True)
    return result



# 2. Five-model comparison on the (already leak-removed) default features
def run_five_model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logger.info("2/6: five-model comparison on leak-removed features ")
    split = chronological_split(df)
    X_train, y_train = xy(split.train, split)
    X_valid, y_valid = xy(split.valid, split)
    X_test, y_test = xy(split.test, split)

    rows = []
    fitted = {}

    def _try(key, ctor, display, needs_eval_set=False, is_catboost=False):
        t0 = time.time()
        try:
            model = ctor()
            if is_catboost:
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            elif needs_eval_set:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            else:
                model.fit(X_train, y_train)
        except ImportError as exc:
            logger.warning("Skipping %s: %s", display, exc)
            return
        fitted[key] = model
        rows.append(ml_error(display, y_test, model.predict(X_test)))
        logger.info("%s trained in %.1fs", display, time.time() - t0)

    _try("catboost", models.catboost_model, "CatBoost", is_catboost=True)
    _try("lightgbm", lambda: models.lightgbm_model(), "LightGBM", needs_eval_set=True)
    _try("histgradientboosting", models.histgradientboosting_model, "HistGradientBoosting")
    _try("xgboost", lambda: models.xgboost_model(), "XGBoost", needs_eval_set=True)
    _try("extratrees", models.extratrees_model, "ExtraTrees")
    _try("mlp", models.mlp_model, "MLP")

    if not rows:
        return pd.DataFrame(), fitted
    summary = pd.concat(rows, ignore_index=True).sort_values("RMSE").reset_index(drop=True)
    return summary, fitted



# 3. Diebold-Mariano pairwise tests
def run_dm_tests(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    logger.info("3/6: Diebold-Mariano pairwise tests")
    if not fitted:
        return pd.DataFrame()
    split = chronological_split(df)
    X_test, y_test = xy(split.test, split)
    y = y_test.values

    preds = {config.MODEL_DISPLAY_NAMES.get(k, k): m.predict(X_test) for k, m in fitted.items()}
    preds[models.snaive_daily().name] = models.snaive_daily().predict(X_test).values
    preds[models.snaive_weekly().name] = models.snaive_weekly().predict(X_test).values

    names = list(preds)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            result = diebold_mariano_test(y, preds[a], preds[b])
            rows.append({"model_a": a, "model_b": b, **result})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)



# 4. Quantile coverage
def run_quantile_coverage(df: pd.DataFrame) -> dict | None:
    logger.info("=== 4/6: LightGBM quantile-interval coverage (PICP) ===")
    try:
        fitted = train_quantile_models(df, OUT_DIR, fast_mode=False)
    except ImportError as exc:
        logger.warning("Skipping quantile coverage: %s", exc)
        return None
    if fitted is None:
        return None
    coverage_path = OUT_DIR / "quantile_coverage.json"
    return json.loads(coverage_path.read_text()) if coverage_path.exists() else None



# 5. Recursive 48-step horizon evaluation
def run_recursive_horizon() -> pd.DataFrame:
    logger.info("5/6: recursive 48-step horizon evaluation")
    from demandforecast.forecast import load_artifacts

    artifacts = load_artifacts()
    if not artifacts.models:
        logger.warning("No trained models found in models/ run scripts/run_training.py first. Skipping.")
        return pd.DataFrame()

    preferred = next(
        (k for k in ("lightgbm_tuned", "xgboost_tuned", "catboost", "lightgbm", "xgboost", "histgradientboosting")
         if k in artifacts.models),
        next(iter(artifacts.models)),
    )
    result = evaluate_recursive_horizon(artifacts, n_steps=48, n_start_points=20, model_key=preferred)
    return result


# 6. Hypothesis-4 re-read: lagged vs. contemporaneous renewable ablation

def run_hypothesis4_reread(df_leak_removed: pd.DataFrame) -> pd.DataFrame:
    logger.info("6/6: Hypothesis-4 re-read (lagged vs. contemporaneous renewables)")
    try:
        import xgboost as xgb  # noqa: F401
    except ImportError:
        logger.warning("xgboost not installed - skipping hypothesis-4 re-read.")
        return pd.DataFrame()

    raw = load_clean_series(config.CLEANED_CSV)
    rows = []

    def _fit_eval(df: pd.DataFrame, label: str) -> float:
        split = chronological_split(df.dropna())
        X_train, y_train = xy(split.train, split)
        X_test, y_test = xy(split.test, split)
        model = models.xgboost_model()
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        mae = float(np.mean(np.abs(y_test.values - model.predict(X_test))))
        rows.append({"config": label, "mae": mae, "n_features": len(split.feature_columns)})
        return mae

    # Baseline: leak-removed matrix as-is (no raw renewables at all).
    base_mae = _fit_eval(df_leak_removed, "leak_removed_baseline")

    # With renewables lagged by 12 periods (legitimately available 6h ahead).
    df_with_lagged_renewables = add_lag_features(raw.copy())
    df_with_lagged_renewables = add_rolling_features(df_with_lagged_renewables)
    df_with_lagged_renewables = add_cyclical_calendar_features(df_with_lagged_renewables)
    for col in config.RENEWABLE_GENERATION_COLUMNS:
        df_with_lagged_renewables[f"{col}_LAG_12"] = df_with_lagged_renewables[col].shift(config.SHORT_TERM_LAGS[0])
    df_with_lagged_renewables = drop_leaky_exogenous_columns(df_with_lagged_renewables)
    df_with_lagged_renewables = trim_and_drop_administrative_columns(df_with_lagged_renewables)
    lagged_mae = _fit_eval(df_with_lagged_renewables, "with_renewables_lagged_12")

    # With renewables contemporaneous (the paper's original ablation target).
    df_contemporaneous = add_lag_features(raw.copy())
    df_contemporaneous = add_rolling_features(df_contemporaneous)
    df_contemporaneous = add_cyclical_calendar_features(df_contemporaneous)
    df_contemporaneous = trim_and_drop_administrative_columns(df_contemporaneous)
    contemporaneous_mae = _fit_eval(df_contemporaneous, "with_renewables_contemporaneous")

    pct_from_lagged = 100 * (lagged_mae - base_mae) / base_mae if base_mae else float("nan")
    pct_from_contemporaneous = 100 * (contemporaneous_mae - base_mae) / base_mae if base_mae else float("nan")
    logger.info(
        "MAE change vs. leak-removed baseline: lagged renewables %.1f%%, contemporaneous renewables %.1f%% "
        "(paper reported 39.01%% using the contemporaneous/leaky version)",
        pct_from_lagged, pct_from_contemporaneous,
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    three_config = run_three_config_comparison()
    if not three_config.empty:
        three_config.to_csv(OUT_DIR / "three_config_comparison.csv", index=False)

    df = pd.read_csv(config.FEATURES_CSV)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.set_index("DATETIME").sort_index()

    five_model, fitted = run_five_model_comparison(df)
    if not five_model.empty:
        five_model.to_csv(OUT_DIR / "five_model_comparison.csv", index=False)

    dm = run_dm_tests(df, fitted)
    if not dm.empty:
        dm.to_csv(OUT_DIR / "dm_tests.csv", index=False)

    quantile_coverage = run_quantile_coverage(df)

    horizon = run_recursive_horizon()
    if not horizon.empty:
        horizon.to_csv(OUT_DIR / "recursive_horizon_48step.csv", index=False)
        by_h = horizon.assign(ape=lambda d: (d.actual_mw - d.forecast_mw).abs() / d.actual_mw * 100).groupby("h")["ape"].mean()
        logger.info("MAPE by horizon step (first 6, then every 12th):\n%s", by_h.iloc[list(range(6)) + list(range(11, 48, 12))])

    h4 = run_hypothesis4_reread(df)
    if not h4.empty:
        h4.to_csv(OUT_DIR / "hypothesis4_reread.csv", index=False)

    logger.info("All diagnostics written to %s", OUT_DIR)
    logger.info(
        "Next: fold five_model_comparison.csv into models/metrics_summary.csv "
        "(or just re-run scripts/run_training.py, which now trains all 10 models), "
        "then update paper/main.tex's Abstract/Table 1/Discussion (see TODO comments)."
    )


if __name__ == "__main__":
    main()
