"""End-to-end model training: load the feature matrix, split
chronologically, fit every model (base + random-search-tuned boosting
variants), evaluate on the 2025 held-out test set, and persist both the
fitted estimators (``models/*.pkl``) and a metrics summary
(``models/metrics_summary.csv`` / ``.json``).

Converted from ``notebooks/04_model_training_and_evaluation_revised.ipynb``
(section 2 baselines, Linear Regression, Random Forest, XGBoost,
LightGBM; section 3 random-search hyperparameter tuning of the two
boosting models, exposed here as :func:`tune_boosting_models`).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from . import config, models
from .metrics import ml_error
from .splits import chronological_split, load_feature_matrix, xy

logger = logging.getLogger(__name__)


def train_base_models(
    df: pd.DataFrame,
    models_dir: Path,
    fast_mode: bool = False,
) -> Dict[str, object]:
    """Fit Linear Regression, Random Forest, XGBoost and LightGBM (base
    hyperparameters) on the training split and return the fitted objects.
    """
    split = chronological_split(df)
    X_train, y_train = xy(split.train, split)
    X_valid, y_valid = xy(split.valid, split)

    fitted: Dict[str, object] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    # Linear Regression 
    lr = models.linear_regression_model().fit(X_train, y_train)
    fitted["linear_regression"] = lr
    joblib.dump(lr, models_dir / config.MODEL_FILENAMES["linear_regression"])
    logger.info("Trained Linear Regression")

    # Random Forest 
    t0 = time.time()
    rf = models.random_forest_model(fast_mode=fast_mode).fit(X_train, y_train)
    fitted["random_forest"] = rf
    joblib.dump(rf, models_dir / config.MODEL_FILENAMES["random_forest"])
    logger.info("Trained Random Forest in %.1fs", time.time() - t0)

    # XGBoost 
    t0 = time.time()
    xgb_model = models.xgboost_model(fast_mode=fast_mode)
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    fitted["xgboost"] = xgb_model
    joblib.dump(xgb_model, models_dir / config.MODEL_FILENAMES["xgboost"])
    logger.info("Trained XGBoost in %.1fs", time.time() - t0)

    # LightGBM
    import lightgbm as lgb

    t0 = time.time()
    lgbm_model = models.lightgbm_model(fast_mode=fast_mode)
    lgbm_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    fitted["lightgbm"] = lgbm_model
    joblib.dump(lgbm_model, models_dir / config.MODEL_FILENAMES["lightgbm"])
    logger.info("Trained LightGBM in %.1fs", time.time() - t0)

    return fitted


def tune_boosting_models(
    df: pd.DataFrame,
    models_dir: Path,
    fast_mode: bool = False,
) -> Dict[str, object]:
    split = chronological_split(df)
    X_train, y_train = xy(split.train, split)
    X_valid, y_valid = xy(split.valid, split)

    fitted: Dict[str, object] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    xgb_tuned = models.xgboost_model(fast_mode=fast_mode, tuned=True)
    xgb_tuned.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    fitted["xgboost_tuned"] = xgb_tuned
    joblib.dump(xgb_tuned, models_dir / config.MODEL_FILENAMES["xgboost_tuned"])
    logger.info("Trained XGBoost (Tuned) in %.1fs", time.time() - t0)

    import lightgbm as lgb

    t0 = time.time()
    lgbm_tuned = models.lightgbm_model(fast_mode=fast_mode, tuned=True)
    lgbm_tuned.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    fitted["lightgbm_tuned"] = lgbm_tuned
    joblib.dump(lgbm_tuned, models_dir / config.MODEL_FILENAMES["lightgbm_tuned"])
    logger.info("Trained LightGBM (Tuned) in %.1fs", time.time() - t0)

    return fitted


def train_additional_models(
    df: pd.DataFrame,
    models_dir: Path,
    fast_mode: bool = False,
    skip_missing: bool = True,
) -> Dict[str, object]:

 
    split = chronological_split(df)
    X_train, y_train = xy(split.train, split)

    X_valid, y_valid = xy(split.valid, split)

    fitted: Dict[str, object] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    def _try(key: str, ctor, display: str, fit_kwargs=None):
        t0 = time.time()
        try:
            model = ctor(fast_mode=fast_mode).fit(X_train, y_train, **(fit_kwargs or {}))
        except ImportError as exc:
            if skip_missing:
                logger.warning("Skipping %s: %s (install the missing package and re-run to fill this in)", display, exc)
                return
            raise
        fitted[key] = model
        joblib.dump(model, models_dir / config.MODEL_FILENAMES[key])
        logger.info("Trained %s in %.1fs", display, time.time() - t0)

    _try(
        "catboost",
        models.catboost_model,
        "CatBoost",
        fit_kwargs=dict(eval_set=(X_valid, y_valid), use_best_model=True),
    )
    _try("histgradientboosting", models.histgradientboosting_model, "HistGradientBoosting")
    _try("extratrees", models.extratrees_model, "ExtraTrees")
    _try("mlp", models.mlp_model, "MLP")

    return fitted


def train_quantile_models(
    df: pd.DataFrame,
    models_dir: Path,
    fast_mode: bool = False,
) -> Optional[Dict[float, object]]:

    from .metrics import prediction_interval_coverage

    split = chronological_split(df)
    X_train, y_train = xy(split.train, split)
    X_test, y_test = xy(split.test, split)

    try:
        constructors = models.lightgbm_quantile_models(fast_mode=fast_mode)
    except ImportError as exc:
        logger.warning("Skipping quantile models: %s", exc)
        return None

    fitted = {}
    preds = {}
    for q, model in constructors.items():
        model.fit(X_train, y_train)
        fitted[q] = model
        preds[q] = model.predict(X_test)
        joblib.dump(model, models_dir / f"lightgbm_quantile_{str(q).replace('.', '')}.pkl")

    lo, hi = min(preds), max(preds)
    picp = prediction_interval_coverage(y_test.values, preds[lo], preds[hi])
    nominal = hi - lo
    logger.info("Quantile interval [q%s, q%s]: nominal coverage %.1f%%, actual PICP %.1f%%", lo, hi, nominal * 100, picp * 100)

    (models_dir / "quantile_coverage.json").write_text(
        json.dumps({"nominal_coverage": nominal, "picp": picp, "quantiles": list(constructors)}, indent=2)
    )
    return fitted


def compute_dm_tests(df: pd.DataFrame, fitted: Dict[str, object]) -> pd.DataFrame:
 
    from .metrics import diebold_mariano_test

    split = chronological_split(df)
    X_test, y_test = xy(split.test, split)
    y = y_test.values

    preds = {
        config.MODEL_DISPLAY_NAMES.get(k, k): m.predict(X_test) for k, m in fitted.items()
    }
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


def evaluate_all_models(df: pd.DataFrame, fitted: Dict[str, object]) -> pd.DataFrame:
    """Evaluate baselines + fitted models on the held-out 2025 test set."""
    split = chronological_split(df)
    X_test, y_test = xy(split.test, split)

    results = []

    snaive_d = models.snaive_daily()
    results.append(ml_error(snaive_d.name, y_test, snaive_d.predict(X_test)))

    snaive_w = models.snaive_weekly()
    results.append(ml_error(snaive_w.name, y_test, snaive_w.predict(X_test)))

    for key, model_obj in fitted.items():
        display = config.MODEL_DISPLAY_NAMES.get(key, key)
        yhat = model_obj.predict(X_test)
        results.append(ml_error(display, y_test, yhat))

    summary = pd.concat(results, ignore_index=True).sort_values("RMSE").reset_index(drop=True)
    return summary


def run_training_pipeline(
    features_csv: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    fast_mode: bool = False,
    include_tuned: bool = True,
    include_additional_models: bool = True,
    include_quantile_models: bool = True,
    include_dm_tests: bool = True,
) -> pd.DataFrame:
 
    features_csv = Path(features_csv or config.FEATURES_CSV)
    models_dir = Path(models_dir or config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_matrix(features_csv)
    fitted = train_base_models(df, models_dir, fast_mode=fast_mode)
    if include_tuned:
        fitted.update(tune_boosting_models(df, models_dir, fast_mode=fast_mode))
    if include_additional_models:
        fitted.update(train_additional_models(df, models_dir, fast_mode=fast_mode))

    summary = evaluate_all_models(df, fitted)

    summary.to_csv(models_dir / "metrics_summary.csv", index=False)
    summary_records = json.loads(summary.to_json(orient="records"))
    (models_dir / "metrics_summary.json").write_text(json.dumps(summary_records, indent=2))

    split = chronological_split(df)
    (models_dir / "feature_columns.json").write_text(json.dumps(split.feature_columns, indent=2))

    if include_dm_tests and fitted:
        dm = compute_dm_tests(df, fitted)
        dm.to_csv(models_dir / "dm_tests.csv", index=False)
        logger.info("Diebold-Mariano tests written to %s", models_dir / "dm_tests.csv")

    if include_quantile_models:
        train_quantile_models(df, models_dir, fast_mode=fast_mode)

    logger.info("Training complete. Leaderboard:\n%s", summary.to_string())
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training_pipeline()
