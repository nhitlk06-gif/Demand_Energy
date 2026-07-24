"""Inference utilities: load trained artifacts, produce forward forecasts,
and backtest models against real held-out history.

This module is what the FastAPI service (``app/api.py``) calls. It supports
three things:

1. ``load_artifacts``  load the trained models + the list of feature
   columns they expect, plus enough recent history to build features from.
2. ``forecast_horizon``  recursively forecast ``ND`` (National Demand,
   in MW) ``n_steps`` settlement periods (each 30 minutes) into the future
   beyond the end of the available history, feeding each prediction back
   in as if it were an observation for the next step's lag/rolling
   features (a standard recursive/direct-multistep strategy for
   autoregressive feature sets).
3. ``backtest_predictions`` run a trained model over the held-out 2025
   test set and return *actual vs predicted* National Demand side by
   side, so the dashboard/API can plot the predicted curve directly
   against the real observed demand curve (as opposed to a forecast that
   runs beyond the end of history, where there is no ground truth yet).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from . import config
from .features import build_feature_row_for_inference, load_clean_series
from .splits import chronological_split, load_feature_matrix, xy

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = config.MODEL_DISPLAY_NAMES


@dataclass
class ForecastArtifacts:
    models: Dict[str, object]
    feature_columns: List[str]
    history: pd.DataFrame  # tail of the cleaned series, target + exogenous cols
    metrics_summary: Optional[List[dict]]
    full_features: Optional[pd.DataFrame] = None  # full feature matrix, for backtesting

    def available_models(self) -> List[str]:
        return list(self.models.keys())


def _load_history(models_dir: Path) -> pd.DataFrame:
    """Prefer the full feature matrix (has everything precomputed); fall
    back to the cleaned series if the feature matrix isn't available.
    """
    if config.FEATURES_CSV.exists():
        df = load_feature_matrix(config.FEATURES_CSV)
        return df
    if config.CLEANED_CSV.exists():
        return load_clean_series(config.CLEANED_CSV)
    raise FileNotFoundError(
        "Neither the feature matrix nor the cleaned CSV were found. "
        "Run the cleaning and feature pipelines first."
    )


def load_artifacts(models_dir: Optional[Path] = None) -> ForecastArtifacts:
    models_dir = Path(models_dir or config.MODELS_DIR)

    fitted = {}
    for key, filename in config.MODEL_FILENAMES.items():
        path = models_dir / filename
        if not path.exists():
            continue
        try:
            fitted[key] = joblib.load(path)
        except ImportError as exc:
            # Scatboost/lightgbm/xgboost .pkl files need their own
            # library to unpickle. Previously this had no try/except, so a
            # missing library (e.g. catboost not installed in a deployment
            # environment) crashed the whole app/API at startup instead of
            # just omitting that one model -- the same failure mode
            # train.py's train_additional_models() already guards against
            # for training, but load_artifacts() did not for inference.
            logger.warning(
                "Skipping model '%s' (%s): %s -- install the missing package "
                "to make this model selectable via the API/dashboard.",
                key, filename, exc,
            )

    feature_columns_path = models_dir / "feature_columns.json"
    if feature_columns_path.exists():
        feature_columns = json.loads(feature_columns_path.read_text())
    else:
        feature_columns = [c for c in _load_history(models_dir).columns if c not in config.TARGET_COLUMNS]

    metrics_summary = None
    metrics_path = models_dir / "metrics_summary.json"
    if metrics_path.exists():
        metrics_summary = json.loads(metrics_path.read_text())

    history = _load_history(models_dir)

    return ForecastArtifacts(
        models=fitted,
        feature_columns=feature_columns,
        history=history,
        metrics_summary=metrics_summary,
        full_features=history,
    )


def _select_model(artifacts: ForecastArtifacts, model_key: Optional[str]):
    if not artifacts.models:
        raise RuntimeError("No trained models are available. Run the training pipeline first.")

    if model_key is None:
        if artifacts.metrics_summary:
            best_name = artifacts.metrics_summary[0]["Model Name"]
            for key, display in MODEL_DISPLAY_NAMES.items():
                if display == best_name and key in artifacts.models:
                    return key, artifacts.models[key]
        for preferred in (
            "catboost", "lightgbm_tuned", "lightgbm", "extratrees",
            "histgradientboosting", "xgboost_tuned", "xgboost", "mlp",
            "random_forest", "linear_regression",
        ):
            if preferred in artifacts.models:
                return preferred, artifacts.models[preferred]
        key = next(iter(artifacts.models))
        return key, artifacts.models[key]

    if model_key not in artifacts.models:
        raise KeyError(
            f"Model '{model_key}' is not available. Options: {list(artifacts.models.keys())}"
        )
    return model_key, artifacts.models[model_key]


def forecast_horizon(
    artifacts: ForecastArtifacts,
    n_steps: int,
    model_key: Optional[str] = None,
    context_periods: int = config.WEEKLY_LAG + 48,
) -> pd.DataFrame:
    """Recursively forecast ``n_steps`` settlement periods (30 min each)
    beyond the end of the stored history using the chosen model.

    Returns a DataFrame indexed by DATETIME with a single ``forecast_mw``
    column (National Demand, MW).
    """
    used_key, model = _select_model(artifacts, model_key)

    history = artifacts.history.tail(context_periods).copy()
    if config.PRIMARY_TARGET not in history.columns:
        raise ValueError("History is missing the primary target column 'ND'.")

    # Ensure the other two target columns exist for lag-feature purposes;
    # if unavailable, approximate with ND (they move almost in lock-step).
    for col in config.TARGET_COLUMNS:
        if col not in history.columns:
            history[col] = history[config.PRIMARY_TARGET]

    predictions = []
    working_history = history.copy()

    for _ in range(n_steps):
        feature_row = build_feature_row_for_inference(working_history, artifacts.feature_columns)
        X_next = feature_row.to_frame().T
        yhat = float(model.predict(X_next)[0])

        next_time = feature_row.name
        predictions.append((next_time, yhat))

        # Only the primary target (ND) is fed back as "the prediction" --
        # TSD and ENGLAND_WALES_DEMAND are no longer autoregressively lagged
        # (see config.AR_LAG_TARGETS), so they must NOT be overwritten with
        # the ND prediction here. Doing so used to silently corrupt
        # EWD_LAG_1 at every recursive step, since TSD/EWD actually sit a
        # near-constant ~2000-2400 MW away from ND. They now fall through to
        # the same same-period-yesterday persistence used for every other
        # raw column below, which is a far better proxy for their own level.
        new_row = {config.PRIMARY_TARGET: yhat}
        new_row["SETTLEMENT_PERIOD"] = feature_row.get("SETTLEMENT_PERIOD")

        # Carry forward exogenous (non-target) raw columns too, or later
        # rolling/lag windows will slide into NaN-filled predicted rows.
        # Approximate with same-period-yesterday persistence, falling back
        # to the last known observation.
        for col in working_history.columns:
            if col in new_row:
                continue
            series = working_history[col]
            if len(series) >= config.STANDARD_PERIODS_PER_DAY:
                new_row[col] = series.iloc[-config.STANDARD_PERIODS_PER_DAY]
            else:
                new_row[col] = series.iloc[-1]

        working_history.loc[next_time] = pd.Series(new_row)

    result = pd.DataFrame(predictions, columns=["datetime", "forecast_mw"]).set_index("datetime")
    result.attrs["model_used"] = used_key
    return result


def evaluate_recursive_horizon(
    artifacts: ForecastArtifacts,
    n_steps: int = 48,
    n_start_points: int = 20,
    model_key: Optional[str] = None,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    import numpy as np

    if artifacts.full_features is None:
        raise RuntimeError("Full feature matrix is not loaded; cannot backtest.")

    used_key, model = _select_model(artifacts, model_key)
    target = config.PRIMARY_TARGET

    full = artifacts.history
    if target not in full.columns:
        raise ValueError(f"History is missing the primary target column '{target}'.")

    min_context = config.WEEKLY_LAG + 48
    test = full.loc[config.TEST_START : config.TEST_END]
    valid_starts = test.index[test.index.get_indexer(test.index) >= 0]
    # A start point needs min_context rows of prior history AND n_steps rows
    # of future ground truth to compare against.
    usable = [
        t for t in valid_starts
        if full.index.get_loc(t) >= min_context and (full.index.get_loc(t) + n_steps) < len(full.index)
    ]
    if not usable:
        raise ValueError("Not enough history/ground-truth around the test set to backtest recursively.")

    rng = np.random.RandomState(random_state)
    chosen = rng.choice(usable, size=min(n_start_points, len(usable)), replace=False)

    rows = []
    for start in chosen:
        start_loc = full.index.get_loc(start)
        history_slice = full.iloc[start_loc - min_context : start_loc].copy()

        local_artifacts = ForecastArtifacts(
            models={used_key: model},
            feature_columns=artifacts.feature_columns,
            history=history_slice,
            metrics_summary=None,
            full_features=None,
        )
        fc = forecast_horizon(local_artifacts, n_steps=n_steps, model_key=used_key, context_periods=min_context)

        future_actual = full[target].iloc[start_loc : start_loc + n_steps]
        yesterday_actual = full[target].iloc[start_loc - 48 : start_loc - 48 + n_steps]
        snaive_actual = full[target].iloc[start_loc - 48 : start_loc]  # same-period, 1 day back, per-step

        for h in range(n_steps):
            rows.append(
                {
                    "start": start,
                    "h": h + 1,
                    "actual_mw": float(future_actual.iloc[h]),
                    "forecast_mw": float(fc["forecast_mw"].iloc[h]),
                    "snaive_daily_mw": float(snaive_actual.iloc[h]) if h < len(snaive_actual) else np.nan,
                    "copy_yesterday_mw": float(yesterday_actual.iloc[h]) if h < len(yesterday_actual) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def evaluate_on_test_set(artifacts: ForecastArtifacts, model_key: Optional[str] = None) -> dict:
    """Return the stored held-out test metrics for the chosen (or best) model."""
    used_key, _ = _select_model(artifacts, model_key)
    display_name = MODEL_DISPLAY_NAMES.get(used_key, used_key)
    if artifacts.metrics_summary:
        for row in artifacts.metrics_summary:
            if row["Model Name"] == display_name:
                return {"model": display_name, **row}
    return {"model": display_name, "message": "No stored metrics found; run the training pipeline."}


def backtest_predictions(
    artifacts: ForecastArtifacts,
    model_key: Optional[str] = None,
    split_name: str = "test",
    max_points: Optional[int] = None,
) -> pd.DataFrame:
    if artifacts.full_features is None:
        raise RuntimeError("Full feature matrix is not loaded; cannot backtest.")

    used_key, model = _select_model(artifacts, model_key)

    split = chronological_split(artifacts.full_features)
    frame = {"train": split.train, "valid": split.valid, "test": split.test}.get(split_name)
    if frame is None or frame.empty:
        raise ValueError(f"Unknown or empty split '{split_name}'. Options: train, valid, test.")

    X, y = xy(frame, split)
    X = X[artifacts.feature_columns]
    yhat = model.predict(X)

    result = pd.DataFrame({"actual_mw": y.values, "predicted_mw": yhat}, index=frame.index)
    result.attrs["model_used"] = used_key

    if max_points and len(result) > max_points:
        # Downsample evenly for lightweight plotting while preserving shape,
        # guaranteeing the result never exceeds max_points rows.
        import numpy as np

        positions = np.linspace(0, len(result) - 1, max_points).round().astype(int)
        positions = sorted(set(positions.tolist()))
        result = result.iloc[positions]

    return result


if __name__ == "__main__":
    pass
