from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandforecast import config
from demandforecast.forecast import ForecastArtifacts, backtest_predictions, forecast_horizon, load_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]
_MODELS_AVAILABLE = (ROOT_DIR / "models" / "xgboost_regressor.pkl").exists()


class _PersistenceModel:
    def predict(self, X):
        return X[f"{config.PRIMARY_TARGET}_LAG_48"].to_numpy()


def _synthetic_history(n_periods: int = config.WEEKLY_LAG + 96) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_periods, freq="30min")
    period_of_day = np.arange(n_periods) % config.PERIODS_PER_DAY
    nd = 25000 + 3000 * np.sin(2 * np.pi * period_of_day / config.PERIODS_PER_DAY)
    df = pd.DataFrame(
        {
            "ND": nd,
            "TSD": nd + 2405.0,
            "ENGLAND_WALES_DEMAND": nd - 2162.0,
            "SETTLEMENT_PERIOD": (period_of_day + 1),
            "EMBEDDED_SOLAR_CAPACITY": 15000.0,
            "EMBEDDED_WIND_CAPACITY": 30000.0,
        },
        index=idx,
    )
    return df


def test_recursive_forecast_does_not_corrupt_tsd_ewd_lags():
    history = _synthetic_history()
    feature_columns = [f"{config.PRIMARY_TARGET}_LAG_{h}" for h in config.SHORT_TERM_LAGS] + [
        f"{config.PRIMARY_TARGET}_LAG_{config.WEEKLY_LAG}"
    ]
    artifacts = ForecastArtifacts(
        models={"stub": _PersistenceModel()},
        feature_columns=feature_columns,
        history=history,
        metrics_summary=None,
    )

    result = forecast_horizon(artifacts, n_steps=48, model_key="stub")
    assert len(result) == 48
    # Persistence model should reproduce yesterday's ND curve closely.
    expected = history["ND"].iloc[-48:].to_numpy()
    assert np.allclose(result["forecast_mw"].to_numpy(), expected, atol=1.0)


def test_recursive_forecast_beats_naive_smoke():
    history = _synthetic_history()
    feature_columns = [f"{config.PRIMARY_TARGET}_LAG_{h}" for h in config.SHORT_TERM_LAGS] + [
        f"{config.PRIMARY_TARGET}_LAG_{config.WEEKLY_LAG}"
    ]
    artifacts = ForecastArtifacts(
        models={"stub": _PersistenceModel()},
        feature_columns=feature_columns,
        history=history,
        metrics_summary=None,
    )
    result = forecast_horizon(artifacts, n_steps=48, model_key="stub")
    actual = history["ND"].iloc[-48:].to_numpy()
    snaive_daily = history["ND"].iloc[-96:-48].to_numpy()  # same period, 1 day earlier

    forecast_mae = np.mean(np.abs(actual - result["forecast_mw"].to_numpy()))
    snaive_mae = np.mean(np.abs(actual - snaive_daily))
    assert forecast_mae <= snaive_mae + 1.0


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="Trained models not available; run the training pipeline first.")
def test_backtest_predictions_shapes_match():
    artifacts = load_artifacts()
    result = backtest_predictions(artifacts, model_key="xgboost", split_name="test", max_points=200)
    assert "actual_mw" in result.columns
    assert "predicted_mw" in result.columns
    assert len(result) <= 200
    assert result.attrs.get("model_used") == "xgboost"


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="Trained models not available; run the training pipeline first.")
def test_backtest_predictions_invalid_split_raises():
    artifacts = load_artifacts()
    with pytest.raises(ValueError):
        backtest_predictions(artifacts, model_key="xgboost", split_name="not_a_split")
