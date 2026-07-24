"""FastAPI backend for the Great Britain National Electricity Demand
Forecasting System (NESO data).

Serves half hourly (30 minute settlement period) forecasts of National
Demand (ND, in MW) for the Great Britain transmission system. Whenever a
trained model artefact (.pkl / .joblib, produced by the research pipeline
in ``src/demandforecast``) is available, this API serves real forecasts
computed from it. If no trained artefact can be loaded -- for example on a
machine without the model files or without the optional ML libraries
installed -- the API automatically falls back to a deterministic,
seeded synthetic National Demand simulator so the service and the
Streamlit dashboard remain fully usable end to end.

Run with:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Path setup: make the research package (src/demandforecast) importable
# regardless of the working directory this API is launched from.
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SETTLEMENT_FREQ = "30min"          # NESO settlement periods are 30 minutes
STEPS_PER_HOUR = 2                 # 1 hour = 2 half-hourly settlement periods
MAX_OUTPUT_POINTS = 2000           # hard cap on points returned per request,
                                   # to keep API payloads and Streamlit tables
                                   # from growing unbounded on long ranges
TEST_SET_START = "2025-01-01"      # sealed test year used throughout the
TEST_SET_END = "2025-12-31"        # research pipeline (paper/main.tex)

# --------------------------------------------------------------------------
# Attempt to load the real trained pipeline. This never raises: if anything
# goes wrong (missing model files, missing optional libraries such as
# catboost/xgboost/lightgbm, corrupted artefacts, etc.) the API falls back
# to the synthetic simulator defined further below instead of failing to
# start.
# --------------------------------------------------------------------------
REAL_MODEL_AVAILABLE = False
_artifacts = None
_history_df: Optional[pd.DataFrame] = None
_dconfig = None
_forecast_horizon = None
_backtest_predictions = None

try:
    from demandforecast import config as _dconfig  # type: ignore
    from demandforecast.forecast import (  # type: ignore
        load_artifacts,
        forecast_horizon as _forecast_horizon,
        backtest_predictions as _backtest_predictions,
    )

    _artifacts = load_artifacts()
    if _artifacts.available_models():
        REAL_MODEL_AVAILABLE = True
        _history_df = _artifacts.history
    else:
        print("[startup] No trained model files found among available_models(); "
              "using synthetic simulator.")
except Exception as exc:  # pragma: no cover - defensive fallback, intentional broad catch
    print(f"[startup] Could not load real model artefacts ({type(exc).__name__}: {exc}); "
          "falling back to the synthetic demand simulator.")
    REAL_MODEL_AVAILABLE = False


def _resolve_default_model_key() -> Optional[str]:
    """Best-scoring model among those actually loadable, matching the same
    logic ``forecast.py`` itself uses when ``model_key`` is left unset."""
    if not REAL_MODEL_AVAILABLE or _artifacts is None:
        return None
    if _artifacts.metrics_summary:
        best_name = _artifacts.metrics_summary[0]["Model Name"]
        for key, display in _dconfig.MODEL_DISPLAY_NAMES.items():
            if display == best_name and key in _artifacts.models:
                return key
    for preferred in (
        "catboost", "lightgbm_tuned", "lightgbm", "extratrees",
        "histgradientboosting", "xgboost_tuned", "xgboost", "mlp",
        "random_forest", "linear_regression",
    ):
        if preferred in _artifacts.models:
            return preferred
    available = _artifacts.available_models()
    return available[0] if available else None


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(
    title="Great Britain National Electricity Demand Forecast API",
    description=(
        "Forecasts National Demand (ND, in MW) for the Great Britain "
        "transmission system at half hourly (30 minute) settlement period "
        "resolution, trained on National Energy System Operator (NESO) "
        "historic demand data. Automatically falls back to a deterministic "
        "synthetic demand simulator when no trained model artefact is "
        "available, so the API and dashboard remain fully functional."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Pydantic request / response schemas
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    real_model_available: bool
    message: str


class ModelInfo(BaseModel):
    key: str
    display_name: str
    is_default: bool


class ModelsResponse(BaseModel):
    real_model_available: bool
    default_model_key: Optional[str]
    models: List[ModelInfo]


def _validate_date_string(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Date must be in YYYY-MM-DD format") from exc
    return value


class PredictFutureRequest(BaseModel):
    start_date: str = Field(..., description="Forecast start date, format YYYY-MM-DD")
    horizon_hours: int = Field(
        ..., gt=0, le=336,
        description="Forecast horizon in hours (e.g. 12, 24, 48, 72, 168)",
    )
    model_key: Optional[str] = Field(
        None,
        description=(
            "Which trained model to use (see GET /models for valid keys). "
            "Omit or set to null to let the API pick the best-scoring model "
            "among those actually available."
        ),
    )

    @field_validator("start_date")
    @classmethod
    def _check_start_date(cls, value: str) -> str:
        return _validate_date_string(value)


class ForecastPoint(BaseModel):
    datetime: str
    predicted_mw: float


class PredictFutureResponse(BaseModel):
    start_date: str
    horizon_hours: int
    horizon_steps: int
    model_used: str
    source: str = Field(..., description='"model" or "synthetic"')
    points: List[ForecastPoint]


class PredictEvaluateRequest(BaseModel):
    start_date: str = Field(..., description="Evaluation window start date, format YYYY-MM-DD")
    end_date: str = Field(..., description="Evaluation window end date, format YYYY-MM-DD")
    model_key: Optional[str] = Field(
        None,
        description=(
            "Which trained model to use (see GET /models for valid keys). "
            "Omit or set to null to let the API pick the best-scoring model "
            "among those actually available."
        ),
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def _check_dates(cls, value: str) -> str:
        return _validate_date_string(value)


class ComparisonPoint(BaseModel):
    datetime: str
    actual_mw: float
    predicted_mw: float
    error_mw: float


class PredictEvaluateResponse(BaseModel):
    start_date: str
    end_date: str
    n_periods: int
    mape: float
    mae: float
    rmse: float
    model_used: str
    source: str = Field(..., description='"model" or "synthetic"')
    points: List[ComparisonPoint]


# --------------------------------------------------------------------------
# Synthetic National Demand simulator
# --------------------------------------------------------------------------
def synthetic_demand(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic, seeded synthetic National Demand simulator.

    Used only when no trained model artefact (.pkl / .joblib) is available.
    Mimics the real GB National Demand shape documented in the accompanying
    research pipeline: a daytime peak, an overnight trough, a weekday versus
    weekend contrast (real data: weekday demand roughly 11.5% above weekend),
    and a winter-peak / summer-trough annual seasonal pattern, plus small
    repeatable noise. This is a stand-in for demonstration purposes only and
    is not a substitute for the real trained pipeline described in
    ``paper/main.tex``.
    """
    hour_of_day = np.asarray(timestamps.hour, dtype=float) + np.asarray(timestamps.minute, dtype=float) / 60.0
    day_of_week = np.asarray(timestamps.dayofweek)  # 0 = Monday ... 6 = Sunday
    day_of_year = np.asarray(timestamps.dayofyear, dtype=float)

    base_load_mw = 24000.0

    # Diurnal shape: overnight trough near 04:00, evening peak near 18:00.
    diurnal = (
        3500.0 * np.sin((hour_of_day - 4.0) / 24.0 * 2 * np.pi - np.pi / 2)
        + 2200.0 * np.exp(-((hour_of_day - 18.0) ** 2) / (2 * 2.2 ** 2))
        - 1800.0 * np.exp(-((hour_of_day - 4.0) ** 2) / (2 * 1.8 ** 2))
    )

    # Weekday versus weekend contrast.
    weekend_dip = np.where(day_of_week >= 5, -1300.0, 0.0)

    # Annual seasonal pattern: winter peak, summer trough.
    seasonal = 3000.0 * np.cos((day_of_year - 15) / 365.25 * 2 * np.pi)

    # Small deterministic pseudo-noise, seeded for repeatability across calls.
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, 250.0, size=len(timestamps))

    values = base_load_mw + diurnal + weekend_dip + seasonal + noise
    return np.clip(values, 8000.0, 48000.0)


# --------------------------------------------------------------------------
# Real-model helpers (only used when REAL_MODEL_AVAILABLE is True)
# --------------------------------------------------------------------------
def real_future_forecast(
    start: pd.Timestamp, horizon_steps: int, model_key: Optional[str] = None
) -> Optional[pd.Series]:
    """Attempt a real forecast from the trained pipeline, using the given
    ``model_key`` (or the API's own best-model choice if None).

    Returns None (triggering the synthetic fallback in the endpoint) if the
    real pipeline is unavailable or the forecast call fails for any reason.
    """
    if not REAL_MODEL_AVAILABLE:
        return None
    try:
        result = _forecast_horizon(_artifacts, n_steps=horizon_steps, model_key=model_key)
        aligned_index = pd.date_range(start=start, periods=len(result), freq=SETTLEMENT_FREQ)
        return pd.Series(result["forecast_mw"].to_numpy(), index=aligned_index)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[predict/future] Real model forecast failed ({exc}); using synthetic fallback.")
        return None


def real_evaluate(
    start: pd.Timestamp, end: pd.Timestamp, model_key: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Attempt to assemble real actual-vs-predicted values for a historical
    window from the trained pipeline's own sealed test-set backtest, using
    the given ``model_key`` (or the API's own best-model choice if None).

    Returns None (triggering the synthetic fallback) if unavailable.
    """
    if not REAL_MODEL_AVAILABLE:
        return None
    try:
        bt = _backtest_predictions(_artifacts, model_key=model_key, split_name="test", max_points=None)
        window = bt.loc[(bt.index >= start) & (bt.index <= end), ["actual_mw", "predicted_mw"]]
        if window.empty:
            return None
        return window
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[predict/evaluate] Real backtest failed ({exc}); using synthetic fallback.")
        return None


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint reporting server status and which forecasting
    backend (real trained model or synthetic simulator) is currently active.
    """
    message = (
        "Great Britain National Demand forecasting API is running. "
        + (
            "Serving real trained model forecasts."
            if REAL_MODEL_AVAILABLE
            else "No trained model artefact found; serving the synthetic demand simulator."
        )
    )
    return HealthResponse(status="ok", real_model_available=REAL_MODEL_AVAILABLE, message=message)


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """List the trained models actually available on this server, so the
    dashboard can offer a model selection control. If no trained model is
    available, returns an empty list (the API then serves the synthetic
    simulator for every request, regardless of any model_key supplied).
    """
    default_key = _resolve_default_model_key()
    model_infos: List[ModelInfo] = []
    if REAL_MODEL_AVAILABLE and _artifacts is not None:
        for key in _artifacts.available_models():
            display_name = _dconfig.MODEL_DISPLAY_NAMES.get(key, key)
            model_infos.append(
                ModelInfo(key=key, display_name=display_name, is_default=(key == default_key))
            )
    return ModelsResponse(
        real_model_available=REAL_MODEL_AVAILABLE,
        default_model_key=default_key,
        models=model_infos,
    )


@app.post("/predict/future", response_model=PredictFutureResponse)
def predict_future(request: PredictFutureRequest) -> PredictFutureResponse:
    """Forecast National Demand forward from ``start_date`` for
    ``horizon_hours`` hours, at 30 minute settlement period resolution.

    ``horizon_hours`` is converted internally to ``horizon_steps =
    horizon_hours * 2`` half hourly settlement periods. An optional
    ``model_key`` selects which trained model to use (see GET /models).
    """
    try:
        start = pd.Timestamp(request.start_date)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid start_date: {exc}") from exc

    if request.model_key and REAL_MODEL_AVAILABLE and _artifacts is not None:
        if request.model_key not in _artifacts.available_models():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown model_key '{request.model_key}'. "
                    f"Valid options: {_artifacts.available_models()}"
                ),
            )

    horizon_steps = request.horizon_hours * STEPS_PER_HOUR
    if horizon_steps > MAX_OUTPUT_POINTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"horizon_hours={request.horizon_hours} implies {horizon_steps} "
                f"settlement periods, above the {MAX_OUTPUT_POINTS}-point response limit. "
                f"Please request a shorter horizon."
            ),
        )

    series = real_future_forecast(start, horizon_steps, model_key=request.model_key)
    source = "model"
    model_used = request.model_key or _resolve_default_model_key() or "synthetic-simulator"
    if series is None:
        timestamps = pd.date_range(start=start, periods=horizon_steps, freq=SETTLEMENT_FREQ)
        values = synthetic_demand(timestamps)
        series = pd.Series(values, index=timestamps)
        source = "synthetic"
        model_used = "synthetic-simulator"

    points = [
        ForecastPoint(datetime=idx.strftime("%Y-%m-%d %H:%M"), predicted_mw=round(float(val), 2))
        for idx, val in series.items()
    ]

    return PredictFutureResponse(
        start_date=request.start_date,
        horizon_hours=request.horizon_hours,
        horizon_steps=horizon_steps,
        model_used=_dconfig.MODEL_DISPLAY_NAMES.get(model_used, model_used) if _dconfig else model_used,
        source=source,
        points=points,
    )


@app.post("/predict/evaluate", response_model=PredictEvaluateResponse)
def predict_evaluate(request: PredictEvaluateRequest) -> PredictEvaluateResponse:
    """Compare forecast against actual National Demand over a historical
    window (intended for the sealed 2025 test period) and report MAPE, MAE,
    and RMSE, alongside the full half hourly comparison series. An optional
    ``model_key`` selects which trained model to evaluate (see GET /models).
    """
    try:
        start = pd.Timestamp(request.start_date)
        end = pd.Timestamp(request.end_date) + pd.Timedelta(hours=23, minutes=30)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date range: {exc}") from exc

    if end < start:
        raise HTTPException(status_code=422, detail="end_date must not be before start_date")

    if request.model_key and REAL_MODEL_AVAILABLE and _artifacts is not None:
        if request.model_key not in _artifacts.available_models():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown model_key '{request.model_key}'. "
                    f"Valid options: {_artifacts.available_models()}"
                ),
            )

    merged = real_evaluate(start, end, model_key=request.model_key)
    source = "model"
    model_used = request.model_key or _resolve_default_model_key() or "synthetic-simulator"
    if merged is None:
        timestamps = pd.date_range(start=start, end=end, freq=SETTLEMENT_FREQ)
        if len(timestamps) > MAX_OUTPUT_POINTS:
            timestamps = timestamps[:MAX_OUTPUT_POINTS]
        actual_values = synthetic_demand(timestamps)
        rng = np.random.default_rng(seed=7)
        predicted_values = actual_values + rng.normal(0.0, 500.0, size=len(timestamps))
        merged = pd.DataFrame(
            {"actual_mw": actual_values, "predicted_mw": predicted_values}, index=timestamps
        )
        source = "synthetic"
        model_used = "synthetic-simulator"
    elif len(merged) > MAX_OUTPUT_POINTS:
        merged = merged.iloc[:MAX_OUTPUT_POINTS]

    actual_arr = merged["actual_mw"].to_numpy()
    predicted_arr = merged["predicted_mw"].to_numpy()

    points = [
        ComparisonPoint(
            datetime=idx.strftime("%Y-%m-%d %H:%M"),
            actual_mw=round(float(a), 2),
            predicted_mw=round(float(p), 2),
            error_mw=round(float(a - p), 2),
        )
        for idx, a, p in zip(merged.index, actual_arr, predicted_arr)
    ]

    return PredictEvaluateResponse(
        start_date=request.start_date,
        end_date=request.end_date,
        n_periods=len(merged),
        mape=round(_mape(actual_arr, predicted_arr), 4),
        mae=round(_mae(actual_arr, predicted_arr), 2),
        rmse=round(_rmse(actual_arr, predicted_arr), 2),
        model_used=_dconfig.MODEL_DISPLAY_NAMES.get(model_used, model_used) if _dconfig else model_used,
        source=source,
        points=points,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
