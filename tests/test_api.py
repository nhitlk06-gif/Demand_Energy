import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


_MODELS_AVAILABLE = (ROOT_DIR / "models" / "xgboost_regressor.pkl").exists()


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="Trained models not available; run the training pipeline first.")
def test_forecast_endpoint():
    resp = client.get("/forecast", params={"steps": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_steps"] == 4
    assert len(body["forecast"]) == 4


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="Trained models not available; run the training pipeline first.")
def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    assert len(resp.json()) > 0


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="Trained models not available; run the training pipeline first.")
def test_backtest_endpoint_returns_actual_and_predicted():
    resp = client.get("/backtest", params={"split": "test", "max_points": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert body["split"] == "test"
    assert len(body["points"]) > 0
    first = body["points"][0]
    assert "actual_mw" in first and "predicted_mw" in first
