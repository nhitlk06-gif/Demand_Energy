import numpy as np
import pytest

from demandforecast.metrics import ml_error, mean_absolute_percentage_error, mean_percentage_error


def test_ml_error_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    result = ml_error("perfect", y, y)
    assert result["R2"][0] == 1.0
    assert result["MAE"][0] == 0.0
    assert result["RMSE"][0] == 0.0
    assert result["MAPE (%)"][0] == 0.0


def test_mean_percentage_error_sign():
    y = np.array([100.0, 100.0])
    yhat_over = np.array([110.0, 110.0])  # model overestimates -> negative MPE
    yhat_under = np.array([90.0, 90.0])  # model underestimates -> positive MPE
    assert mean_percentage_error(y, yhat_over) < 0
    assert mean_percentage_error(y, yhat_under) > 0


def test_mean_absolute_percentage_error():
    y = np.array([100.0, 200.0])
    yhat = np.array([110.0, 190.0])
    mape = mean_absolute_percentage_error(y, yhat)
    # |10/100| = 10%, |10/200| = 5% -> average 7.5%
    assert mape == pytest.approx(7.5)
