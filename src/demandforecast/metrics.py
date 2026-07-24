"""Evaluation metrics used throughout the pipeline (from notebook 04)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_percentage_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed mean percentage error — diagnoses systematic over/under bias."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean((y - yhat) / y))


def mean_absolute_percentage_error(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def ml_error(model_name: str, y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    """R2, MAE, RMSE and MAPE for one set of predictions, as a one-row frame."""
    r2 = r2_score(y, yhat)
    mae = mean_absolute_error(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mape = mean_absolute_percentage_error(y, yhat)
    return pd.DataFrame(
        {"Model Name": model_name, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape},
        index=[0],
    )


def diebold_mariano_test(
    y: np.ndarray,
    yhat_1: np.ndarray,
    yhat_2: np.ndarray,
    power: int = 2,
    h: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy of two forecasts.

    Tests H0: the two models have the same expected loss (here, squared
    error by default -- ``power=2``; use ``power=1`` for absolute error).
    Returns the DM statistic and a two-sided p-value (via a normal
    approximation with a small-sample Harvey-Leybourne-Newbold correction),
    plus the mean loss differential so the sign tells you which model is
    better (positive => model 2 has lower average loss).

    ``h`` is the forecast horizon in steps; for the 1-step-ahead comparisons
    in Table 1 of the review this stays at 1 (no autocorrelation
    correction needed for a single-step loss series).
    """
    y = np.asarray(y, dtype=float)
    e1 = y - np.asarray(yhat_1, dtype=float)
    e2 = y - np.asarray(yhat_2, dtype=float)
    loss1 = np.abs(e1) ** power
    loss2 = np.abs(e2) ** power
    d = loss1 - loss2
    n = len(d)

    d_bar = float(np.mean(d))
    # Newey-West-style long-run variance with (h-1) lags of autocovariance.
    gamma0 = float(np.var(d, ddof=0))
    var_d = gamma0
    for lag in range(1, h):
        cov = float(np.cov(d[lag:], d[:-lag])[0, 1]) if n > lag else 0.0
        var_d += 2 * (1 - lag / h) * cov
    var_d = max(var_d, 1e-12) / n

    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey, Leybourne & Newbold (1997) small-sample correction.
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln

    from scipy import stats

    p_value = float(2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1)))

    return {
        "dm_statistic": float(dm_stat_corrected),
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "n": n,
    }


def prediction_interval_coverage(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """PICP fraction of true values falling inside [lower, upper].

    Compare against the nominal coverage of the quantiles used to build the
    interval (e.g. q05/q95 -> nominal 90%) to check for over-confidence.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y >= lower) & (y <= upper)
    return float(np.mean(inside))


def mean_absolute_scaled_error(
    y: np.ndarray,
    yhat: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 48,
) -> float:
    """MASE (Hyndman & Koehler, 2006): MAE scaled by the in-sample MAE of a
    seasonal-naive forecast. Unlike MAPE it does not blow up or shrink
    artificially with the denominator's magnitude, which is exactly the
    distortion the review flags in the peak-hour MAPE discussion.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mae = float(np.mean(np.abs(y - yhat)))
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = float(np.mean(naive_errors))
    if scale == 0:
        return float("nan")
    return mae / scale


def cross_validation(
    X_training: pd.DataFrame,
    kfold: int,
    model_name: str,
    model,
    feature_cols,
    target_col: str = "ND",
    validation_days: int = 120,
    verbose: bool = False,
) -> pd.DataFrame:
    """Expanding-window time series cross validation.

    Splits ``X_training`` into ``kfold`` consecutive train/validation folds,
    each validation window ``validation_days`` long, walking backwards from
    the most recent data. Mirrors the walk-forward evaluation used in
    notebook 04.
    """
    r2_list, mae_list, rmse_list, mape_list = [], [], [], []

    for k in reversed(range(1, kfold + 1)):
        validation_start = X_training.index.max() - pd.Timedelta(days=k * validation_days)
        validation_end = X_training.index.max() - pd.Timedelta(days=(k - 1) * validation_days)

        training_fold = X_training[X_training.index < validation_start]
        validation_fold = X_training[
            (X_training.index >= validation_start) & (X_training.index <= validation_end)
        ]

        X_train_fold, y_train_fold = training_fold[feature_cols], training_fold[target_col]
        X_valid_fold, y_valid_fold = validation_fold[feature_cols], validation_fold[target_col]

        if verbose:
            print(
                f"[{model_name}] fold k={k}: train={len(X_train_fold):,} rows, "
                f"valid={len(X_valid_fold):,} rows"
            )

        fitted = model.fit(X_train_fold, y_train_fold)
        yhat_fold = fitted.predict(X_valid_fold)
        fold_result = ml_error(model_name, y_valid_fold, yhat_fold)

        r2_list.append(fold_result["R2"][0])
        mae_list.append(fold_result["MAE"][0])
        rmse_list.append(fold_result["RMSE"][0])
        mape_list.append(fold_result["MAPE (%)"][0])

    return pd.DataFrame(
        {
            "Model Name": model_name,
            "R2 CV": f"{np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}",
            "MAE CV": f"{np.mean(mae_list):.2f} +/- {np.std(mae_list):.2f}",
            "RMSE CV": f"{np.mean(rmse_list):.2f} +/- {np.std(rmse_list):.2f}",
            "MAPE CV (%)": f"{np.mean(mape_list):.2f} +/- {np.std(mape_list):.2f}",
        },
        index=[0],
    )
