"""Model definitions and thin wrappers around the eight forecasting models
compared in notebook 04: SNaive Daily, SNaive Weekly, Linear Regression,
Random Forest, XGBoost, LightGBM, and the random-search-tuned XGBoost /
LightGBM variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from . import config


class Predictor(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...


@dataclass
class SNaiveModel:
    """Seasonal-naive baseline: prediction is a single lag column, read
    directly out of the feature matrix. No parameters to fit.
    """

    lag_column: str
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SNaiveModel":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.lag_column]


def snaive_daily() -> SNaiveModel:
    return SNaiveModel(lag_column=f"{config.PRIMARY_TARGET}_LAG_48", name="SNaive Daily")


def snaive_weekly() -> SNaiveModel:
    return SNaiveModel(lag_column=f"{config.PRIMARY_TARGET}_LAG_336", name="SNaive Weekly")


def linear_regression_model() -> LinearRegression:
    return LinearRegression()


def random_forest_model(fast_mode: bool = False, **overrides) -> RandomForestRegressor:
    params = dict(config.RF_PARAMS)
    if fast_mode:
        params.update(n_estimators=15, max_depth=8)
    params.update(overrides)
    return RandomForestRegressor(**params)


def xgboost_model(fast_mode: bool = False, tuned: bool = False, **overrides):
    import xgboost as xgb

    params = dict(config.XGB_TUNED_PARAMS if tuned else config.XGB_PARAMS)
    if fast_mode:
        params.update(n_estimators=150)
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def lightgbm_model(fast_mode: bool = False, tuned: bool = False, **overrides):
    import lightgbm as lgb

    params = dict(config.LGBM_TUNED_PARAMS if tuned else config.LGBM_PARAMS)
    if fast_mode:
        params.update(n_estimators=150)
    params.update(overrides)
    return lgb.LGBMRegressor(**params)




def catboost_model(fast_mode: bool = False, **overrides):
    from catboost import CatBoostRegressor

    params = dict(config.CATBOOST_PARAMS)
    if fast_mode:
        params.update(iterations=200)
    params.update(overrides)
    return CatBoostRegressor(**params)


def histgradientboosting_model(fast_mode: bool = False, **overrides):
    """sklearn built-in; no extra install. Used to check whether the
    review's headline findings (the horizon leak, the peak-hour MAPE
    artifact) are properties of the *data*, not of LightGBM specifically."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(config.HISTGB_PARAMS)
    if fast_mode:
        params.update(max_iter=100)
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def extratrees_model(fast_mode: bool = False, **overrides):
    """sklearn built-in. Bagging-family counterpart to Random Forest, used
    as a same-family contrast to check whether model rankings are stable
    across the leaky vs. leak-removed configurations."""
    from sklearn.ensemble import ExtraTreesRegressor

    params = dict(config.EXTRATREES_PARAMS)
    if fast_mode:
        params.update(n_estimators=50, max_depth=12)
    params.update(overrides)
    return ExtraTreesRegressor(**params)


def mlp_model(fast_mode: bool = False, **overrides):
    """sklearn built-in. A neural baseline, so the write-up can honestly
    say deep learning was tried rather than assumed unnecessary."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(config.MLP_PARAMS)
    if fast_mode:
        params.update(max_iter=100, hidden_layer_sizes=(32,))
    params.update(overrides)
    # MLPs need scaled inputs; trees/boosting do not, hence the scaler only
    # lives inside this one model's pipeline.
    return make_pipeline(StandardScaler(), MLPRegressor(**params))


def lightgbm_quantile_models(fast_mode: bool = False, quantiles=None, **overrides):
    """One LightGBM model per quantile in ``config.QUANTILE_LEVELS`` (default
    [0.05, 0.5, 0.95]), used to build a 90% prediction interval and report
    PICP -- see ``metrics.prediction_interval_coverage``. The Conclusion
    promises interval-dependent operations (unit commitment, reserve
    procurement) that a point forecast cannot actually support; this is the
    ~5 lines of code needed to check whether the resulting intervals are
    trustworthy.
    """
    import lightgbm as lgb

    quantiles = quantiles or config.QUANTILE_LEVELS
    fitted_constructors = {}
    for q in quantiles:
        params = dict(config.LGBM_QUANTILE_PARAMS)
        params["alpha"] = q
        if fast_mode:
            params.update(n_estimators=150)
        params.update(overrides)
        fitted_constructors[q] = lgb.LGBMRegressor(**params)
    return fitted_constructors
