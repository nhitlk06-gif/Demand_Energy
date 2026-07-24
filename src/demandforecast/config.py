"""Central configuration and constants for the demandforecast pipeline.

All the "magic numbers" that were scattered across the four source
notebooks (01_pre_eda_and_cleaning, 02_eda_revised,
03_feature_engineering_and_selection_revised,
04_model_training_and_evaluation_revised) live here, in one place, so that
the cleaning, feature-engineering, training and inference modules all agree
on the same conventions.

This package is the **short-horizon (6-hour-ahead) variant** of the Great Britain
national electricity demand forecaster: the two nearest autoregressive lags
(30 and 60 minutes back, ``LAG_1`` / ``LAG_2``) are deliberately *excluded*
from the feature set so that the minimum lag available to the models is
``LAG_12`` (6 hours). This removes the "read the answer off the previous
settlement period" shortcut and forces the models to rely on the daily/
weekly seasonal structure and exogenous columns (embedded wind/solar,
interconnector flows) instead, which is a fairer test of genuine short-term
forecasting skill. See ``docs/ARCHITECTURE.md`` for details.

NOTE -- horizon-consistency fix (2026-07): an earlier revision fixed the
autoregressive lag leak (LAG_1 / LAG_2) but left a second, same-shape leak
in place: 14 of the 43 exogenous columns are recorded at the *same*
settlement period as the target ("at time t"), not at or before t-12. That
is 12 raw generation/interconnector readings (wind, solar, pumped-storage
pumping and the nine interconnector flow columns) that are simply not
observable 6 hours before the fact, plus two installed-capacity columns,
which *are* legitimate because they are planning figures known well in
advance and change on a monthly, not half-hourly, timescale.
``EXOGENOUS_LEAKY_COLUMNS`` below is exactly those 12 columns; they are
dropped before feature selection (see
:func:`demandforecast.features.drop_leaky_exogenous_columns`). Removing
them moves held-out MAPE from ~3.59% to ~5.02% -- worse numbers, but
numbers that respect the horizon the paper claims. A second, independent
bug in the old recursive multi-step forecaster (it copied the ND
prediction into the TSD and ENGLAND_WALES_DEMAND lag features too, even
though those series sit ~2000-2400 MW above/below ND) is fixed by no
longer building autoregressive lags for TSD/ENGLAND_WALES_DEMAND at all
-- see ``AR_LAG_TARGETS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT_DIR = SRC_DIR.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"

CLEANED_CSV = DATA_PROCESSED_DIR / "electricity_cleaned.csv"

FEATURES_CSV = DATA_PROCESSED_DIR / "electricity_features_fixed.csv"
FEATURES_CSV_LEAKY_REFERENCE = DATA_PROCESSED_DIR / "electricity_features.csv"


# Raw source files (National Grid ESO historic demand data, 2020-2025)

RAW_FILE_PATHS = {
    2020: "demanddata_2020.csv",
    2021: "demanddata_2021.csv",
    2022: "demanddata_2022.csv",
    2023: "demanddata_2023.csv",
    2024: "demanddata_2024.csv",
    2025: "demanddata_2025.csv",
}

# Settlement date string formats differ by source year.
SETTLEMENT_DATE_FORMATS = {
    2020: "%d-%b-%Y",
    2021: "%d-%b-%Y",
    2022: "%d-%b-%Y",
    2023: "%d-%b-%y",
    2024: "%d-%b-%Y",
    2025: "%Y-%m-%d",
}

# Column dropped because it only exists from 2023 onward (structural missingness).
STRUCTURALLY_MISSING_COLUMNS = ["SCOTTISH_TRANSFER"]

# Number of standard 30-minute settlement periods per day.
STANDARD_PERIODS_PER_DAY = 48

# Demand columns used for the physical sanity check (<=0 treated as invalid).
DEMAND_COLUMNS = ["ND", "TSD", "ENGLAND_WALES_DEMAND"]


# Feature engineering

TARGET_COLUMNS = ["ND", "TSD", "ENGLAND_WALES_DEMAND"]
PRIMARY_TARGET = "ND"

# Minimum lag is 12 periods (6 hours) — LAG_1 / LAG_2 are intentionally
# excluded, see the module docstring.
SHORT_TERM_LAGS = [12, 24, 36, 48]


MIN_LEAD_PERIODS = 12
WEEKLY_LAG = 336

# Which of the three demand series get autoregressive LAG_*/ROLL_* features.
# Only the primary target (ND) is lagged. TSD and ENGLAND_WALES_DEMAND used
# to be lagged too, but they sit a near-constant ~2000-2400 MW away from ND
# and are almost collinear with it; the old recursive forecaster (see
# forecast.py) fed its ND prediction back in as if it were also the next
# TSD/EWD observation, corrupting EWD_LAG_1 (the #2 most important feature
# in the original write-up) at every step of a multi-step forecast. Not
# lagging them removes the bug outright rather than papering over it.
AR_LAG_TARGETS = [PRIMARY_TARGET]

ROLLING_WINDOWS = [8, 24, 48]
SOLAR_COLUMN = "EMBEDDED_SOLAR_GENERATION"

PERIODS_PER_DAY = 48
DAYS_PER_WEEK = 7

MUTUAL_INFO_SAMPLE_SIZE = 15000
MUTUAL_INFO_THRESHOLD = 0.01
MUTUAL_INFO_RANDOM_STATE = 42

# Test set is sealed off during feature selection to avoid leakage.
FEATURE_SELECTION_CUTOFF = "2024-12-31 23:30:00"

ADMINISTRATIVE_COLUMNS_TO_DROP = ["SETTLEMENT_DATE", "DAYOFWEEK", "SOURCE_FILE_YEAR"]


# Horizon-consistency fix: exogenous columns recorded at time t

# These 12 raw columns are physical readings taken at the *same* settlement
# period as the target, not forecasts or planning figures, so they are not
# actually available 6 hours ahead of time (the horizon the model claims).
# They must be dropped before the feature matrix is finalised. The two
# installed-*capacity* columns are kept: unlike generation/flow readings,
# nameplate capacity is a planning number that is known months in advance
# and essentially constant at 30-minute resolution, so using its
# same-period value is not a leak.
EXOGENOUS_LEAKY_COLUMNS = [
    "EMBEDDED_WIND_GENERATION",
    "EMBEDDED_SOLAR_GENERATION",
    "PUMP_STORAGE_PUMPING",
    "IFA_FLOW",
    "IFA2_FLOW",
    "BRITNED_FLOW",
    "MOYLE_FLOW",
    "EAST_WEST_FLOW",
    "NEMO_FLOW",
    "NSL_FLOW",
    "ELECLINK_FLOW",
    "VIKING_FLOW",
]

# Renewable-generation columns specifically (subset of the leaky list above),
# used by the "Hypothesis 4 re-read" diagnostic (see scripts/run_full_diagnostics.py):
# the paper's own ablation of these two columns is, in effect, a leakage-size
# measurement rather than a finding about renewables' forecasting value.
RENEWABLE_GENERATION_COLUMNS = ["EMBEDDED_WIND_GENERATION", "EMBEDDED_SOLAR_GENERATION"]

# Cyclical sin/cos pairs must be selected (or dropped) together by the
# mutual-information filter - dropping only one half of a pair (e.g. the
# original run dropped DOW_COS but kept DOW_SIN) makes the day-of-week
# encoding non-injective (sin alone cannot distinguish all 7 days).
CYCLICAL_FEATURE_GROUPS = [["PERIOD_SIN", "PERIOD_COS"], ["DOW_SIN", "DOW_COS"]]

# Columns that mutual-information filtering must never drop regardless of
# score, because a downstream module depends on them structurally (not just
# statistically) -SETTLEMENT_PERIOD is needed to reconstruct PERIOD_SIN/COS
# for inference.
MUTUAL_INFO_PROTECTED_COLUMNS = ["SETTLEMENT_PERIOD"]

# Chronological train / valid / test split (matches notebook 04)

TRAIN_END = "2023-12-31 23:30:00"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31 23:30:00"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31 23:30:00"

RANDOM_STATE = 42


# Model hyperparameters (defaults mirrored from notebook 04)

RF_PARAMS = dict(n_estimators=50, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50,
)

LGBM_PARAMS = dict(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
)

# Hyperparameters found by the random-search tuning step in notebook 04
# (``## 3. HYPERPARAMETER FINE TUNING``), used by ``tune_boosting_models``.
XGB_TUNED_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=50,
)

LGBM_TUNED_PARAMS = dict(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_samples=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

MODEL_FILENAMES = {
    "linear_regression": "linear_regression.pkl",
    "random_forest": "random_forest_regressor.pkl",
    "xgboost": "xgboost_regressor.pkl",
    "lightgbm": "lightgbm_regressor.pkl",
    "xgboost_tuned": "xgboost_regressor_tuned.pkl",
    "lightgbm_tuned": "lightgbm_regressor_tuned.pkl",
    "catboost": "catboost_regressor.pkl",
    "histgradientboosting": "histgradientboosting_regressor.pkl",
    "extratrees": "extratrees_regressor.pkl",
    "mlp": "mlp_regressor.pkl",
}

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
    "xgboost_tuned": "XGBoost Regressor (Tuned)",
    "lightgbm_tuned": "LightGBM Regressor (Tuned)",
    "catboost": "CatBoost Regressor",
    "histgradientboosting": "HistGradientBoosting Regressor",
    "extratrees": "ExtraTrees Regressor",
    "mlp": "MLP Regressor",
}

# --- Five additional models added per the reviewer's diagnostic pass -------

CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_state=RANDOM_STATE,
    early_stopping_rounds=50,
    verbose=False,
)

HISTGB_PARAMS = dict(random_state=RANDOM_STATE)

EXTRATREES_PARAMS = dict(n_estimators=50, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)

MLP_PARAMS = dict(random_state=RANDOM_STATE, max_iter=500, early_stopping=True)

# Quantiles used by the LightGBM quantile-regression model (see models.py /
# Table "5 model" item 2 in the review): 3 quantiles -> a 90% central
# prediction interval [q05, q95] plus the median.
QUANTILE_LEVELS = [0.05, 0.5, 0.95]
LGBM_QUANTILE_PARAMS = dict(
    objective="quantile",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)


@dataclass
class PipelineConfig:
    """Bundle of paths/knobs that scripts and the API can override."""

    data_raw_dir: Path = DATA_RAW_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR
    models_dir: Path = MODELS_DIR
    cleaned_csv: Path = CLEANED_CSV
    features_csv: Path = FEATURES_CSV
    fast_mode: bool = False
    """When True, use lighter hyperparameters so the whole pipeline can be
    exercised quickly (e.g. in CI or a quick local demo)."""

    def ensure_dirs(self) -> None:
        for d in (self.data_processed_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)
