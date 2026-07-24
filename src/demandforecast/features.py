"""Feature engineering pipeline.

Converted from ``notebooks/03_feature_engineering_and_selection.ipynb``.
Turns the cleaned time series (``electricity_cleaned.csv``) into a dense,
purely-numeric feature matrix ready for model training:

1. Autoregressive lag features at ``[1, 2, 12, 24, 36, 48]`` (short-term) and
   ``336`` (weekly) settlement periods, on all three target columns.
2. Leakage-safe rolling mean/std statistics (window sizes 8/24/48 periods),
   always computed on a series shifted by one period first.
3. Cyclical sin/cos encoding of the settlement period (daily cycle, T=48)
   and day-of-week (weekly cycle, T=7), plus an ``IS_WEEKEND`` flag.
4. Trimming the first 336 rows (insufficient weekly-lag history) and
   dropping non-numeric/administrative columns.
5. Mutual-information feature selection against the primary target (``ND``),
   computed only on the pre-test (2020-2024) portion of the data to avoid
   feature-selection leakage into the held-out 2025 test set.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from . import config

logger = logging.getLogger(__name__)


def load_clean_series(clean_csv: Path) -> pd.DataFrame:
    """Load the cleaned CSV and index it by a continuous 30-minute DATETIME."""
    df = pd.read_csv(clean_csv)
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"])
    df = df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]).reset_index(drop=True)

    df["DATETIME"] = df["SETTLEMENT_DATE"] + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")
    df = df.set_index("DATETIME").sort_index()
    df = df.asfreq("30min")
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_columns: Sequence[str] = config.AR_LAG_TARGETS,
    short_term_lags: Sequence[int] = config.SHORT_TERM_LAGS,
    weekly_lag: int = config.WEEKLY_LAG,
) -> pd.DataFrame:
    """Add autoregressive lag columns ``{TARGET}_LAG_{H}``.

    Only ``config.AR_LAG_TARGETS`` (by default just the primary target, ND)
    are lagged. TSD and ENGLAND_WALES_DEMAND used to be lagged as well, but
    they move in near lock-step with ND (offset by a roughly constant
    ~2000-2400 MW), so their lags added little signal on top of ND's own
    lags, while making the recursive multi-step forecaster fragile: it had
    no way to predict TSD/EWD, so it just copied the ND prediction into
    their lag slots at every step, silently corrupting EWD_LAG_1 (the
    original write-up's #2 most important feature). See config.py.
    """
    df = df.copy()
    for col in target_columns:
        for h in short_term_lags:
            df[f"{col}_LAG_{h}"] = df[col].shift(h)
        df[f"{col}_LAG_{weekly_lag}"] = df[col].shift(weekly_lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = config.ROLLING_WINDOWS,
    solar_column: str = config.SOLAR_COLUMN,
    primary_target: str = config.PRIMARY_TARGET,
) -> pd.DataFrame:
    """Add leakage-safe rolling mean/std features.

    BUG FIXED (dot 5, xem SUA_DOI_DA_LAM.md muc "Dong bo so lieu"): ham nay
    tung shift(1) truoc khi rolling, tuc dung du lieu tinh den t-30phut cho
    mot bai toan du bao truoc 6 gio (t-12). Do la cung mot loai loi lech
    chan troi da sua o add_lag_features(), chi khac la no nam trong ham
    rolling nen khong bi phat hien cung dot voi 12 cot ngoai sinh do tai t.
    Gio shift theo config.MIN_LEAD_PERIODS (mac dinh 12 = 6 gio) de nhat
    quan voi SHORT_TERM_LAGS va voi ma tran notebook 04 dang dung
    (electricity_features_fixed.csv).
    """
    df = df.copy()
    lead = config.MIN_LEAD_PERIODS
    for w in windows:
        df[f"{primary_target}_ROLL_MEAN_{w}"] = df[primary_target].shift(lead).rolling(window=w).mean()
        df[f"{primary_target}_ROLL_STD_{w}"] = df[primary_target].shift(lead).rolling(window=w).std()

    if solar_column in df.columns:
        for w in windows:
            df[f"SOLAR_ROLL_MEAN_{w}"] = df[solar_column].shift(lead).rolling(window=w).mean()

    return df


def add_cyclical_calendar_features(
    df: pd.DataFrame,
    periods_per_day: int = config.PERIODS_PER_DAY,
    days_per_week: int = config.DAYS_PER_WEEK,
) -> pd.DataFrame:
    """Add sin/cos encodings of settlement-period-of-day and day-of-week."""
    df = df.copy()
    df["PERIOD_SIN"] = np.sin(2 * np.pi * df["SETTLEMENT_PERIOD"] / periods_per_day)
    df["PERIOD_COS"] = np.cos(2 * np.pi * df["SETTLEMENT_PERIOD"] / periods_per_day)

    df["DAYOFWEEK"] = df.index.dayofweek
    df["DOW_SIN"] = np.sin(2 * np.pi * df["DAYOFWEEK"] / days_per_week)
    df["DOW_COS"] = np.cos(2 * np.pi * df["DAYOFWEEK"] / days_per_week)

    df["IS_WEEKEND"] = (df["DAYOFWEEK"] >= 5).astype(int)
    return df


def trim_and_drop_administrative_columns(
    df: pd.DataFrame,
    weekly_lag: int = config.WEEKLY_LAG,
    administrative_columns: Sequence[str] = config.ADMINISTRATIVE_COLUMNS_TO_DROP,
) -> pd.DataFrame:
    """Drop the first ``weekly_lag`` rows and non-numeric/admin columns."""
    df_trimmed = df.iloc[weekly_lag:].copy()
    existing_admin_cols = [c for c in administrative_columns if c in df_trimmed.columns]
    df_trimmed = df_trimmed.drop(columns=existing_admin_cols)
    return df_trimmed


def drop_leaky_exogenous_columns(
    df: pd.DataFrame,
    leaky_columns: Sequence[str] = config.EXOGENOUS_LEAKY_COLUMNS,
) -> pd.DataFrame:
    """Drop exogenous columns recorded at the *same* settlement period as
    the target (wind/solar generation, pumped-storage pumping, the nine
    interconnector flows). These are not observable at the 6-hour-ahead
    decision point the model claims to operate at, so leaving them in
    inflates held-out accuracy without representing anything the model
    could actually use in production. Installed-capacity columns are left
    alone - they are planning figures, not point-in-time readings.
    """
    existing = [c for c in leaky_columns if c in df.columns]
    if existing:
        logger.info("Dropping %s contemporaneous (time-t) exogenous columns: %s", len(existing), existing)
    return df.drop(columns=existing)


def select_features_by_mutual_information(
    df_trimmed: pd.DataFrame,
    primary_target: str = config.PRIMARY_TARGET,
    target_columns: Sequence[str] = config.TARGET_COLUMNS,
    sample_size: int = config.MUTUAL_INFO_SAMPLE_SIZE,
    threshold: float = config.MUTUAL_INFO_THRESHOLD,
    random_state: int = config.MUTUAL_INFO_RANDOM_STATE,
    selection_cutoff: str = config.FEATURE_SELECTION_CUTOFF,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Rank candidate features by mutual information against ``primary_target``.

    Only rows up to ``selection_cutoff`` (i.e. Train+Valid, 2020-2024) are used
    to compute the scores, so the held-out test year never leaks into the
    feature-selection decision.
    """
    candidate_cols = [c for c in df_trimmed.columns if c not in target_columns]

    mi_pool = df_trimmed.loc[:selection_cutoff]
    n_available = mi_pool.shape[0]
    n_sample = min(sample_size, n_available)

    sample = mi_pool.sample(n=n_sample, random_state=random_state)
    X_sample = sample[candidate_cols]
    y_sample = sample[primary_target]

    mi_scores = mutual_info_regression(X_sample, y_sample, random_state=random_state)
    mi_table = (
        pd.DataFrame({"feature": candidate_cols, "mi_score": mi_scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    kept = set(mi_table[mi_table["mi_score"] > threshold]["feature"].tolist())
    dropped = set(mi_table[mi_table["mi_score"] <= threshold]["feature"].tolist())

    # Cyclical sin/cos pairs must survive or fall together: keeping only
    # PERIOD_SIN or only DOW_COS makes the encoding lose information the MI
    # filter (which scores each column independently) cannot see, because
    # neither half alone can distinguish all points on the cycle. If either
    # half of a pair clears the threshold, keep both.
    for group in config.CYCLICAL_FEATURE_GROUPS:
        present = [c for c in group if c in candidate_cols]
        if present and any(c in kept for c in present):
            for c in present:
                kept.add(c)
                dropped.discard(c)

    # A handful of columns are structurally required downstream (e.g.
    # SETTLEMENT_PERIOD is needed to reconstruct PERIOD_SIN/COS at
    # inference time) regardless of their MI score against ND.
    for c in config.MUTUAL_INFO_PROTECTED_COLUMNS:
        if c in candidate_cols:
            kept.add(c)
            dropped.discard(c)

    kept_list = [f for f in mi_table["feature"] if f in kept]
    dropped_list = [f for f in mi_table["feature"] if f in dropped]

    logger.info("Mutual information: kept %s features, dropped %s", len(kept_list), len(dropped_list))
    return kept_list, dropped_list, mi_table


def gatekeeper_assertions(
    df_final: pd.DataFrame,
    target_columns: Sequence[str] = config.TARGET_COLUMNS,
) -> None:
    """Acceptance checks before writing the final feature matrix to disk."""
    assert list(df_final.columns[: len(target_columns)]) == list(target_columns), (
        "Target columns are not preserved at the head of the matrix!"
    )
    n_missing = int(df_final.isna().sum().sum())
    assert n_missing == 0, f"Final feature matrix still has {n_missing} missing values!"
    n_non_numeric = df_final.select_dtypes(exclude=[np.number]).shape[1]
    assert n_non_numeric == 0, "Final feature matrix still has non-numeric columns!"


def run_feature_pipeline(
    clean_csv: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """End-to-end feature pipeline: clean CSV -> validated feature matrix CSV."""
    clean_csv = Path(clean_csv or config.CLEANED_CSV)
    output_csv = Path(output_csv or config.FEATURES_CSV)

    df = load_clean_series(clean_csv)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_calendar_features(df)
    df = drop_leaky_exogenous_columns(df)

    df_trimmed = trim_and_drop_administrative_columns(df)

    kept, dropped, _mi_table = select_features_by_mutual_information(df_trimmed)

    final_columns = list(config.TARGET_COLUMNS) + kept
    df_final = df_trimmed[final_columns].copy()

    gatekeeper_assertions(df_final)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv, index=True, index_label="DATETIME")
    logger.info("Saved feature matrix to %s (%s rows, %s columns)", output_csv, *df_final.shape)
    return df_final


def build_feature_row_for_inference(
    history: pd.DataFrame,
    feature_columns: Iterable[str],
    primary_target: str = config.PRIMARY_TARGET,
) -> pd.Series:
    """Build a single feature row for the *next* timestamp given history.

    ``history`` must be a DATETIME-indexed, 30-minute-frequency DataFrame
    containing at least the target columns (``ND``, ``TSD``,
    ``ENGLAND_WALES_DEMAND``) and, ideally, ``EMBEDDED_SOLAR_GENERATION``,
    covering at least the last 336 settlement periods (7 days) so that all
    lag/rolling features can be computed. Used by the recursive multi-step
    forecaster in :mod:`demandforecast.forecast`.
    """
    next_time = history.index[-1] + pd.Timedelta(minutes=30)
    row = {}

    for col in config.AR_LAG_TARGETS:
        series = history[col]
        for h in config.SHORT_TERM_LAGS:
            row[f"{col}_LAG_{h}"] = series.iloc[-h] if len(series) >= h else np.nan
        row[f"{col}_LAG_{config.WEEKLY_LAG}"] = (
            series.iloc[-config.WEEKLY_LAG] if len(series) >= config.WEEKLY_LAG else np.nan
        )

    lead = config.MIN_LEAD_PERIODS
    for w in config.ROLLING_WINDOWS:
        trimmed = history[primary_target].iloc[: len(history) - (lead - 1)] if lead > 1 else history[primary_target]
        window_slice = trimmed.iloc[-w:]
        row[f"{primary_target}_ROLL_MEAN_{w}"] = window_slice.mean()
        row[f"{primary_target}_ROLL_STD_{w}"] = window_slice.std()

    if config.SOLAR_COLUMN in history.columns:
        for w in config.ROLLING_WINDOWS:
            trimmed = history[config.SOLAR_COLUMN].iloc[: len(history) - (lead - 1)] if lead > 1 else history[config.SOLAR_COLUMN]
            window_slice = trimmed.iloc[-w:]
            row[f"SOLAR_ROLL_MEAN_{w}"] = window_slice.mean()

    period = ((next_time.hour * 60 + next_time.minute) // 30) + 1
    row["SETTLEMENT_PERIOD"] = period
    row["PERIOD_SIN"] = np.sin(2 * np.pi * period / config.PERIODS_PER_DAY)
    row["PERIOD_COS"] = np.cos(2 * np.pi * period / config.PERIODS_PER_DAY)

    dow = next_time.dayofweek
    row["DOW_SIN"] = np.sin(2 * np.pi * dow / config.DAYS_PER_WEEK)
    row["DOW_COS"] = np.cos(2 * np.pi * dow / config.DAYS_PER_WEEK)
    row["IS_WEEKEND"] = int(dow >= 5)

    # Any remaining requested feature that is a raw (contemporaneous, e.g.
    # exogenous renewable-generation/flow) column we have no future value
    # for: approximate it with a seasonal-naive persistence (same
    # settlement period one day earlier), falling back to the last known
    # observation. This keeps the recursive multi-step forecaster usable
    # beyond the end of the available history.
    for col in feature_columns:
        if col in row:
            continue
        if col in history.columns:
            series = history[col]
            if len(series) >= config.STANDARD_PERIODS_PER_DAY:
                row[col] = series.iloc[-config.STANDARD_PERIODS_PER_DAY]
            else:
                row[col] = series.iloc[-1]

    full_row = pd.Series(row, name=next_time)
    return full_row.reindex(list(feature_columns))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_feature_pipeline()
