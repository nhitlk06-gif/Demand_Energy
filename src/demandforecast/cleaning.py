"""Raw data ingestion and cleaning pipeline.

Converted from ``notebooks/01_pre_eda_and_cleaning.ipynb``. Turns the six
raw National Grid ESO ``demanddata_YYYY.csv`` files (2020-2025) into a single
clean, gap-free, 48-settlement-period-per-day time series.

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_raw_year_files(raw_dir: Path, file_paths: Dict[int, str] | None = None) -> pd.DataFrame:
    file_paths = file_paths or config.RAW_FILE_PATHS
    frames = []
    for year, filename in file_paths.items():
        path = Path(raw_dir) / filename
        df_year = pd.read_csv(path)
        df_year["SOURCE_FILE_YEAR"] = year
        frames.append(df_year)
        logger.info("Year %s: read %s rows, %s columns", year, df_year.shape[0], df_year.shape[1])

    df_raw = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    logger.info("Merged raw shape: %s rows, %s columns", df_raw.shape[0], df_raw.shape[1])
    return df_raw


def parse_settlement_date(df: pd.DataFrame, formats: Dict[int, str] | None = None) -> pd.Series:
    """Parse ``SETTLEMENT_DATE`` using the correct per-year string format."""
    formats = formats or config.SETTLEMENT_DATE_FORMATS
    parsed_parts = []
    for year, group in df.groupby("SOURCE_FILE_YEAR"):
        fmt = formats[year]
        parsed_parts.append(pd.to_datetime(group["SETTLEMENT_DATE"], format=fmt))
    return pd.concat(parsed_parts).sort_index()


def _resynchronise_one_day(
    group: pd.DataFrame, numeric_cols: Iterable[str], standard_periods: int
) -> pd.DataFrame:

    n_periods = len(group)
    group_sorted = group.sort_values("SETTLEMENT_PERIOD").reset_index(drop=True)

    if n_periods == standard_periods:
        result = group_sorted[list(numeric_cols)].copy()
        result["SETTLEMENT_PERIOD"] = range(1, standard_periods + 1)
        result["SETTLEMENT_DATE_PARSED_TMP"] = group_sorted["SETTLEMENT_DATE_PARSED_TMP"].iloc[0]
        result["SOURCE_FILE_YEAR"] = group_sorted["SOURCE_FILE_YEAR"].iloc[0]
        return result

    source_positions = np.linspace(0, 1, n_periods)
    target_positions = np.linspace(0, 1, standard_periods)

    new_data = {}
    for col in numeric_cols:
        if col == "SETTLEMENT_PERIOD":
            continue
        new_data[col] = np.interp(target_positions, source_positions, group_sorted[col].values)

    result = pd.DataFrame(new_data)
    result["SETTLEMENT_PERIOD"] = range(1, standard_periods + 1)
    result["SETTLEMENT_DATE_PARSED_TMP"] = group_sorted["SETTLEMENT_DATE_PARSED_TMP"].iloc[0]
    result["SOURCE_FILE_YEAR"] = group_sorted["SOURCE_FILE_YEAR"].iloc[0]
    return result


def resynchronise_time_axis(
    df_raw: pd.DataFrame, standard_periods: int = config.STANDARD_PERIODS_PER_DAY
) -> pd.DataFrame:
    df_raw = df_raw.copy()
    df_raw["SETTLEMENT_DATE_PARSED_TMP"] = parse_settlement_date(df_raw)

    numeric_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != "SOURCE_FILE_YEAR"]

    pieces = []
    for _, group in df_raw.groupby("SETTLEMENT_DATE_PARSED_TMP"):
        pieces.append(_resynchronise_one_day(group, numeric_cols, standard_periods))

    df_clean = pd.concat(pieces, ignore_index=True)
    df_clean = df_clean.rename(columns={"SETTLEMENT_DATE_PARSED_TMP": "SETTLEMENT_DATE"})
    df_clean = df_clean.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]).reset_index(drop=True)

    n_off_cadence = int((df_clean.groupby("SETTLEMENT_DATE")["SETTLEMENT_PERIOD"].count() != standard_periods).sum())
    if n_off_cadence != 0:
        raise AssertionError("Some days still do not have the standard number of settlement periods!")

    return df_clean


def drop_structurally_missing_columns(
    df: pd.DataFrame, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    columns = list(columns or config.STRUCTURALLY_MISSING_COLUMNS)
    existing = [c for c in columns if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        logger.info("Dropped structurally-missing columns: %s", existing)
    return df


def interpolate_gaps_and_anomalies(
    df: pd.DataFrame, demand_columns: Iterable[str] = config.DEMAND_COLUMNS
) -> pd.DataFrame:
    df = df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]).reset_index(drop=True)

    for col in demand_columns:
        n_anomalous = int((df[col] <= 0).sum())
        if n_anomalous > 0:
            logger.info("Found %s physically invalid (<=0) values in %s; marking as NaN", n_anomalous, col)
            df.loc[df[col] <= 0, col] = np.nan

    cols_to_interpolate = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in ("SETTLEMENT_PERIOD", "SOURCE_FILE_YEAR")
    ]
    for col in cols_to_interpolate:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    return df


def safety_assertions(
    df: pd.DataFrame,
    demand_columns: Iterable[str] = config.DEMAND_COLUMNS,
    standard_periods: int = config.STANDARD_PERIODS_PER_DAY,
) -> None:
    n_missing = int(df.isna().sum().sum())
    assert n_missing == 0, f"Cleaned data still contains {n_missing} missing values!"

    n_invalid = 0
    for col in demand_columns:
        n_invalid += int((df[col] <= 0).sum())
    assert n_invalid == 0, f"Cleaned data still contains {n_invalid} non-positive demand readings!"

    expected_rows = df["SETTLEMENT_DATE"].nunique() * standard_periods
    assert df.shape[0] == expected_rows, "Row count does not match the expected 48-period grid!"


def run_cleaning_pipeline(
    raw_dir: Path | None = None,
    output_csv: Path | None = None,
    file_paths: Dict[int, str] | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir or config.DATA_RAW_DIR)
    output_csv = Path(output_csv or config.CLEANED_CSV)

    df_raw = load_raw_year_files(raw_dir, file_paths)
    df_clean = resynchronise_time_axis(df_raw)
    df_clean = drop_structurally_missing_columns(df_clean)
    df_clean = interpolate_gaps_and_anomalies(df_clean)

    safety_assertions(df_clean)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    logger.info("Saved cleaned data to %s (%s rows, %s columns)", output_csv, *df_clean.shape)
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cleaning_pipeline()
