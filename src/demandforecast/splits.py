"""Chronological train / valid / test splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from . import config


@dataclass
class DataSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    training: pd.DataFrame  # train + valid, used for cross-validation
    feature_columns: List[str]
    target_column: str


def load_feature_matrix(features_csv) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.set_index("DATETIME").sort_index()
    if df.isna().sum().sum() > 0:
        df = df.dropna()
    return df


def chronological_split(
    df: pd.DataFrame,
    target_column: str = config.PRIMARY_TARGET,
    target_columns: List[str] | None = None,
    train_end: str = config.TRAIN_END,
    valid_start: str = config.VALID_START,
    valid_end: str = config.VALID_END,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> DataSplit:
    """Split a DATETIME-indexed feature matrix into train/valid/test folds."""
    target_columns = target_columns or config.TARGET_COLUMNS
    feature_columns = [c for c in df.columns if c not in target_columns]

    train = df.loc[:train_end]
    valid = df.loc[valid_start:valid_end]
    test = df.loc[test_start:test_end]
    training = df.loc[:valid_end]

    return DataSplit(
        train=train,
        valid=valid,
        test=test,
        training=training,
        feature_columns=feature_columns,
        target_column=target_column,
    )


def xy(df: pd.DataFrame, split: DataSplit) -> Tuple[pd.DataFrame, pd.Series]:
    return df[split.feature_columns], df[split.target_column]
