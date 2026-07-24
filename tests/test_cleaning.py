import numpy as np
import pandas as pd
import pytest

from demandforecast import cleaning


def _fake_day(date_str, n_periods, source_year, base_value=100.0):
    return pd.DataFrame(
        {
            "SETTLEMENT_DATE": [date_str] * n_periods,
            "SETTLEMENT_PERIOD": list(range(1, n_periods + 1)),
            "ND": np.linspace(base_value, base_value + n_periods, n_periods),
            "TSD": np.linspace(base_value, base_value + n_periods, n_periods),
            "ENGLAND_WALES_DEMAND": np.linspace(base_value, base_value + n_periods, n_periods),
            "SOURCE_FILE_YEAR": source_year,
        }
    )


def test_resynchronise_standard_day_untouched():
    df_raw = _fake_day("01-Jan-2021", 48, 2021)
    df_raw["SETTLEMENT_DATE_PARSED_TMP"] = pd.to_datetime(df_raw["SETTLEMENT_DATE"], format="%d-%b-%Y")
    out = cleaning._resynchronise_one_day(
        df_raw, [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != "SOURCE_FILE_YEAR"], 48
    )
    assert len(out) == 48
    assert list(out["SETTLEMENT_PERIOD"]) == list(range(1, 49))


def test_resynchronise_short_day_upsampled():
    df_raw = _fake_day("28-Mar-2021", 46, 2021)
    df_raw["SETTLEMENT_DATE_PARSED_TMP"] = pd.to_datetime(df_raw["SETTLEMENT_DATE"], format="%d-%b-%Y")
    out = cleaning._resynchronise_one_day(
        df_raw, [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != "SOURCE_FILE_YEAR"], 48
    )
    assert len(out) == 48


def test_resynchronise_long_day_downsampled():
    df_raw = _fake_day("31-Oct-2021", 50, 2021)
    df_raw["SETTLEMENT_DATE_PARSED_TMP"] = pd.to_datetime(df_raw["SETTLEMENT_DATE"], format="%d-%b-%Y")
    out = cleaning._resynchronise_one_day(
        df_raw, [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != "SOURCE_FILE_YEAR"], 48
    )
    assert len(out) == 48


def test_interpolate_gaps_and_anomalies_removes_non_positive():
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 4),
            "SETTLEMENT_PERIOD": [1, 2, 3, 4],
            "ND": [100.0, -5.0, 0.0, 110.0],
            "TSD": [100.0, 105.0, 108.0, 110.0],
            "ENGLAND_WALES_DEMAND": [100.0, 105.0, 108.0, 110.0],
            "SOURCE_FILE_YEAR": 2021,
        }
    )
    out = cleaning.interpolate_gaps_and_anomalies(df)
    assert (out["ND"] > 0).all()
    assert out["ND"].isna().sum() == 0


def test_safety_assertions_raises_on_missing_values():
    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 2),
            "SETTLEMENT_PERIOD": [1, 2],
            "ND": [100.0, np.nan],
            "TSD": [100.0, 100.0],
            "ENGLAND_WALES_DEMAND": [100.0, 100.0],
        }
    )
    with pytest.raises(AssertionError):
        cleaning.safety_assertions(df, standard_periods=2)
