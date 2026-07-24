import numpy as np
import pandas as pd

from demandforecast import features


def _synthetic_series(n=500):
    idx = pd.date_range("2024-01-01", periods=n, freq="30min")
    rng = np.random.default_rng(42)
    base = 25000 + 2000 * np.sin(np.linspace(0, 40 * np.pi, n))
    df = pd.DataFrame(
        {
            "ND": base + rng.normal(0, 50, n),
            "TSD": base * 1.02 + rng.normal(0, 50, n),
            "ENGLAND_WALES_DEMAND": base * 0.9 + rng.normal(0, 50, n),
            "EMBEDDED_SOLAR_GENERATION": np.clip(np.sin(np.linspace(0, 40 * np.pi, n)), 0, None) * 500,
            "SETTLEMENT_PERIOD": (np.arange(n) % 48) + 1,
        },
        index=idx,
    )
    return df


def test_add_lag_features_shapes_and_values():
    df = _synthetic_series()
    out = features.add_lag_features(df, short_term_lags=[12, 24], weekly_lag=48)
    assert "ND_LAG_12" in out.columns
    assert out["ND_LAG_12"].iloc[20] == df["ND"].iloc[8]
    # LAG_1 / LAG_2 are intentionally not part of this project's default
    # config (minimum lag = 12), but add_lag_features itself is generic and
    # will still compute whatever lags it's asked for.


def test_add_rolling_features_no_leakage():
    df = _synthetic_series()
    out = features.add_rolling_features(df, windows=[4])
    # Rolling mean at position i must only use data available at least
    # config.MIN_LEAD_PERIODS (6h) before i -- not just one period back.
    from demandforecast import config
    lead = config.MIN_LEAD_PERIODS
    manual = df["ND"].shift(lead).rolling(window=4).mean()
    pd.testing.assert_series_equal(out["ND_ROLL_MEAN_4"], manual, check_names=False)


def test_inference_row_matches_training_row():
    """Regression test for the training/inference feature-construction
    mismatch found in the NESO Horizon review: build_feature_row_for_inference
    must reproduce exactly the same rolling-window values that
    add_rolling_features() computes at training time (both must end the
    window at t-MIN_LEAD_PERIODS, not t-1)."""
    from demandforecast import config

    df = _synthetic_series(n=400)
    trained = features.add_rolling_features(df, windows=config.ROLLING_WINDOWS)

    # Simulate "predicting the next row" by handing the inference builder
    # only the history strictly before the last row, then compare its
    # rolling features against what training computed for that last row.
    history = df.iloc[:-1]
    feature_columns = [c for c in trained.columns if "ROLL_" in c]
    row = features.build_feature_row_for_inference(history, feature_columns)

    last_row_trained = trained.iloc[-1]
    for w in config.ROLLING_WINDOWS:
        mean_col = f"ND_ROLL_MEAN_{w}"
        std_col = f"ND_ROLL_STD_{w}"
        assert row[mean_col] == pytest_approx(last_row_trained[mean_col]), (
            f"{mean_col}: inference={row[mean_col]} vs training={last_row_trained[mean_col]}"
        )
        assert row[std_col] == pytest_approx(last_row_trained[std_col]), (
            f"{std_col}: inference={row[std_col]} vs training={last_row_trained[std_col]}"
        )


def pytest_approx(value, tol=1e-6):
    """Tiny local stand-in for pytest.approx so this file has no hard
    dependency on pytest being installed to be read/run manually."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tol * max(1.0, abs(value))
    return _Approx()


def test_cyclical_features_are_bounded():
    df = _synthetic_series()
    out = features.add_cyclical_calendar_features(df)
    assert out["PERIOD_SIN"].between(-1, 1).all()
    assert out["PERIOD_COS"].between(-1, 1).all()
    assert set(out["IS_WEEKEND"].unique()).issubset({0, 1})


def test_build_feature_row_for_inference_returns_requested_columns():
    df = _synthetic_series(n=400)
    lagged = features.add_lag_features(df, short_term_lags=[12, 24], weekly_lag=48)
    rolled = features.add_rolling_features(lagged, windows=[4, 8])
    calendar = features.add_cyclical_calendar_features(rolled)
    feature_cols = [c for c in calendar.columns if c not in ("ND", "TSD", "ENGLAND_WALES_DEMAND")]

    row = features.build_feature_row_for_inference(df, feature_cols)
    assert list(row.index) == feature_cols
    assert row.name == df.index[-1] + pd.Timedelta(minutes=30)
