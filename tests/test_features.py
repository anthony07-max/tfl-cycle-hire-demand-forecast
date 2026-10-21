"""Tests for the feature pipeline. The fixture is the first 400 real days (2019-01-01 onwards) of data/processed/daily.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from features import FEATURES, HORIZON, LAGS, MIN_VALID_HIRES, add_calendar, add_lags, build_features

FIX = Path(__file__).parent / "fixtures" / "daily_sample_2019.csv"


@pytest.fixture
def daily():
    return pd.read_csv(FIX, parse_dates=["date"])


def test_fixture_is_consecutive(daily):
    assert daily.date.diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_all_features_created(daily):
    out = build_features(daily)
    assert set(FEATURES) <= set(out.columns)
    assert len(out) == len(daily)


def test_lags_are_exact_shifts(daily):
    out = add_lags(daily)
    for lag in LAGS:
        i = lag + 5
        assert out.loc[i, f"lag_{lag}"] == daily.loc[i - lag, "hires"]


def test_no_feature_uses_data_newer_than_horizon(daily):
    """Changing the target on day t must not change any feature on days t .. t+HORIZON-1."""
    base = build_features(daily)
    t = 380
    bumped = daily.copy()
    bumped.loc[t, "hires"] += 50_000
    new = build_features(bumped)
    lag_cols = [c for c in FEATURES if c.startswith(("lag_", "roll"))]
    pd.testing.assert_frame_equal(base.loc[t:t + HORIZON - 1, lag_cols], new.loc[t:t + HORIZON - 1, lag_cols])
    assert not base.loc[t + HORIZON, lag_cols].equals(new.loc[t + HORIZON, lag_cols])


def test_leaky_lag_rejected(daily):
    with pytest.raises(ValueError):
        add_lags(daily, lags=[1])


def test_gap_in_dates_rejected(daily):
    with pytest.raises(ValueError):
        add_lags(daily.drop(index=10))


def test_outage_days_do_not_feed_lags(daily):
    d = daily.copy()
    d.loc[100, "hires"] = 0
    out = add_lags(d)
    assert out.loc[100, "anomaly"] == 1
    assert np.isnan(out.loc[107, "lag_7"])
    assert MIN_VALID_HIRES > 0


def test_calendar_flags(daily):
    out = add_calendar(daily).set_index("date")
    assert out.loc["2019-01-05", "is_weekend"] == 1          # Saturday
    assert out.loc["2019-12-25", "bank_holiday"] == 1         # Christmas Day (GOV.UK)
    assert out.loc["2019-12-27", "xmas_period"] == 1
    assert out.loc["2019-08-26", "non_working_day"] == 1      # Summer bank holiday, Monday
    assert out.loc["2019-08-27", "non_working_day"] == 0
    assert out.doy_sin.between(-1, 1).all()
