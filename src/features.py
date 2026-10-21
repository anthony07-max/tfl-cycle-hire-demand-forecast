"""Cleaning and feature engineering for daily cycle-hire forecasting.

All lag features are shifted by at least HORIZON days, so every feature for day t is
known by day t - HORIZON. The model is therefore a genuine 7-day-ahead forecast
(weather features assume a weather forecast is available for the target day).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZON = 7
LAGS = [7, 14, 21, 28, 364]
MIN_VALID_HIRES = 1000  # days below this are scheme outages/closures, not demand
WEATHER_COLS = ["temp_max", "temp_mean", "precip_mm", "precip_hours", "wind_max", "sunshine_h", "daylight_h"]


def load_hires(path: Path) -> pd.DataFrame:
    """Read the London Datastore workbook (sheet 'Data', columns 'Day' and 'Number of Bicycle Hires')."""
    raw = pd.read_excel(path, sheet_name="Data", header=None)
    hdr = raw.index[raw.iloc[:, 1].astype(str).str.strip().eq("Day")][0]
    df = raw.iloc[hdr + 1:, [1, 2]].dropna()
    df.columns = ["date", "hires"]
    df["date"] = pd.to_datetime(df["date"])
    df["hires"] = df["hires"].astype(int)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_weather(path: Path) -> pd.DataFrame:
    d = json.loads(Path(path).read_text())["daily"]
    w = pd.DataFrame(d).rename(columns={
        "time": "date", "temperature_2m_max": "temp_max", "temperature_2m_mean": "temp_mean",
        "precipitation_sum": "precip_mm", "precipitation_hours": "precip_hours", "wind_speed_10m_max": "wind_max",
        "sunshine_duration": "sunshine_h", "daylight_duration": "daylight_h"})
    w["date"] = pd.to_datetime(w["date"])
    w["sunshine_h"] /= 3600
    w["daylight_h"] /= 3600
    return w


def load_bank_holidays(path: Path) -> pd.DataFrame:
    ev = json.loads(Path(path).read_text())["england-and-wales"]["events"]
    return pd.DataFrame({"date": pd.to_datetime([e["date"] for e in ev]), "holiday": [e["title"] for e in ev]})


def merge_inputs(hires, weather, holidays, start="2019-01-01"):
    df = hires[hires.date >= start].merge(weather, on="date", how="left")
    df["bank_holiday"] = df.date.isin(holidays.date).astype(int)
    missing = df[WEATHER_COLS].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{missing} days have no weather")
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df.date
    df["dow"] = d.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["month"] = d.dt.month
    df["year"] = d.dt.year
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # Christmas to New Year is a well-known low-demand period that is not all bank holidays
    df["xmas_period"] = (((d.dt.month == 12) & (d.dt.day >= 24)) | ((d.dt.month == 1) & (d.dt.day == 1))).astype(int)
    df["non_working_day"] = ((df.is_weekend == 1) | (df.get("bank_holiday", 0) == 1)).astype(int)
    return df


def add_lags(df: pd.DataFrame, target="hires", lags=LAGS, horizon=HORIZON) -> pd.DataFrame:
    """Lag and rolling features using only information at least `horizon` days old.
    Assumes one row per consecutive day, sorted by date."""
    df = df.sort_values("date").copy()
    if not df.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("dates must be consecutive daily rows")
    df["anomaly"] = (df[target] < MIN_VALID_HIRES).astype(int)
    clean = df[target].where(df.anomaly == 0)  # outage days become missing so they never feed a lag
    for lag in lags:
        if lag < horizon:
            raise ValueError(f"lag {lag} would leak (< horizon {horizon})")
        df[f"lag_{lag}"] = clean.shift(lag)
    past = clean.shift(horizon)
    df["roll7_mean"] = past.rolling(7, min_periods=5).mean()
    df["roll28_mean"] = past.rolling(28, min_periods=20).mean()
    df["roll28_std"] = past.rolling(28, min_periods=20).std()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_lags(add_calendar(df))
    out["rain_x_weekend"] = out.precip_mm * out.is_weekend
    out["temp_sq"] = out.temp_max ** 2
    return out


FEATURES = (["dow", "is_weekend", "month", "doy_sin", "doy_cos", "bank_holiday", "xmas_period", "non_working_day"]
            + WEATHER_COLS + ["rain_x_weekend", "temp_sq"]
            + [f"lag_{l}" for l in LAGS] + ["roll7_mean", "roll28_mean", "roll28_std"])
