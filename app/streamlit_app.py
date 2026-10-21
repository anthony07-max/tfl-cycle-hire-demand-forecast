"""Streamlit app: Santander Cycles demand and 7-day-ahead forecast. Run: streamlit run app/streamlit_app.py"""
import json
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
st.set_page_config(page_title="TfL Cycle Hire Forecast", layout="wide")


@st.cache_data
def load():
    feats = pd.read_csv(P / "features.csv", parse_dates=["date"])
    return (feats, pd.read_csv(P / "cv_results.csv"), pd.read_csv(P / "holdout_predictions.csv", parse_dates=["date"]),
            pd.read_csv(P / "feature_importance.csv"), json.loads((P / "key_findings.json").read_text()))


@st.cache_resource
def model():
    return joblib.load(ROOT / "models" / "lgbm_cycle_hires.joblib")


feats, cv, hold, imp, F = load()
M = model()
st.title("Santander Cycles (TfL cycle hire) demand forecasting")
st.caption(f"TfL daily hires {F['first_day']} to {F['last_day']}, Open-Meteo weather and GOV.UK bank holidays. "
           "Built by Anthony Adusei-Poku (github.com/anthony07-max).")
c = st.columns(4)
c[0].metric("Hires Jan-Sep 2026", f"{F['jan_sep_2026']/1e6:.2f}m", f"{F['jan_sep_2026_vs_2025_pct']:+.1f}% vs 2025")
c[1].metric("Record day", f"{F['record_hires']:,}", F["record_day"], delta_color="off")
c[2].metric("Hold-out MAPE (LightGBM)", f"{F['holdout']['lightgbm']['mape']}%",
            f"naive {F['holdout']['seasonal_naive']['mape']}%", delta_color="off")
c[3].metric("Wet summer workday", f"-{F['wet_day_drop_pct']}%", "5mm+ rain vs dry", delta_color="off")

t1, t2, t3, t4 = st.tabs(["Demand history", "Weather & calendar", "Model evaluation", "What-if forecast"])
with t1:
    d = feats[feats.anomaly == 0]
    win = st.slider("Rolling average (days)", 1, 56, 28)
    fig = go.Figure([go.Scatter(x=d.date, y=d.hires, name="Daily", line=dict(width=.6, color="#bbb")),
                     go.Scatter(x=d.date, y=d.hires.rolling(win, center=True).mean(), name=f"{win}-day mean", line=dict(color="#e4002b"))])
    st.plotly_chart(fig, width="stretch")
with t2:
    d = feats[(feats.anomaly == 0)].assign(day_type=lambda x: x.non_working_day.map({0: "Working day", 1: "Weekend / bank holiday"}))
    months = st.slider("Months", 1, 12, (4, 9))
    d = d[d.month.between(*months)]
    st.plotly_chart(px.scatter(d, x="temp_max", y="hires", color="precip_mm", facet_col="day_type", opacity=.6,
                               color_continuous_scale="Blues", labels={"temp_max": "Max temp (°C)", "precip_mm": "Rain (mm)"}),
                    width="stretch")
with t3:
    st.plotly_chart(px.bar(cv, x="fold", y="mape", color="model", barmode="group", hover_data=["test_start", "test_end", "mae"],
                           title="Time-series CV: MAPE by fold"), width="stretch")
    long = hold.melt("date", ["hires", "lightgbm", "seasonal_naive", "ridge"], var_name="series", value_name="value")
    st.plotly_chart(px.line(long, x="date", y="value", color="series", title=f"Hold-out year ({F['holdout_period']})"), width="stretch")
    st.dataframe(pd.DataFrame(F["holdout"]).T.rename_axis("model"), width="stretch")
    st.plotly_chart(px.bar(imp.head(15)[::-1], x="gain_pct", y="feature", orientation="h", title="Feature importance (gain %)"), width="stretch")
with t4:
    st.write("Pick a real day, then change its weather to see how the saved LightGBM model's 7-day-ahead forecast responds.")
    valid = feats[(feats.anomaly == 0)].dropna(subset=M["features"])
    day = st.date_input("Day", valid.date.max().date(), min_value=valid.date.min().date(), max_value=valid.date.max().date())
    row = valid[valid.date == pd.Timestamp(day)]
    if row.empty:
        st.warning("No complete feature row for that day (outage nearby).")
    else:
        row = row.iloc[0].copy()
        a, b, cc = st.columns(3)
        row["temp_max"] = a.slider("Max temp (°C)", -5.0, 38.0, float(row.temp_max))
        row["temp_mean"] = row.temp_max - 4
        row["precip_mm"] = b.slider("Rain (mm)", 0.0, 40.0, float(row.precip_mm))
        row["precip_hours"] = min(24.0, row.precip_hours if row.precip_mm > 0 else 0.0)
        row["sunshine_h"] = cc.slider("Sunshine (hours)", 0.0, 16.0, float(row.sunshine_h))
        row["temp_sq"], row["rain_x_weekend"] = row.temp_max ** 2, row.precip_mm * row.is_weekend
        pred = M["model"].predict(pd.DataFrame([row[M["features"]]]).astype(float))[0]
        st.metric("Forecast hires", f"{pred:,.0f}", f"actual on the day: {int(row.hires):,}", delta_color="off")
        st.caption("Weather means and rain hours are approximated from your inputs. The model was refit on all data, "
                   "so days shown here are in-sample.")
