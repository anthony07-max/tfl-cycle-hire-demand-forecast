"""Exploratory statistics, model evaluation charts and key_findings.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, C = ROOT / "data" / "processed", ROOT / "charts"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3})
RED, BLUE, GREY = "#e4002b", "#0019a8", "#888888"  # Santander red, TfL blue


def save(fig, name):
    fig.tight_layout(); fig.savefig(C / name, bbox_inches="tight"); plt.close(fig)


def main():
    C.mkdir(exist_ok=True)
    d = pd.read_csv(P / "daily.csv", parse_dates=["date"])
    d = d[d.hires >= 1000].copy()  # drop the 3 outage days
    d["dow"], d["year"], d["month"] = d.date.dt.dayofweek, d.date.dt.year, d.date.dt.month
    d["working"] = (d.dow < 5) & (d.bank_holiday == 0)
    F = {"first_day": str(d.date.min().date()), "last_day": str(d.date.max().date()), "n_days": len(d)}

    # 1. Long-run series
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d.date, d.hires, color=GREY, lw=.4, alpha=.6, label="Daily hires")
    ax.plot(d.date, d.set_index("date").hires.rolling(28, center=True).mean().values, color=RED, lw=2, label="28-day average")
    ax.set_title("Santander Cycles daily hires, Jan 2019 to Sep 2026"); ax.set_ylabel("Hires per day"); ax.legend()
    save(fig, "01_daily_hires.png")

    full = d[d.year <= 2025].groupby("year").hires.agg(["sum", "mean"])
    F["annual_totals"] = {int(k): int(v) for k, v in full["sum"].items()}
    best = d.loc[d.hires.idxmax()]
    F["record_day"], F["record_hires"] = str(best.date.date()), int(best.hires)
    ytd = d[d.month <= 9].groupby("year").hires.sum()
    F["jan_sep_2026"], F["jan_sep_2025"], F["jan_sep_2019"] = int(ytd[2026]), int(ytd[2025]), int(ytd[2019])
    F["jan_sep_2026_vs_2025_pct"] = round(100 * (ytd[2026] / ytd[2025] - 1), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ytd.index.astype(str), ytd.values / 1e6, color=[RED if y == 2026 else BLUE for y in ytd.index])
    for x, v in zip(ytd.index.astype(str), ytd.values): ax.text(x, v / 1e6 + .05, f"{v/1e6:.2f}m", ha="center", fontsize=8)
    ax.set_title("Hires January to September, by year"); ax.set_ylabel("Millions")
    save(fig, "02_jan_sep_by_year.png")

    # 2. Weather effects (working days, Apr-Sep only to hold season roughly fixed)
    s = d[d.working & d.month.between(4, 9)]
    dry, wet = s[s.precip_mm < 0.5].hires.median(), s[s.precip_mm >= 5].hires.median()
    F["summer_workday_median_dry"], F["summer_workday_median_wet5mm"] = int(dry), int(wet)
    F["wet_day_drop_pct"] = round(100 * (1 - wet / dry), 1)
    ww = d[d.working]
    slope = np.polyfit(ww.temp_max, ww.hires, 1)[0]
    F["hires_per_degC_workdays"] = int(round(slope))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sc = axes[0].scatter(ww.temp_max, ww.hires, c=np.log1p(ww.precip_mm), cmap="Blues", s=6)
    axes[0].set_xlabel("Max temperature (°C)"); axes[0].set_ylabel("Hires"); axes[0].set_title("Working days: warmer = busier, rain (darker) = quieter")
    fig.colorbar(sc, ax=axes[0], label="log(1 + rain mm)")
    bins = pd.cut(s.precip_mm, [-.1, .5, 2, 5, 10, 100], labels=["<0.5", "0.5-2", "2-5", "5-10", "10+"])
    med = s.groupby(bins, observed=True).hires.median()
    axes[1].bar(med.index.astype(str), med.values, color=BLUE)
    axes[1].set_xlabel("Daily rain (mm)"); axes[1].set_title("Median hires, Apr-Sep working days")
    save(fig, "03_weather_effects.png")

    # 3. Weekly pattern and bank holidays
    dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by = d[d.bank_holiday == 0].groupby("dow").hires.mean()
    F["mean_by_weekday"] = {dn[i]: int(v) for i, v in by.items()}
    bh = d[d.bank_holiday == 1].hires.mean()
    F["bank_holiday_mean"], F["workday_mean"] = int(bh), int(d[d.working].hires.mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dn + ["Bank hol"], list(by.values) + [bh], color=[BLUE] * 5 + [RED] * 2 + [GREY])
    ax.set_title("Average daily hires by day type (2019-2026)"); ax.set_ylabel("Hires")
    save(fig, "04_day_of_week.png")

    # 4. Model results
    cv = pd.read_csv(P / "cv_results.csv")
    names = {"seasonal_naive": "Seasonal naive (last week)", "ridge": "Ridge regression", "lightgbm": "LightGBM"}
    fig, ax = plt.subplots(figsize=(9, 4))
    w = .27
    for i, m in enumerate(names):
        sub = cv[cv.model == m]
        ax.bar(sub.fold + (i - 1) * w, sub.mape, w, label=names[m], color=[GREY, BLUE, RED][i])
    ax.set_xticks(cv.fold.unique(), [f"Fold {f}\n{r.test_start[:7]}" for f, r in cv[cv.model == "lightgbm"].set_index("fold").iterrows()])
    ax.set_ylabel("MAPE (%)"); ax.set_title("Time-series cross-validation (5 x 91-day folds, 7-day-ahead)"); ax.legend()
    save(fig, "05_cv_mape.png")
    F["cv_mean"] = {m: {"mae": round(g.mae.mean()), "mape": round(g.mape.mean(), 1)} for m, g in cv.groupby("model")}

    h = pd.read_csv(P / "holdout_predictions.csv", parse_dates=["date"])
    def m_(c): return round(float(np.mean(np.abs(h.hires - h[c]))) ), round(float(np.mean(np.abs(h.hires - h[c]) / h.hires) * 100), 1)
    F["holdout"] = {c: dict(zip(["mae", "mape"], m_(c))) for c in names}
    F["holdout_period"] = f"{h.date.min().date()} to {h.date.max().date()}"
    F["mae_reduction_vs_naive_pct"] = round(100 * (1 - F["holdout"]["lightgbm"]["mae"] / F["holdout"]["seasonal_naive"]["mae"]), 1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(h.date, h.hires, color="black", lw=1, label="Actual")
    ax.plot(h.date, h.lightgbm, color=RED, lw=1, label=f"LightGBM (MAPE {F['holdout']['lightgbm']['mape']}%)")
    ax.plot(h.date, h.seasonal_naive, color=GREY, lw=.8, alpha=.7, label=f"Seasonal naive (MAPE {F['holdout']['seasonal_naive']['mape']}%)")
    ax.set_title("Hold-out year: 7-day-ahead forecasts vs actual"); ax.legend(); ax.set_ylabel("Hires")
    save(fig, "06_holdout_forecast.png")
    err = (h.lightgbm - h.hires)
    worst = h.assign(err=err).loc[err.abs().nlargest(3).index]
    F["holdout_worst_days"] = [{"date": str(r.date.date()), "actual": int(r.hires), "pred": int(r.lightgbm)} for r in worst.itertuples()]

    imp = pd.read_csv(P / "feature_importance.csv").head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.feature, imp.gain_pct, color=BLUE); ax.set_xlabel("% of total gain"); ax.set_title("LightGBM feature importance (top 12)")
    save(fig, "07_feature_importance.png")
    F["top_features"] = pd.read_csv(P / "feature_importance.csv").head(5)[["feature", "gain_pct"]].values.tolist()

    (P / "key_findings.json").write_text(json.dumps(F, indent=2, default=str))
    print(json.dumps(F, indent=2, default=str))


if __name__ == "__main__":
    main()
