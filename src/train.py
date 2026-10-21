"""Time-series cross-validation, final hold-out test, saved model and an experiment log.

Models compared (7-day-ahead daily forecast):
  * seasonal naive  - same weekday last week (lag 7)
  * ridge regression on the same features (linear benchmark)
  * LightGBM gradient boosting
"""
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES, HORIZON

ROOT = Path(__file__).resolve().parents[1]
P, MODELS, EXP = ROOT / "data" / "processed", ROOT / "models", ROOT / "experiments"
HOLDOUT_DAYS = 365
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=2, verbose=-1)


def mae(y, p): return float(np.mean(np.abs(y - p)))
def mape(y, p): return float(np.mean(np.abs(y - p) / y) * 100)


def models():
    return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "lightgbm": lgb.LGBMRegressor(**LGB_PARAMS)}


def evaluate(train, test):
    out, preds = {}, {"seasonal_naive": test.lag_7.values}
    for name, m in models().items():
        m.fit(train[FEATURES], train.hires)
        preds[name] = m.predict(test[FEATURES])
    for name, p in preds.items():
        out[name] = {"mae": mae(test.hires.values, p), "mape": mape(test.hires.values, p)}
    return out, preds


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    t0 = time.time()
    df = pd.read_csv(P / "features.csv", parse_dates=["date"])
    n_anom = int(df.anomaly.sum())
    df = df[df.anomaly == 0].dropna(subset=FEATURES).reset_index(drop=True)
    print(f"excluded {n_anom} outage days (<1,000 hires); modelling {len(df)} days")
    dev, hold = df.iloc[:-HOLDOUT_DAYS], df.iloc[-HOLDOUT_DAYS:]

    # Expanding-window CV on the development period: 5 folds of 91 days each
    cv_rows = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5, test_size=91).split(dev)):
        res, _ = evaluate(dev.iloc[tr], dev.iloc[te])
        for name, r in res.items():
            cv_rows.append({"fold": k + 1, "test_start": dev.iloc[te].date.min().date(), "test_end": dev.iloc[te].date.max().date(),
                            "model": name, **r})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(P / "cv_results.csv", index=False)
    cv_summary = cv.groupby("model")[["mae", "mape"]].mean().round(2)
    print("CV (mean of 5 folds)\n", cv_summary)

    # Final hold-out: last 365 days, trained on everything before
    res, preds = evaluate(dev, hold)
    hold_df = hold[["date", "hires"]].copy()
    for name, p in preds.items():
        hold_df[name] = np.round(p).astype(int)
    hold_df.to_csv(P / "holdout_predictions.csv", index=False)
    print("Hold-out", json.dumps(res, indent=1))

    # Refit LightGBM on all data and save it
    final = lgb.LGBMRegressor(**LGB_PARAMS).fit(df[FEATURES], df.hires)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "horizon_days": HORIZON,
                 "trained_to": str(df.date.max().date())}, MODELS / "lgbm_cycle_hires.joblib")
    imp = pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")})
    imp["gain_pct"] = (100 * imp.gain / imp.gain.sum()).round(2)
    imp.sort_values("gain", ascending=False).to_csv(P / "feature_importance.csv", index=False)

    # Experiment tracking: one JSON line per run
    EXP.mkdir(exist_ok=True)
    run = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
           "data_from": str(df.date.min().date()), "data_to": str(df.date.max().date()), "n_rows": len(df),
           "horizon_days": HORIZON, "holdout_days": HOLDOUT_DAYS, "features": FEATURES, "lightgbm_params": LGB_PARAMS,
           "cv_mean": cv_summary.to_dict(orient="index"), "holdout": res, "seconds": round(time.time() - t0, 1)}
    with open(EXP / "runs.jsonl", "a") as f:
        f.write(json.dumps(run, default=str) + "\n")
    print("logged run to experiments/runs.jsonl")


if __name__ == "__main__":
    main()
