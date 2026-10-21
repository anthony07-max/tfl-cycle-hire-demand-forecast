"""Merge hires, weather and bank holidays into data/processed/daily.csv and build the feature table."""
from pathlib import Path
from features import load_hires, load_weather, load_bank_holidays, merge_inputs, build_features

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = merge_inputs(load_hires(RAW / "tfl_daily_hires.xlsx"), load_weather(RAW / "weather_london.json"),
                      load_bank_holidays(RAW / "bank_holidays.json"))
    df.to_csv(OUT / "daily.csv", index=False)
    feats = build_features(df)
    feats.to_csv(OUT / "features.csv", index=False)
    print(f"{len(df)} days {df.date.min().date()} to {df.date.max().date()}, {df.bank_holiday.sum()} bank holidays")


if __name__ == "__main__":
    main()
