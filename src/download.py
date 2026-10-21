"""Download the three public inputs into data/raw/.

1. TfL daily Santander Cycles hires (London Datastore, "Number of Bicycle Hires", OGL v2)
2. Open-Meteo historical weather for central London (CC BY 4.0)
3. GOV.UK bank holidays for England and Wales (OGL v3)
"""
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
START, END = "2019-01-01", "2026-09-30"  # GOV.UK bank-holiday JSON starts in 2019; hires published to Sep 2026
FILES = {
    # London Datastore re-publishes this file monthly; update the URL from
    # https://data.london.gov.uk/dataset/number-bicycle-hires when a newer month appears.
    "tfl_daily_hires.xlsx": "https://data.london.gov.uk/download/2r84d/ac29363e-e0cb-47cc-a97a-e216d900a6b0/tfl-daily-cycle-hires-sep26-update.xlsx",
    "weather_london.json": (
        "https://archive-api.open-meteo.com/v1/archive?latitude=51.5072&longitude=-0.1276"
        f"&start_date={START}&end_date={END}"
        "&daily=temperature_2m_max,temperature_2m_mean,precipitation_sum,precipitation_hours,"
        "wind_speed_10m_max,sunshine_duration,daylight_duration&timezone=Europe%2FLondon"
    ),
    "bank_holidays.json": "https://www.gov.uk/bank-holidays.json",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio project; public open data)"}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print("Downloading", url[:110])
        r = requests.get(url, headers=HEADERS, timeout=180)
        r.raise_for_status()
        (RAW / name).write_bytes(r.content)
    print("Done:", sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()
