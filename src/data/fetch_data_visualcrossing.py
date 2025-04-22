import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.utils.config import load_config

config = load_config()
weather_cfg = config["weather"]
vc_cfg = config.get("visualcrossing", {})
OUTPUT_DIR = Path(weather_cfg["output_path"])

def fetch_historical_visualcrossing(start_year=2015, end_year=2024):
    api_key = vc_cfg.get("api_key")
    if not api_key:
        print("[ERROR] Visual Crossing API key not found in config.yaml")
        return

    latitude = weather_cfg["latitude"]
    longitude = weather_cfg["longitude"]
    location = f"{latitude},{longitude}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}"
        params = {
            "unitGroup": "metric",
            "key": api_key,
            "include": "days",
            "contentType": "json"
        }

        print(f"[INFO] Fetching {year}...")
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"[ERROR] Failed for {year}: {response.status_code}")
            print(response.text)
            continue

        data = response.json()
        days = data.get("days", [])
        if not days:
            print(f"[WARNING] No data returned for {year}")
            continue

        df = pd.DataFrame(days)
        df["year"] = year

        file_path = OUTPUT_DIR / f"visualcrossing_{year}.csv"
        df.to_csv(file_path, index=False)
        print(f"[✅] Saved data for {year} to {file_path}")

if __name__ == "__main__":
    fetch_historical_visualcrossing(start_year=2015, end_year=2016)
