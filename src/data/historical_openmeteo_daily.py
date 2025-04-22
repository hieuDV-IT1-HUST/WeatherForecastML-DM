import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path
from datetime import date
from src.utils.config import load_config

# Load config
config = load_config()
weather_cfg = config["weather"]
daily_fields = weather_cfg["daily_fields"]
OUTPUT_DIR = Path(weather_cfg["output_path"])

# Setup Open-Meteo client with retry & caching
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# API URL and parameters
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": weather_cfg["latitude"],
    "longitude": weather_cfg["longitude"],
    "start_date": "1980-01-01",
    "end_date": "2020-12-31",
    "daily": weather_cfg["daily_fields"],
    "timezone": weather_cfg["timezone"]
}

# Call API
responses = openmeteo.weather_api(url, params=params)
response = responses[0]
daily = response.Daily()

daily_data = {
    "date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
}

for i, var_name in enumerate(daily_fields):
    daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()

# Convert to DataFrame and export
df = pd.DataFrame(daily_data)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
file_path = OUTPUT_DIR / f"openmeteo_historical_daily.csv"
df.to_csv(file_path, index=False, encoding="utf-8")
print(f"[✅] Saved historical weather data to {file_path}")
