import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.config import load_config
import time

config = load_config()
weather_cfg = config["weather"]
OUTPUT_DIR = Path(weather_cfg["output_path"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"

hourly_fields = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
    "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "wind_gusts_10m", "wind_direction_100m", "wind_direction_10m", "wind_speed_100m",
    "wind_speed_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm", "is_day", "sunshine_duration", "albedo",
    "boundary_layer_height", "wet_bulb_temperature_2m",
    "total_column_integrated_water_vapour", "snow_depth_water_equivalent",
    "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
    "global_tilted_irradiance", "terrestrial_radiation", "shortwave_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant",
    "diffuse_radiation_instant", "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant", "terrestrial_radiation_instant"
]

start_year = 1940
end_year = 2025

for year in range(start_year, end_year + 1):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31" if year < end_year else "2025-04-20"

    params = {
        "latitude": weather_cfg["latitude"],
        "longitude": weather_cfg["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_fields,
        "timezone": weather_cfg["timezone"]
    }

    try:
        print(f"[INFO] Fetching data for {start_date} to {end_date}...")
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }

        for i, var_name in enumerate(hourly_fields):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(hourly_data)
        file_path = OUTPUT_DIR / f"openmeteo_hourly_{year}.csv"
        df.to_csv(file_path, index=False)
        print(f"[✅] Saved hourly data to {file_path}")

    except Exception as e:
        print(f"[❌] Failed to fetch data for {year}: {e}")

    time.sleep(10)  # sleep 10s to avoid API rate limit
