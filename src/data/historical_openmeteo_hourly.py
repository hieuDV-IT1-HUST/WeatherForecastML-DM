import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path
from datetime import date

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 21.0285,
    "longitude": 105.8542,
    "start_date": "1947-01-29",
    "end_date": "2025-04-20",
    "hourly": [
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
    ],
    "timezone": "Asia/Bangkok"
}

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

hourly_fields = params["hourly"]
for i, var_name in enumerate(hourly_fields):
    hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

df = pd.DataFrame(hourly_data)
file_path = OUTPUT_DIR / f"openmeteo_hourly.csv"
df.to_csv(file_path, index=False)
print(f"[✅] Saved hourly data to {file_path}")
