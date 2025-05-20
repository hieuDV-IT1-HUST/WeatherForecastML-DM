import pandas as pd
from pathlib import Path

def merge_hourly_data(input_dir="data/raw", output_file="data/processed/full_hourly_weather.csv"):
    data_dir = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("openmeteo_hourly_*.csv"))
    if not all_files:
        print("[❌] No files found to merge.")
        return

    print(f"[INFO] Found {len(all_files)} files. Merging...")
    df_all = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)
    df_all.to_csv(output_path, index=False)
    print(f"[✅] Merged file saved to: {output_path}")

if __name__ == "__main__":
    merge_hourly_data()
