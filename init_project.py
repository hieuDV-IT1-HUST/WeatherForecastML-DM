import os

structure = {
    "data": ["raw", "processed", "trained_data", "scores&predictions"],
    "notebooks": {
        "EDA": ["EDA.ipynb", "EDA_daily.ipynb"],
        "models": ["FeaturesFilter.ipynb", "LinearRegression.ipynb"],
        "pre_process":["preprocess.ipynb", "preprocess_daily.ipynb"]
    },
    "src": {
        "data": ["fetch_data_visualcrossing.py", "historical_openmeteo_daily", "historical_openmeteo_hourly",
                 "merge_hourly_data.py", "save_results.py"],
        "utils": ["config.py", "data_loader.py"]
    },
    "api": ["main.py", "routes.py", "model_loader.py"],
    "": ["README.md", ".gitignore", "requirements.txt", "config.yaml"]
}

def create_structure(base_path="."):
    def make(path):
        full_path = os.path.join(base_path, path)
        dir_name = os.path.dirname(full_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if "." in os.path.basename(full_path):
            open(full_path, 'w').close()
        else:
            os.makedirs(full_path, exist_ok=True)

    for key, value in structure.items():
        if isinstance(value, list):
            for item in value:
                make(os.path.join(key, item))
        elif isinstance(value, dict):
            for subkey, files in value.items():
                for f in files:
                    make(os.path.join(key, subkey, f))

    print("✅ Project is initialized into current folder.")

if __name__ == "__main__":
    create_structure()
