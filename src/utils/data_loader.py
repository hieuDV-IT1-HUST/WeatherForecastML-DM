import os
from pathlib import Path
import pandas as pd
import joblib
import json
from src.utils.config import load_config

def load_preprocessed_data(target_variable: str, config_path="../../config.yaml"):
    """Load preprocessed data and pipeline components based on config."""
    
    config = load_config(config_path)
    
    root = Path(os.getcwd()).resolve().parent.parent
    method = config["method"]
    
    data_dir = root / "data"
    trained_data_dir = data_dir / "trained_data"
    target_variable_dir = trained_data_dir / method / target_variable
    data = {
        "ROOT": root,
        "DATA": data_dir,
        "TRAINED_DATA": trained_data_dir,
        "METHOD": method,
        "selected_features": json.load(open(target_variable_dir / "selected_features.json")),
        "X_train": joblib.load(target_variable_dir / "X_train.pkl"),
        "X_test": joblib.load(target_variable_dir / "X_test.pkl"),
        "y_train": joblib.load(target_variable_dir / "y_train.pkl"),
        "y_test": joblib.load(target_variable_dir / "y_test.pkl"),
        "preprocessor": joblib.load(target_variable_dir / "preprocessor.pkl")
    }

    return data
