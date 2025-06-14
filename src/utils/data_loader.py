import os
from pathlib import Path
import joblib
import json

def load_preprocessed_data(method="SingleOutput", target_variable=""):
    """Load preprocessed data and pipeline components based on config."""
    
    root = Path(os.getcwd()).resolve().parent.parent
    
    data_dir = root / "data"
    trained_data_dir = data_dir / "trained_data"
    if target_variable != "":
        target_variable_dir = trained_data_dir / method / target_variable
    else:
        target_variable_dir = trained_data_dir / method
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
