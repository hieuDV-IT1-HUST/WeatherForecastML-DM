import os
import sys
import joblib
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline

root = Path(os.getcwd()).resolve().parent.parent
if root not in sys.path:
    sys.path.append(str(root))

# Hàm đánh giá và quyết định lưu mô hình
def save_if_better(model: Pipeline, model_name: str, target_variable: str, r2_new: float):
    model_target_path = Path(root / "data" / "models" / f"{model_name}_{target_variable}.pkl")
    scores_csv_path = Path(root / "data" / "scores&predictions" / "SingleOutput" / target_variable / "model_scores.csv")

    if scores_csv_path.exists():
        df = pd.read_csv(scores_csv_path)
        row = df[df["model"] == model_name]
        if not row.empty:
            r2_old = float(row["r2"].values[0])
        else:
            r2_old = -float("inf")
    else:
        r2_old = -float("inf")

    if r2_new > r2_old:
        print(f"[INFO] New model better (R2: {r2_new:.4f} > {r2_old:.4f}), saving checkpoint.")
        joblib.dump(model, model_target_path)
    else:
        print(f"[INFO] New model worse (R2: {r2_new:.4f} <= {r2_old:.4f}), keeping previous checkpoint.")

