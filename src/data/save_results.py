from typing import Literal
import pandas as pd
import numpy as np
from pathlib import Path

import pandas as pd
from pathlib import Path

def save_model_scores(model_name: str, save_dir: Path,
                      mae: float | None = None, rmse: float | None = None, r2: float | None = None,
                      cv_rmse_mean: float | None = None, cv_rmse_std: float | None = None,
                      cv_mae_mean: float | None = None, cv_mae_std: float | None = None,
                      cv_r2_mean: float | None = None, cv_r2_std: float | None = None,
                      category: Literal["classifier", "regressor"] = "regressor",
                      accuracy: float | None = None, f1: float | None = None, roc_auc: float | None = None,
                      cv_accuracy_mean: float | None = None, cv_accuracy_std: float | None = None,
                      cv_f1_mean: float | None = None, cv_f1_std: float | None = None,
                      cv_roc_auc_mean: float | None = None, cv_roc_auc_std: float | None = None
                      ):
    """Save model scores to model_scores.csv without overwriting non-None values with None"""

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    scores_path = save_dir / "model_scores.csv"

    if category == "regressor":
        current_scores = {
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std
        }
    else:
        current_scores = {
            "model": model_name,
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc,
            "cv_accuracy_mean": cv_accuracy_mean,
            "cv_accuracy_std": cv_accuracy_std,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "cv_roc_auc_mean": cv_roc_auc_mean,
            "cv_roc_auc_std": cv_roc_auc_std,
        }
    # Prepare current update

    if scores_path.exists():
        prev_scores = pd.read_csv(scores_path)

        if model_name in prev_scores["model"].values:
            # Update only the fields with non-None values
            idx = prev_scores[prev_scores["model"] == model_name].index[0]
            for key, value in current_scores.items():
                if key == "model":
                    continue
                if value is not None:
                    prev_scores.at[idx, key] = value
            scores_df = prev_scores
        else:
            # Append new model
            score_row = pd.DataFrame([current_scores])
            scores_df = pd.concat([prev_scores, score_row], ignore_index=True)
    else:
        # Create new file with the row
        scores_df = pd.DataFrame([current_scores])

    scores_df.to_csv(scores_path, index=False)

def save_model_predictions(model_name: str, y_true: pd.Series, y_pred: np.ndarray, save_dir: Path):
    """Save model predictions to model_predictions.csv"""
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    preds_path = save_dir / "model_predictions.csv"

    pred_df = pd.DataFrame({
        "actual": y_true.values,
        f"{model_name}_pred": y_pred
    })

    if preds_path.exists():
        prev_preds = pd.read_csv(preds_path)
        # So sánh actual
        if not np.allclose(prev_preds["actual"].values, pred_df["actual"].values, rtol=1e-5, atol=1e-8):
            raise ValueError(f"[ERROR] 'actual' values do not match existing file! "
                            f"Ensure you are using the same y_test.")
        # Drop old prediction column of the same model if exists
        prev_preds = prev_preds.drop(columns=[col for col in prev_preds.columns if col == f"{model_name}_pred"], errors='ignore')
        # Ensure "actual" is from the new y_true
        prev_preds = prev_preds.drop(columns=["actual"], errors='ignore')
        preds_df = pd.concat([pred_df[["actual"]], prev_preds, pred_df[[f"{model_name}_pred"]]], axis=1)
    else:
        preds_df = pred_df

    preds_df.to_csv(preds_path, index=False)
