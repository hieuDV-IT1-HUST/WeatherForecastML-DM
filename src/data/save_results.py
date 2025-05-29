import pandas as pd
import numpy as np
from pathlib import Path

def save_model_scores(model_name: str, mae: float, rmse: float, r2: float,
                      cv_rmse_mean: float, cv_rmse_std: float, cv_mae_mean: float, cv_mae_std: float,
                      cv_r2_mean: float, cv_r2_std: float, save_dir: Path):
    """Save model scores to model_scores.csv"""
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    scores_path = save_dir / "model_scores.csv"

    score_row = pd.DataFrame([{
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
    }])

    if scores_path.exists():
        prev_scores = pd.read_csv(scores_path)
        prev_scores = prev_scores[prev_scores["model"] != model_name]
        scores_df = pd.concat([prev_scores, score_row], ignore_index=True)
    else:
        scores_df = score_row

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

# Example usage in notebook:
# from src.data.save_results import save_model_scores, save_model_predictions
# save_model_scores("LGBMR", mae, rmse, r2, -np.mean(cv_scores), np.std(cv_scores), TRAINED_DATA / METHOD)
# save_model_predictions("LGBMR", y_test, y_pred_lbgmr, TRAINED_DATA / METHOD)
