import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
import sys
from pathlib import Path
path = Path(os.getcwd()).resolve()
if path not in sys.path:
    sys.path.append(str(path))
from IPython.display import display
import dataframe_image as dfi
from matplotlib import colors

DATA = path / "data"
TRAINED_DATA = DATA / "trained_data"
MODELS = DATA / "models"
METHOD = "SingleOutput"

target_variables = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "rain",
    "shortwave_radiation"
]

def load_and_process_scores(data_dir, method, target_variable):
    file_path = os.path.join(data_dir, "scores&predictions", method, target_variable, "model_scores.csv")
    df = pd.read_csv(file_path)

    df["CV MAE"] = df["cv_mae_mean"].round(3).astype(str) + " ± " + df["cv_mae_std"].round(3).astype(str)
    df["CV RMSE"] = df["cv_rmse_mean"].round(3).astype(str) + " ± " + df["cv_rmse_std"].round(3).astype(str)
    df["CV R²"] = df["cv_r2_mean"].round(3).astype(str) + " ± " + df["cv_r2_std"].round(3).astype(str)

    df_scores = pd.DataFrame({
        "Model": df["model"],
        "MAE": df["mae"].round(3),
        "RMSE": df["rmse"].round(3),
        "R²": df["r2"].round(3),
        "CV MAE": df["CV MAE"],
        "CV RMSE": df["CV RMSE"],
        "CV R²": df["CV R²"],
        "cv_mae_mean": df["cv_mae_mean"],
        "cv_rmse_mean": df["cv_rmse_mean"],
        "cv_r2_mean": df["cv_r2_mean"],
    })

    # Sắp xếp theo R² giảm dần
    df_scores = df_scores.sort_values(by="R²", ascending=False).reset_index(drop=True)

    return df_scores

def highlight_gradient_from_values(values, color="YlGn"):

    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.colormaps.get_cmap(color)

    styles = []
    for v in values:
        rgba = cmap(norm(v))
        r, g, b = [int(255 * c) for c in rgba[:3]]
        # Độ sáng dựa theo công thức của W3C
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "black" if brightness > 140 else "white"
        styles.append(
            f"background-color: rgb({r}, {g}, {b}); color: {text_color}"
        )
    return styles


def display_styled_table(df, title=None):
    df_display = df[["Model", "MAE", "RMSE", "R²", "CV MAE", "CV RMSE", "CV R²"]].copy()

    styled = df_display.style \
        .apply(lambda _: highlight_gradient_from_values(df["MAE"]), subset=["MAE"], axis=0) \
        .apply(lambda _: highlight_gradient_from_values(df["RMSE"]), subset=["RMSE"], axis=0) \
        .apply(lambda _: highlight_gradient_from_values(df["R²"]), subset=["R²"], axis=0) \
        .apply(lambda _: highlight_gradient_from_values(df["cv_mae_mean"]), subset=["CV MAE"], axis=0) \
        .apply(lambda _: highlight_gradient_from_values(df["cv_rmse_mean"]), subset=["CV RMSE"], axis=0) \
        .apply(lambda _: highlight_gradient_from_values(df["cv_r2_mean"]), subset=["CV R²"], axis=0) \
        .format(precision=3, na_rep="-") \
        .set_caption(title) \
        .set_properties(**{"text-align": "center"}) \
        .set_table_styles([
            {"selector": "caption", "props": [("caption-side", "top"), ("font-weight", "bold"), ("font-size", "16px")]},
            {"selector": "th", "props": [("text-align", "center")]},  # căn giữa tiêu đề
            {"selector": "td", "props": [("text-align", "center")]},  # căn giữa nội dung ô
        ])
    
    return styled

def display_styled_scores_table(df: pd.DataFrame, title="Evaluation Metrics"):
    return (
        df.style.set_caption(title)
        .format({
            "MAE (rad)": "{:.5f}", "MAE (°)": "{:.3f}",
            "RMSE (rad)": "{:.5f}", "RMSE (°)": "{:.3f}",
            "Angular R²": "{:.5f}"
        })
        .set_table_styles([{
            'selector': 'caption',
            'props': [('color', 'black'), ('font-size', '16px'), ('text-align', 'center')]
        }])
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])], overwrite=False)
        .background_gradient(subset=["MAE (°)", "RMSE (°)"], cmap='Reds')
        .background_gradient(subset=["Angular R²"], cmap='Greens')
        .background_gradient(subset=["MAE (rad)", "RMSE (rad)"], cmap='Blues')
        .hide()
    )

# df_results = pd.read_csv(Path(DATA / "scores&predictions" / METHOD / "wind_direction_10m" / "model_scores.csv"))
# # Hiển thị bảng
# display(display_styled_scores_table(df_results, "Wind Direction Evaluation (Angular Metrics)"))

# dfi.export(display_styled_scores_table(df_results),
#            Path(DATA / "image" / "wind_direction_10m_model_scores.png"))

for target_variable in target_variables:
    df_scores = load_and_process_scores(DATA, METHOD, target_variable)
    styled = display_styled_table(df_scores, title=f"Performance for {target_variable}")
    IMAGE_PATH = Path(DATA / "image"/ "model_scores" / f"{target_variable}_model_scores.png")
    dfi.export(styled, IMAGE_PATH)

    