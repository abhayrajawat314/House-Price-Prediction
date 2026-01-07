import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.Components.MultimodalFusionPipeline import MultimodalFusionPipeline
from src.Components.fusion_model import build_fusion_model


TEST_CSV = "Data/final_train_test_data/test_df.csv"
TEST_PRICE_CSV = None

MODEL_PATH = "artifacts/fusion_model.keras"
OUTPUT_PATH = "artifacts/test_predictions.csv"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


pipeline = MultimodalFusionPipeline(
    train_csv_path=TEST_CSV,
    price_csv_path=TEST_PRICE_CSV,
    tabular_encoder_path="artifacts/tabular_encoder.keras",
    tabular_scaler_path="artifacts/tabular_scaler.pkl",
    tabular_features_path="artifacts/tabular_features.pkl",
    satellite_encoder_path="artifacts/satellite_encoder.keras",
    satellite_image_dir="Data/train_images_CNN/satellite"
)

ids, tab_emb, sat_emb = pipeline.generate_embeddings()

model = build_fusion_model(
    tab_dim=tab_emb.shape[1],
    sat_dim=sat_emb.shape[1]
)
model.load_weights(MODEL_PATH)

log_pred = model.predict([tab_emb, sat_emb]).squeeze()
price_pred = np.exp(log_pred)

results = pd.DataFrame({
    "id": ids,
    # "predicted_log_price": log_pred,
    "predicted_price": price_pred
})

if TEST_PRICE_CSV:
    price_df = pd.read_csv(TEST_PRICE_CSV)
    id_to_y = dict(zip(price_df["id"].astype(str), price_df["log_price"]))

    y_true_log = np.array([id_to_y[i] for i in ids])
    y_true = np.exp(y_true_log)

    print("\n📊 Evaluation")
    print("RMSE:", rmse(y_true, price_pred))
    print("MAE :", mean_absolute_error(y_true, price_pred))
    print("R²  :", r2_score(y_true, price_pred))

results.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved → {OUTPUT_PATH}")
