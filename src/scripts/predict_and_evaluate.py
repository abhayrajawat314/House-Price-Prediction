import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.Components.MultimodalFusionPipeline import MultimodalFusionPipeline
from src.Components.fusion_model import build_fusion_model


TEST_CSV = "Data/final_train_test_data/train_df.csv"
TEST_PRICE_CSV = None  

FUSION_WEIGHTS_PATH = "artifacts/fusion_model.keras"
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
    street_encoder_path="artifacts/street_encoder.keras",
    satellite_image_dir="Data/train_images_CNN/satellite",
    street_image_dir="Data/train_images_CNN/street",
    missing_street_ids_csv="Data/train_images_CNN/missing_street_ids.csv"
)


ids, tab_emb, sat_emb, street_emb = pipeline.generate_embeddings()

print("Embeddings generated:")
print("Tabular  :", tab_emb.shape)
print("Satellite:", sat_emb.shape)
print("Street   :", street_emb.shape)


fusion_model = build_fusion_model(
    tab_dim=tab_emb.shape[1],
    sat_dim=sat_emb.shape[1],
    street_dim=street_emb.shape[1]
)

fusion_model.load_weights(FUSION_WEIGHTS_PATH)


log_price_pred = fusion_model.predict(
    [tab_emb, sat_emb, street_emb],
    verbose=0
).squeeze()

price_pred = np.exp(log_price_pred)


results_df = pd.DataFrame({
    "id": ids,
    "predicted_log_price": log_price_pred,
    "predicted_price": price_pred
})



if TEST_PRICE_CSV is not None:
    price_df = pd.read_csv(TEST_PRICE_CSV)
    price_df["id"] = price_df["id"].astype(str).str.strip()

    id_to_logprice = dict(
        zip(price_df["id"], price_df["log_price"])
    )

    y_log_true = np.array([id_to_logprice[i] for i in ids])
    y_true = np.exp(y_log_true)

    metrics = {
        "RMSE_log": rmse(y_log_true, log_price_pred),
        "RMSE_price": rmse(y_true, price_pred),
        "MAE_price": mean_absolute_error(y_true, price_pred),
        "R2_price": r2_score(y_true, price_pred),
    }

    print("\n📊 Test Set Evaluation")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:15s}: {v:,.4f}")

    results_df["true_price"] = y_true
    results_df["abs_error"] = np.abs(y_true - price_pred)


results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved → {OUTPUT_PATH}")
