from tensorflow.keras.callbacks import EarlyStopping

from src.Components.MultimodalFusionPipeline import MultimodalFusionPipeline
from src.Components.fusion_model import build_fusion_model


pipeline = MultimodalFusionPipeline(
    train_csv_path="Data/final_train_test_data/train_df.csv",
    price_csv_path="Data/price/price_reference_train.csv",
    tabular_encoder_path="artifacts/tabular_encoder.keras",
    tabular_scaler_path="artifacts/tabular_scaler.pkl",
    tabular_features_path="artifacts/tabular_features.pkl",
    satellite_encoder_path="artifacts/satellite_encoder.keras",
    satellite_image_dir="Data/train_images_CNN/satellite"
)

ids, tab_emb, sat_emb = pipeline.generate_embeddings()
y_log = pipeline.load_targets(ids)

model = build_fusion_model(
    tab_dim=tab_emb.shape[1],
    sat_dim=sat_emb.shape[1]
)

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    [tab_emb, sat_emb],
    y_log,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

model.save("artifacts/fusion_model.keras")
print("Fusion model trained")
