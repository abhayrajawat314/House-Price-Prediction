from src.Components.MultimodalFusionPipeline import MultimodalFusionPipeline
from src.Components.fusion_model import build_fusion_model
from tensorflow.keras.callbacks import EarlyStopping


pipeline = MultimodalFusionPipeline(
    "Data/final_train_test_data/train_df.csv",
    "Data/price/price_reference_train.csv",
    "artifacts/tabular_encoder.keras",
    "artifacts/tabular_scaler.pkl",
    "artifacts/tabular_features.pkl",
    "artifacts/satellite_encoder.keras",
    "artifacts/street_encoder.keras",
    "Data/train_images_CNN/satellite",
    "Data/train_images_CNN/street",
    "Data/train_images_CNN/missing_street_ids.csv"
)


ids, tab_emb, sat_emb, street_emb = pipeline.generate_embeddings()
y_log = pipeline.load_targets(ids)


model = build_fusion_model(
    tab_emb.shape[1],
    sat_emb.shape[1],
    street_emb.shape[1]
)

model.compile(
    optimizer="adam",
    loss="mse"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,                 # not too aggressive
    restore_best_weights=True,
    verbose=1
)

model.fit(
    [tab_emb, sat_emb, street_emb],
    y_log,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

model.save("artifacts/fusion_model3.keras")

print("Fusion model trained with early stopping")
