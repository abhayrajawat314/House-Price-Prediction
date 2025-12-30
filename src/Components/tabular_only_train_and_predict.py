import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


TRAIN_CSV = "Data/final_train_test_data/train_df.csv"
TEST_CSV  = "Data/final_train_test_data/test_df.csv"

MODEL_PATH   = "artifacts/tabular_only_model.keras"
SCALER_PATH  = "artifacts/tabular_only_scaler.pkl"
FEATURE_PATH = "artifacts/tabular_only_features.pkl"
OUTPUT_PATH  = "artifacts/tabular_only_test_predictions.csv"


train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

for df in [train_df, test_df]:
    df["id"] = df["id"].astype(str).str.strip()


DROP_COLS = ["id", "price", "log_price"]
feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

X_train = train_df[feature_cols].values
y_train = train_df["log_price"].values
X_test  = test_df[feature_cols].values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)
joblib.dump(feature_cols, FEATURE_PATH)



model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(1, name="log_price")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)



model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save(MODEL_PATH)
print("✅ Tabular-only model trained & saved")

log_price_pred = model.predict(X_train, verbose=0).squeeze()
price_pred = np.exp(log_price_pred)



results_df = pd.DataFrame({
    "id": train_df["id"],
    "predicted_log_price": log_price_pred,
    "predicted_price": price_pred
})

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved → {OUTPUT_PATH}")
