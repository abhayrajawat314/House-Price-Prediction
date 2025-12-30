import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class TabularEncoder:
    def __init__(
        self,
        train_csv_path,
        id_col="id",
        target_col="log_price",
        drop_cols=("price", "log_price"),
        embedding_dim=64
    ):
        """
        train_csv_path : Data/final_train_test_data/train_df.csv
        id_col         : id (alignment only)
        target_col     : training target (log_price)
        drop_cols      : columns to remove from features
        embedding_dim  : output embedding size
        """
        self.train_csv_path = train_csv_path
        self.id_col = id_col
        self.target_col = target_col
        self.drop_cols = set(drop_cols)
        self.embedding_dim = embedding_dim

        self.scaler = StandardScaler()
        self.feature_cols = None

        self.encoder = None
        self.training_model = None

 
    def load_data(self):
        df = pd.read_csv(self.train_csv_path)

        df[self.id_col] = df[self.id_col].astype(str).str.strip()

        cols_to_drop = {self.id_col} | self.drop_cols
        cols_to_drop = cols_to_drop.intersection(df.columns)

        self.feature_cols = [c for c in df.columns if c not in cols_to_drop]

        X = df[self.feature_cols].values
        y = df[self.target_col].values

        return X, y


    def build_encoder(self, input_dim):
        self.encoder = Sequential([
            Dense(256, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),

            Dense(self.embedding_dim, activation="relu", name="tab_embedding")
        ])
        return self.encoder


    def build_training_model(self, input_dim):
        self.build_encoder(input_dim)

        self.training_model = Sequential([
            self.encoder,
            Dense(1, name="log_price")
        ])

        self.training_model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="mse"
        )

        return self.training_model


    def train_and_save(
        self,
        encoder_path,
        scaler_path,
        feature_path,
        epochs=100,
        batch_size=32,
        validation_split=0.2
    ):
        """
        Trains tabular embeddings and saves:
        - encoder model
        - scaler
        - feature column list
        """

        
        X, y = self.load_data()

        
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, scaler_path)

       
        joblib.dump(self.feature_cols, feature_path)

       
        model = self.build_training_model(input_dim=X_scaled.shape[1])

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=14,
            restore_best_weights=True,
            verbose=1
        )
        model.fit(
            X_scaled,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=early_stop,
            verbose=1
        )

        
        self.encoder.save(encoder_path)

        print("Tabular encoder trained and saved")
