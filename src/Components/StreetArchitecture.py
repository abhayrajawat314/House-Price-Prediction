import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization,GlobalAveragePooling2D,Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class StreetCNNEncoder:
    def __init__(
        self,
        image_dir,
        id_source_csv,
        price_csv_path,
        missing_ids_csv,
        id_col="id",
        target_col="log_price",
        embedding_dim=64,
        image_size=(224, 224)
    ):
        self.image_dir = image_dir
        self.id_source_csv = id_source_csv
        self.price_csv_path = price_csv_path
        self.missing_ids_csv = missing_ids_csv
        self.id_col = id_col
        self.target_col = target_col
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        self.encoder = None
        self.valid_ids = self._get_valid_ids()

    def _get_valid_ids(self):
        all_ids = pd.read_csv(self.id_source_csv)[self.id_col].astype(str).str.strip()
        missing = set(pd.read_csv(self.missing_ids_csv).iloc[:, 0].astype(str).str.strip())
        return [
            i for i in all_ids
            if i not in missing and os.path.exists(os.path.join(self.image_dir, f"{i}.jpg"))
        ]

    def _data_generator(self):
        price_df = pd.read_csv(self.price_csv_path)
        id_to_y = dict(zip(price_df[self.id_col].astype(str), price_df[self.target_col]))

        for i in self.valid_ids:
            img = load_img(os.path.join(self.image_dir, f"{i}.jpg"), target_size=self.image_size)
            img = img_to_array(img) / 255.0
            yield img, id_to_y[i]

    def build_encoder(self):
        self.encoder = Sequential([

        Conv2D(32, 3, padding="same", activation="relu",
               input_shape=(*self.image_size, 3)),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(48, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(96, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),

        GlobalAveragePooling2D(),

        Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        BatchNormalization(),
        Dropout(0.2),

        Dense(self.embedding_dim, activation="relu")
    ])

        return self.encoder


    def train_and_save(self, encoder_path, epochs=70, batch_size=16):
        self.build_encoder()
        model = Sequential([self.encoder, Dense(1)])
        model.compile(optimizer=Adam(1e-4), loss="mse")

        ds = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        ).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        model.fit(ds, epochs=epochs, steps_per_epoch=len(self.valid_ids)//batch_size,callbacks=early_stop)
        self.encoder.save(encoder_path)

    def generate_embeddings(self, model):
        df = pd.read_csv(self.id_source_csv)
        ids = df[self.id_col].astype(str).str.strip().values

        embeddings = []
        for i in ids:
            path = os.path.join(self.image_dir, f"{i}.jpg")
            if not os.path.exists(path):
                embeddings.append(np.zeros(self.embedding_dim))
            else:
                img = img_to_array(load_img(path, target_size=self.image_size)) / 255.0
                embeddings.append(model.predict(img[None, ...], verbose=0)[0])

        return ids, np.vstack(embeddings)
