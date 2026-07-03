import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, callbacks

from config import DF_TRAIN_ANN_SCALE, DF_TEST_ANN_SCALE, TARGET_COLUMN, RESULTS_DIR

MODEL_NAME = "ann"
DATASET_NAME = "scale"


def load_dataset():
    train_df = pd.read_csv(DF_TRAIN_ANN_SCALE)
    test_df = pd.read_csv(DF_TEST_ANN_SCALE)

    X_train = train_df.drop(columns=[TARGET_COLUMN]).values
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df.drop(columns=[TARGET_COLUMN]).values
    y_test = test_df[TARGET_COLUMN].values

    return X_train, y_train, X_test, y_test


def score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"MSE": float(mse), "MAE": float(mae), "R2": float(r2)}


class R2Logger(callbacks.Callback):
    """Prints train R2 and test R2 at the end of every epoch."""

    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict(self.X_train, verbose=0).flatten()
        test_pred = self.model.predict(self.X_test, verbose=0).flatten()

        train_r2 = score(self.y_train, train_pred)["R2"]
        test_r2 = score(self.y_test, test_pred)["R2"]

        print(f"  epoch {epoch + 1}: train_R2={train_r2:.4f}  test_R2={test_r2:.4f}")


def build_model(input_dim, hidden_layers=(64, 32), dropout=0.0, learning_rate=0.001):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_evaluate_save(
    dataset_name,
    X_train,
    y_train,
    X_test,
    y_test,
    hidden_layers=(1024, 256, 64, 32),
    dropout=0.2,
    learning_rate=0.001,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    patience=10,
):
    model = build_model(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    r2_logger = R2Logger(X_train, y_train, X_test, y_test)

    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, r2_logger],
        verbose=1,
    )
    train_time = time.time() - t0

    t0 = time.time()
    train_pred = model.predict(X_train, verbose=0).flatten()
    train_predict_time = time.time() - t0

    t0 = time.time()
    test_pred = model.predict(X_test, verbose=0).flatten()
    test_time = time.time() - t0

    metrics = {
        "model": MODEL_NAME,
        "dataset": dataset_name,
        "epochs_ran": len(history.history["loss"]),
        "train_time_sec": train_time,
        "train_predict_time_sec": train_predict_time,
        "test_time_sec": test_time,
        "train": score(y_train, train_pred),
        "test": score(y_test, test_pred),
    }

    model_dir = os.path.join(RESULTS_DIR, dataset_name, MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(model_dir, "model.keras"))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dir, "history.csv"), index=False)

    plt.figure()
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"ANN training loss ({dataset_name})")
    plt.legend()
    plt.savefig(os.path.join(model_dir, "loss_curve.png"))
    plt.close()

    return metrics


def run(**train_kwargs):
    X_train, y_train, X_test, y_test = load_dataset()

    print(f"Training {MODEL_NAME} on {DATASET_NAME} ...")
    metrics = train_evaluate_save(DATASET_NAME, X_train, y_train, X_test, y_test, **train_kwargs)

    print(
        f"  train R2={metrics['train']['R2']:.4f}  "
        f"test R2={metrics['test']['R2']:.4f}  "
        f"test MSE={metrics['test']['MSE']:.4f}  "
        f"test MAE={metrics['test']['MAE']:.4f}"
    )

    return metrics


if __name__ == "__main__":
    run()
