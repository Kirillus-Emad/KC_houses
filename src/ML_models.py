import os
import json
import time

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config import (
    DF_TRAIN_SCALE,
    DF_TEST_SCALE,
    DF_TRAIN_TRANS_SCALE,
    DF_TEST_TRANS_SCALE,
    TARGET_COLUMN,
    RESULTS_DIR
)


DATASETS = {
    "1": ("scale", DF_TRAIN_SCALE, DF_TEST_SCALE),
    "2": ("trans_scale", DF_TRAIN_TRANS_SCALE, DF_TEST_TRANS_SCALE),
}

MODELS = {
    "1": ("linear_regression", lambda: LinearRegression()),
    "2": ("ridge", lambda: Ridge()),
    "3": ("lasso", lambda: Lasso()),
    # "4": ("polynomial_regression", lambda: make_pipeline(PolynomialFeatures(degree=2), Lasso())),
    "5": ("decision_tree", lambda: DecisionTreeRegressor(random_state=42,max_depth=6)),
    "6": ("random_forest", lambda: RandomForestRegressor(random_state=42,max_depth=6,n_jobs=-1)),
    "7": ("xgboost", lambda: XGBRegressor(random_state=42,n_jobs=-1,max_depth=2)),
}


def load_dataset(dataset_key):
    dataset_name, train_path, test_path = DATASETS[dataset_key]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return dataset_name, X_train, y_train, X_test, y_test


def score(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_evaluate_save(model_key, dataset_name, X_train, y_train, X_test, y_test):
    model_name, model_builder = MODELS[model_key]
    model = model_builder()

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    train_pred = model.predict(X_train)
    train_predict_time = time.time() - t0

    t0 = time.time()
    test_pred = model.predict(X_test)
    test_time = time.time() - t0

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "train_time_sec": train_time,
        "train_predict_time_sec": train_predict_time,
        "test_time_sec": test_time,
        "train": score(y_train, train_pred),
        "test": score(y_test, test_pred),
    }

    model_dir = os.path.join(RESULTS_DIR, dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def save_comparison(dataset_name, all_metrics):
    rows = []
    for m in all_metrics:
        rows.append({
            "model": m["model"],
            "train_MSE": m["train"]["MSE"],
            "train_MAE": m["train"]["MAE"],
            "train_R2": m["train"]["R2"],
            "test_MSE": m["test"]["MSE"],
            "test_MAE": m["test"]["MAE"],
            "test_R2": m["test"]["R2"],
            "train_time_sec": m["train_time_sec"],
            "test_time_sec": m["test_time_sec"],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(RESULTS_DIR, dataset_name, "comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    return comparison_df


def run(dataset_key, model_keys):
    dataset_name, X_train, y_train, X_test, y_test = load_dataset(dataset_key)

    all_metrics = []
    for model_key in model_keys:
        model_name = MODELS[model_key][0]
        print(f"Training {model_name} on {dataset_name} ...")

        metrics = train_evaluate_save(model_key, dataset_name, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)

        print(
            f"  train R2={metrics['train']['R2']:.4f}  "
            f"test R2={metrics['test']['R2']:.4f}  "
            f"test MSE={metrics['test']['MSE']:.4f}  "
            f"test MAE={metrics['test']['MAE']:.4f}"
        )

    comparison_df = save_comparison(dataset_name, all_metrics)
    print("\nComparison:")
    print(comparison_df.to_string(index=False))

    return all_metrics


def prompt_dataset():
    print("Select dataset:")
    for key, (name, _, _) in DATASETS.items():
        print(f"  {key}. {name}")

    choice = input("Dataset number: ").strip()
    if choice not in DATASETS:
        raise ValueError(f"Invalid dataset choice: {choice}")

    return choice


def prompt_models():
    print("\nSelect model(s):")
    print("  0. All models")
    for key, (name, _) in MODELS.items():
        print(f"  {key}. {name}")

    choice = input("Model number(s) (comma separated, or 0 for all): ").strip()

    if choice == "0":
        return list(MODELS.keys())

    model_keys = [c.strip() for c in choice.split(",") if c.strip()]
    for key in model_keys:
        if key not in MODELS:
            raise ValueError(f"Invalid model choice: {key}")

    return model_keys


if __name__ == "__main__":
    dataset_key = prompt_dataset()
    model_keys = prompt_models()

    run(dataset_key, model_keys)
