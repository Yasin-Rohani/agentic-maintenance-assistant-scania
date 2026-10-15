
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/processed/train_features.parquet"
VAL_PATH = ROOT / "data/processed/validation_features.parquet"

MODEL_PATH = ROOT / "models/catboost_baseline.joblib"
PRED_PATH = ROOT / "reports/tables/validation_predictions_catboost.csv"

ID_COL = "vehicle_id"
TRAIN_TARGET = "in_study_repair"
VAL_TARGET = "class_label"


def prepare_data():
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)

    y_train = train[TRAIN_TARGET].astype(int)
    y_val = (val[VAL_TARGET] != 0).astype(int)

    X_train = train.drop(
        columns=[
            ID_COL,
            TRAIN_TARGET,
            "length_of_study_time_step",
        ],
        errors="ignore",
    )

    X_val = val.drop(
        columns=[ID_COL, VAL_TARGET],
        errors="ignore",
    )

    # Keep identical features and their order.
    X_val = X_val.reindex(columns=X_train.columns)

    categorical_columns = [
        col for col in X_train.columns
        if col.startswith("Spec_")
    ]

    # Explicit categorical missing-value representation.
    for col in categorical_columns:
        X_train[col] = X_train[col].fillna("MISSING").astype(str)
        X_val[col] = X_val[col].fillna("MISSING").astype(str)

    # Numeric features retain NaN; CatBoost handles them.
    numeric_columns = [
        col for col in X_train.columns
        if col not in categorical_columns
    ]

    for col in numeric_columns:
        X_train[col] = pd.to_numeric(
            X_train[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        X_val[col] = pd.to_numeric(
            X_val[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Categorical columns:", categorical_columns)

    return X_train, y_train, X_val, y_val, val[ID_COL], categorical_columns


def main():
    X_train, y_train, X_val, y_val, vehicle_ids, cats = prepare_data()

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        thread_count=4,
        verbose=100,
        allow_writing_files=False,
    )

    print("\nTraining CatBoost baseline...")

    model.fit(
        X_train,
        y_train,
        cat_features=cats,
    )

    probabilities = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    print("\nCatBoost Validation Results")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    predictions_df = pd.DataFrame({
        "vehicle_id": vehicle_ids.to_numpy(),
        "true_label_binary": y_val.to_numpy(),
        "failure_probability": probabilities,
    })

    predictions_df.to_csv(PRED_PATH, index=False)

    print("\nSaved model:", MODEL_PATH)
    print("Saved predictions:", PRED_PATH)


if __name__ == "__main__":
    main()
