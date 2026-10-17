
from pathlib import Path
import gc
import json

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "tables"

VALIDATION_PATH = (
    DATA_DIR / "validation_temporal_features.parquet"
)

TRAINING_FILES = {
    "fixed": DATA_DIR / "train_temporal_features.parquet",
    "random": DATA_DIR / "train_temporal_features_random.parquet",
}

EXCLUDED_COLUMNS = [
    "vehicle_id",
    "class_label",
    "binary_label",
]

CATEGORICAL_COLUMNS = [
    f"Spec_{i}" for i in range(8)
]

CLASSES = [0, 1, 2, 3, 4]

RANDOM_STATE = 42


# ============================================================
# 2. OFFICIAL SCANIA COST MATRIX
# Rows: actual classes
# Columns: predicted classes
# ============================================================

COST_MATRIX = np.array([
    [0,   7,   8,   9,  10],
    [200, 0,   7,   8,   9],
    [300, 200, 0,   7,   8],
    [400, 300, 200, 0,   7],
    [500, 400, 300, 200, 0],
], dtype=np.float64)


def calculate_cost(y_true, y_pred):
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)

    return float(
        COST_MATRIX[actual, predicted].sum()
    )


# ============================================================
# 3. DATA PREPARATION
# ============================================================

def prepare_features(train_df, val_df):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if set(feature_columns) != (
        set(val_df.columns) - set(EXCLUDED_COLUMNS)
    ):
        raise ValueError(
            "Train and validation feature columns differ."
        )

    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()

    # Learn categorical vocabularies only from training data.
    category_mapping = {}

    for column in CATEGORICAL_COLUMNS:

        train_values = (
            X_train[column]
            .fillna("MISSING")
            .astype(str)
        )

        val_values = (
            X_val[column]
            .fillna("MISSING")
            .astype(str)
        )

        categories = sorted(train_values.unique().tolist())

        category_mapping[column] = categories

        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        X_train[column] = train_values.astype(dtype)
        X_val[column] = val_values.astype(dtype)

    # Ensure other features are numeric.
    numerical_columns = [
        column for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    X_train[numerical_columns] = X_train[
        numerical_columns
    ].astype("float32")

    X_val[numerical_columns] = X_val[
        numerical_columns
    ].astype("float32")

    return (
        X_train,
        X_val,
        feature_columns,
        category_mapping,
    )


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_val,
    validation_df,
    sampling_name,
):

    y_true = validation_df[
        "class_label"
    ].to_numpy(dtype=int)

    probabilities = model.predict_proba(X_val)

    # Reorder probabilities to official class order.
    class_positions = [
        list(model.classes_).index(c)
        for c in CLASSES
    ]

    probabilities = probabilities[:, class_positions]

    # Standard multiclass prediction.
    predicted_argmax = probabilities.argmax(axis=1)

    # Minimum expected official cost.
    # Exploratory policy: probabilities are not calibrated yet.
    expected_costs = probabilities @ COST_MATRIX

    predicted_min_cost = expected_costs.argmin(axis=1)

    argmax_cost = calculate_cost(
        y_true,
        predicted_argmax,
    )

    min_cost_policy_cost = calculate_cost(
        y_true,
        predicted_min_cost,
    )

    no_action_predictions = np.zeros_like(y_true)

    no_action_cost = calculate_cost(
        y_true,
        no_action_predictions,
    )

    summary = {
        "sampling": sampling_name,
        "accuracy": float(
            accuracy_score(y_true, predicted_argmax)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true, predicted_argmax
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predicted_argmax,
                average="macro",
                zero_division=0,
            )
        ),
        "official_cost_argmax": argmax_cost,
        "official_cost_min_expected": min_cost_policy_cost,
        "official_cost_no_action": no_action_cost,
    }

    print("\n" + "=" * 55)
    print(f"VALIDATION RESULTS — {sampling_name.upper()}")
    print("=" * 55)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nClassification report (argmax):")

    print(
        classification_report(
            y_true,
            predicted_argmax,
            labels=CLASSES,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        predicted_argmax,
        labels=CLASSES,
    )

    print("\nConfusion matrix (argmax):")
    print(cm)

    # Store predictions for reproducibility.
    predictions = pd.DataFrame({
        "vehicle_id": validation_df["vehicle_id"],
        "actual_class": y_true,
        "predicted_argmax": predicted_argmax,
        "predicted_min_cost": predicted_min_cost,
    })

    for index, class_id in enumerate(CLASSES):
        predictions[f"prob_class_{class_id}"] = (
            probabilities[:, index]
        )

    prediction_path = (
        REPORT_DIR
        / f"validation_multiclass_{sampling_name}.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    cm_path = (
        REPORT_DIR
        / f"confusion_matrix_multiclass_{sampling_name}.csv"
    )

    pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in CLASSES],
        columns=[f"predicted_{c}" for c in CLASSES],
    ).to_csv(cm_path)

    return summary


# ============================================================
# 5. TRAIN ONE MODEL
# ============================================================

def train_one_model(sampling_name, validation_df):

    train_path = TRAINING_FILES[sampling_name]

    print("\n" + "=" * 60)
    print(f"TRAINING MULTICLASS LIGHTGBM: {sampling_name.upper()}")
    print("=" * 60)

    train_df = pd.read_parquet(train_path)

    if set(train_df["class_label"].unique()) != set(CLASSES):
        raise ValueError("All five classes must be present.")

    train_ids = set(train_df["vehicle_id"])
    val_ids = set(validation_df["vehicle_id"])

    if train_ids & val_ids:
        raise ValueError(
            "Vehicle overlap between train and validation!"
        )

    X_train, X_val, feature_columns, category_mapping = (
        prepare_features(train_df, validation_df)
    )

    y_train = train_df[
        "class_label"
    ].to_numpy(dtype=int)

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    print("\nTraining class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())

    model = LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=-1,
    )

    print("\nFitting model...", flush=True)

    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
    )

    model_path = (
        MODEL_DIR
        / f"lightgbm_multiclass_{sampling_name}.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "category_mapping": category_mapping,
            "sampling": sampling_name,
        },
        model_path,
    )

    print("Saved model:", model_path)

    summary = evaluate_model(
        model,
        X_val,
        validation_df,
        sampling_name,
    )

    del X_train, X_val, train_df, model
    gc.collect()

    return summary


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading official validation dataset...")

    validation_df = pd.read_parquet(
        VALIDATION_PATH
    )

    print("Validation shape:", validation_df.shape)

    results = []

    for sampling_name in ["fixed", "random"]:

        summary = train_one_model(
            sampling_name,
            validation_df,
        )

        results.append(summary)

    results_df = pd.DataFrame(results)

    summary_path = (
        REPORT_DIR
        / "multiclass_lightgbm_comparison.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\nSaved comparison:", summary_path)
    print("Experiment completed.")


if __name__ == "__main__":
    main()
