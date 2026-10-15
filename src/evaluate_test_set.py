
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models/lightgbm_baseline.joblib"
TEST_PATH = PROJECT_ROOT / "data/processed/test_features.parquet"
SCENARIO_PATH = (
    PROJECT_ROOT
    / "reports/tables/cost_scenario_summary_lightgbm.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports/tables"


def load_test_data():
    print("Loading test features...")

    df = pd.read_parquet(TEST_PATH)

    # Match the binary evaluation used on validation.
    # Confirm the interpretation of class_label with
    # the SCANIA documentation before final publication.
    y_test = (df["class_label"] != 0).astype(int)

    X_test = df.drop(
        columns=["vehicle_id", "class_label"],
        errors="ignore",
    )

    X_test = X_test.select_dtypes(include=["number"])

    return df["vehicle_id"], X_test, y_test


def align_features(model, X_test):
    # Use exactly the features seen during training.
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model has no recorded feature names. "
            "Verify the training feature schema."
        )

    train_columns = list(model.feature_names_in_)

    missing = set(train_columns) - set(X_test.columns)

    if missing:
        print(f"Warning: {len(missing)} expected features are missing.")

    # Missing features must remain NaN for the imputer.
    # Do not silently treat them as observed zero values.
    return X_test.reindex(columns=train_columns)


def evaluate_predictions(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)

    vehicle_ids, X_test, y_test = load_test_data()

    X_test = align_features(model, X_test)

    print("Test feature shape:", X_test.shape)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\nTEST PERFORMANCE")
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    default_metrics = evaluate_predictions(
        y_test, probabilities, 0.50
    )

    print("\nConfusion matrix at threshold 0.50:")
    print(
        [
            [default_metrics["tn"], default_metrics["fp"]],
            [default_metrics["fn"], default_metrics["tp"]],
        ]
    )

    print(
        classification_report(
            y_test,
            (probabilities >= 0.50).astype(int),
            zero_division=0,
        )
    )

    scenarios = pd.read_csv(SCENARIO_PATH)

    results = []

    for _, scenario in scenarios.iterrows():
        # Reuse validation-selected thresholds.
        # Never optimize thresholds on the test labels.
        threshold = float(scenario["best_cost_threshold"])

        metrics = evaluate_predictions(
            y_test, probabilities, threshold
        )

        metrics["scenario"] = scenario["scenario"]
        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\nSCENARIO EVALUATION ON TEST")
    print(results_df.to_string(index=False))

    predictions_df = pd.DataFrame({
        "vehicle_id": vehicle_ids,
        "true_label_binary": y_test,
        "failure_probability": probabilities,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions_lightgbm.csv",
        index=False,
    )

    results_df.to_csv(
        OUTPUT_DIR / "test_scenario_metrics.csv",
        index=False,
    )

    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()
