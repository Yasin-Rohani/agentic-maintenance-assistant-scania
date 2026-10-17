
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    average_precision_score,
    roc_auc_score,
)

from train_multiclass_lightgbm import (
    prepare_features,
    COST_MATRIX,
    calculate_cost,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "tables"

FIT_PATH = DATA_DIR / "train_random_model_fit.parquet"
CAL_PATH = DATA_DIR / "train_random_calibration.parquet"
VAL_PATH = DATA_DIR / "validation_temporal_features.parquet"

CLASSES = np.arange(5)
EPS = 1e-7


def normalize_probabilities(probabilities):
    probabilities = np.asarray(
        probabilities, dtype=np.float64
    )

    probabilities = np.clip(probabilities, EPS, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    return probabilities


def vehicle_weights(vehicle_ids):
    """
    Assign equal total calibration weight to each vehicle.

    Vehicles with more snapshots receive lower weight
    per snapshot, preventing them from dominating calibration.
    """
    ids = pd.Series(vehicle_ids).reset_index(drop=True)

    counts = ids.map(ids.value_counts())
    weights = (
        1.0 / counts
    ).to_numpy(dtype=np.float64, copy=True)

    # Normalize without modifying a read-only array.
    weights = weights / weights.mean()

    return weights

def logit_transform(probabilities):
    p = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def train_sigmoid_calibrators(probabilities, targets, weights):
    """
    Fit a separate sigmoid model for each class.
    """
    calibrators = []

    for class_id in CLASSES:
        binary_target = (targets == class_id).astype(int)

        calibrator = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=1000,
            random_state=42,
        )

        x = logit_transform(
            probabilities[:, class_id]
        ).reshape(-1, 1)

        calibrator.fit(
            x,
            binary_target,
            sample_weight=weights,
        )

        calibrators.append(calibrator)

    return calibrators


def train_isotonic_calibrators(probabilities, targets, weights):
    calibrators = []

    for class_id in CLASSES:
        binary_target = (targets == class_id).astype(int)

        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

        calibrator.fit(
            probabilities[:, class_id],
            binary_target,
            sample_weight=weights,
        )

        calibrators.append(calibrator)

    return calibrators


def apply_calibrators(probabilities, calibrators, method):
    columns = []

    for class_id, calibrator in enumerate(calibrators):
        p = probabilities[:, class_id]

        if method == "sigmoid":
            calibrated = calibrator.predict_proba(
                logit_transform(p).reshape(-1, 1)
            )[:, 1]
        elif method == "isotonic":
            calibrated = calibrator.predict(p)
        else:
            raise ValueError(f"Unknown method: {method}")

        columns.append(calibrated)

    combined = np.column_stack(columns)

    # One-vs-rest calibrated probabilities may not sum to one.
    return normalize_probabilities(combined)


def evaluate(name, y_true, probabilities):
    probabilities = normalize_probabilities(probabilities)

    one_hot = np.eye(5)[y_true]

    brier = np.mean(
        np.sum((probabilities - one_hot) ** 2, axis=1)
    )

    ll = log_loss(
        y_true,
        probabilities,
        labels=CLASSES,
    )

    argmax_prediction = probabilities.argmax(axis=1)

    expected_costs = probabilities @ COST_MATRIX
    cost_prediction = expected_costs.argmin(axis=1)

    y_failure = (y_true > 0).astype(int)
    failure_score = probabilities[:, 1:].sum(axis=1)

    y_class4 = (y_true == 4).astype(int)
    class4_score = probabilities[:, 4]

    result = {
        "method": name,
        "brier": float(brier),
        "log_loss": float(ll),
        "pr_auc_within48": float(
            average_precision_score(y_failure, failure_score)
        ),
        "roc_auc_within48": float(
            roc_auc_score(y_failure, failure_score)
        ),
        "pr_auc_class4": float(
            average_precision_score(y_class4, class4_score)
        ),
        "official_cost_argmax": calculate_cost(
            y_true, argmax_prediction
        ),
        "official_cost_min_expected": calculate_cost(
            y_true, cost_prediction
        ),
        "predicted_class4_mean": float(
            probabilities[:, 4].mean()
        ),
    }

    return result, argmax_prediction, cost_prediction


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model-fitting, calibration and validation data...")

    fit_df = pd.read_parquet(FIT_PATH)
    cal_df = pd.read_parquet(CAL_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    fit_ids = set(fit_df["vehicle_id"])
    cal_ids = set(cal_df["vehicle_id"])
    val_ids = set(val_df["vehicle_id"])

    assert not fit_ids.intersection(cal_ids)
    assert not fit_ids.intersection(val_ids)
    assert not cal_ids.intersection(val_ids)

    print("Vehicle overlap checks: PASSED")

    # Prepare validation and calibration using categories
    # learned only from model-fitting vehicles.
    X_fit, X_cal, feature_columns, category_mapping = (
        prepare_features(fit_df, cal_df)
    )

    _, X_val, _, _ = prepare_features(fit_df, val_df)

    y_fit = fit_df["class_label"].to_numpy(dtype=int)
    y_cal = cal_df["class_label"].to_numpy(dtype=int)
    y_val = val_df["class_label"].to_numpy(dtype=int)

    print("Model fitting:", X_fit.shape)
    print("Calibration:", X_cal.shape)
    print("Validation:", X_val.shape)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )

    print("\nTraining LightGBM...")
    model.fit(
        X_fit,
        y_fit,
        categorical_feature=[
            f"Spec_{i}" for i in range(8)
        ],
    )

    cal_probs = normalize_probabilities(
        model.predict_proba(X_cal)
    )

    val_probs = normalize_probabilities(
        model.predict_proba(X_val)
    )

    weights = vehicle_weights(cal_df["vehicle_id"])

    print("\nFitting sigmoid calibrators...")
    sigmoid = train_sigmoid_calibrators(
        cal_probs, y_cal, weights
    )

    print("Fitting isotonic calibrators...")
    isotonic = train_isotonic_calibrators(
        cal_probs, y_cal, weights
    )

    methods = {
        "uncalibrated": val_probs,
        "sigmoid": apply_calibrators(
            val_probs, sigmoid, "sigmoid"
        ),
        "isotonic": apply_calibrators(
            val_probs, isotonic, "isotonic"
        ),
    }

    results = []

    for method_name, probabilities in methods.items():
        metrics, argmax_pred, cost_pred = evaluate(
            method_name, y_val, probabilities
        )

        results.append(metrics)

        predictions = pd.DataFrame({
            "vehicle_id": val_df["vehicle_id"].to_numpy(),
            "actual_class": y_val,
            "predicted_argmax": argmax_pred,
            "predicted_min_cost": cost_pred,
        })

        for class_id in CLASSES:
            predictions[f"prob_class_{class_id}"] = (
                probabilities[:, class_id]
            )

        output = (
            REPORT_DIR
            / f"validation_calibration_{method_name}.csv"
        )

        predictions.to_csv(output, index=False)

    summary = pd.DataFrame(results)

    summary_path = (
        REPORT_DIR / "multiclass_calibration_comparison.csv"
    )

    summary.to_csv(summary_path, index=False)

    bundle_path = (
        MODEL_DIR / "lightgbm_multiclass_calibration_experiment.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "sigmoid_calibrators": sigmoid,
            "isotonic_calibrators": isotonic,
            "feature_columns": feature_columns,
            "category_mapping": category_mapping,
        },
        bundle_path,
    )

    print("\n" + "=" * 80)
    print("MULTICLASS CALIBRATION COMPARISON")
    print("=" * 80)

    print(
        summary.round(6).to_string(index=False)
    )

    print("\nSaved summary:", summary_path)
    print("Saved model and calibrators:", bundle_path)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
