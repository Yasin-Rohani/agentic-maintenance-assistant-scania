
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models/"
    "lightgbm_multiclass_calibration_experiment.joblib"
)

FIT_PATH = (
    ROOT / "data/processed/train_random_model_fit.parquet"
)

VAL_PATH = (
    ROOT / "data/processed/validation_temporal_features.parquet"
)

OUTPUT_PATH = (
    ROOT / "reports/tables/model_drift_sensitivity.csv"
)

FEATURE = "397_31_recent_std"
SHIFTS = [0.0, 0.25, 0.50, 1.0]

COST_MATRIX = np.array([
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
], dtype=float)

EPS = 1e-7


def normalize(probs):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, EPS, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def calibrate(probs, calibrators):
    calibrated = []

    for i, calibrator in enumerate(calibrators):
        calibrated.append(
            calibrator.predict_proba(
                logit(probs[:, i]).reshape(-1, 1)
            )[:, 1]
        )

    return normalize(np.column_stack(calibrated))


def prepare_features(df, bundle):
     X = df[bundle["feature_columns"]].copy()

     for col, categories in bundle["category_mapping"].items():
        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        values = X[col].fillna("MISSING").astype(str)

        # Handle categories not seen during training.
        values = values.where(
            values.isin(categories),
            other=None,
        )

        X[col] = pd.Series(
            pd.Categorical(values, dtype=dtype),
            index=X.index,
        )

     numeric_cols = X.select_dtypes(
        exclude=["category"]
     ).columns

     X[numeric_cols] = X[numeric_cols].astype("float32")

     return X

def evaluate(y, probs):
    probs = normalize(probs)

    expected_cost = probs @ COST_MATRIX

    decisions = expected_cost.argmin(axis=1)

    cost = COST_MATRIX[y, decisions].sum()

    binary_target = (y > 0).astype(int)
    binary_scores = probs[:, 1:].sum(axis=1)

    brier = np.mean(
        np.sum((probs - np.eye(5)[y]) ** 2, axis=1)
    )

    return {
        "brier": float(brier),
        "log_loss": float(
            log_loss(y, probs, labels=np.arange(5))
        ),
        "pr_auc_within48": float(
            average_precision_score(
                binary_target, binary_scores
            )
        ),
        "official_cost": float(cost),
        "vehicles_flagged": int((decisions > 0).sum()),
    }, decisions


def main():
    print("Loading model and validation data...")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    sigmoid = bundle["sigmoid_calibrators"]

    validation = pd.read_parquet(VAL_PATH)

    reference = pd.read_parquet(
        FIT_PATH,
        columns=["vehicle_id", FEATURE],
    )

    reference = (
        reference.groupby("vehicle_id", group_keys=False)
        .sample(n=1, random_state=42)
    )

    ref_values = reference[FEATURE].to_numpy(dtype=float)
    ref_values = ref_values[np.isfinite(ref_values)]

    if len(ref_values) < 100:
        raise ValueError("Insufficient reference observations.")

    scale = float(np.std(ref_values))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid reference standard deviation.")

    y = validation["class_label"].to_numpy(dtype=int)

    results = []
    baseline_decisions = None
    baseline_probs = None

    for shift in SHIFTS:
        modified = validation.copy()

        values = modified[FEATURE].to_numpy(
            dtype=float, copy=True
        )

        finite = np.isfinite(values)
        values[finite] += shift * scale

        modified[FEATURE] = values

        X = prepare_features(modified, bundle)

        raw_probs = normalize(model.predict_proba(X))
        probs = calibrate(raw_probs, sigmoid)

        metrics, decisions = evaluate(y, probs)

        if shift == 0.0:
            baseline_decisions = decisions.copy()
            baseline_probs = probs.copy()

        metrics.update({
            "shift_std": shift,
            "decision_change_rate": float(
                np.mean(decisions != baseline_decisions)
            ),
            "mean_absolute_probability_change": float(
                np.mean(np.abs(probs - baseline_probs))
            ),
        })

        results.append(metrics)

        print(
            f"Shift={shift:.2f} SD | "
            f"Cost={metrics['official_cost']:.0f} | "
            f"PR-AUC={metrics['pr_auc_within48']:.5f} | "
            f"Decision changes="
            f"{metrics['decision_change_rate']:.2%}",
            flush=True,
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 75)
    print("MODEL DRIFT SENSITIVITY COMPARISON")
    print("=" * 75)

    print(results_df.round(6).to_string(index=False))

    print("\nSaved:", OUTPUT_PATH)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
