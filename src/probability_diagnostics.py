
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = np.arange(5)


def evaluate_probabilities(name):
    path = (
        REPORT_DIR
        / f"validation_multiclass_{name}.csv"
    )

    df = pd.read_csv(path)

    y_true = df["actual_class"].to_numpy(dtype=int)

    probability_columns = [
        f"prob_class_{i}" for i in CLASSES
    ]

    probabilities = df[
        probability_columns
    ].to_numpy(dtype=float)

    # Normalize for numerical stability.
    probabilities = np.clip(
        probabilities, 1e-15, 1.0
    )

    probabilities /= probabilities.sum(
        axis=1, keepdims=True
    )

    # One-hot representation of actual classes.
    y_onehot = np.eye(5)[y_true]

    multiclass_brier = np.mean(
        np.sum(
            (probabilities - y_onehot) ** 2,
            axis=1,
        )
    )

    multiclass_logloss = log_loss(
        y_true,
        probabilities,
        labels=CLASSES,
    )

    results = {
        "sampling": name,
        "multiclass_brier": multiclass_brier,
        "multiclass_logloss": multiclass_logloss,
    }

    print("\n" + "=" * 55)
    print(f"PROBABILITY DIAGNOSTICS: {name.upper()}")
    print("=" * 55)

    print(
        "Multiclass Brier Score:",
        round(multiclass_brier, 6),
    )

    print(
        "Multiclass Log Loss:",
        round(multiclass_logloss, 6),
    )

    print("\nObserved vs Predicted Class Rates:")

    for class_id in CLASSES:
        observed = np.mean(
            y_true == class_id
        )

        predicted = np.mean(
            probabilities[:, class_id]
        )

        print(
            f"Class {class_id}: "
            f"Observed={observed:.5f}, "
            f"Predicted={predicted:.5f}"
        )

    return results


def main():
    results = []

    for name in ["fixed", "random"]:
        results.append(
            evaluate_probabilities(name)
        )

    comparison = pd.DataFrame(results)

    output_path = (
        REPORT_DIR
        / "probability_diagnostics_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 55)
    print("FINAL PROBABILITY COMPARISON")
    print("=" * 55)

    print(
        comparison.to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
