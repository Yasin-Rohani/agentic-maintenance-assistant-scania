
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "tables"

METHODS = ["uncalibrated", "sigmoid", "isotonic"]

COST_MATRIX = np.array([
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
])


def main():
    results = []

    for method in METHODS:
        path = (
            REPORT_DIR
            / f"validation_calibration_{method}.csv"
        )

        df = pd.read_csv(path)

        actual = df["actual_class"].to_numpy(dtype=int)
        predicted = df["predicted_min_cost"].to_numpy(
            dtype=int
        )

        cm = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1, 2, 3, 4],
        )

        class0 = actual == 0
        positive = actual > 0

        false_alerts = int(
            np.sum(class0 & (predicted > 0))
        )

        detected_positive = int(
            np.sum(positive & (predicted > 0))
        )

        missed_positive = int(
            np.sum(positive & (predicted == 0))
        )

        total_cost = float(
            COST_MATRIX[actual, predicted].sum()
        )

        results.append({
            "method": method,
            "total_cost": total_cost,
            "vehicles_flagged": int(
                np.sum(predicted > 0)
            ),
            "false_alerts": false_alerts,
            "detected_positive": detected_positive,
            "missed_positive": missed_positive,
            "binary_recall": (
                detected_positive / max(1, positive.sum())
            ),
            "false_alert_rate": (
                false_alerts / max(1, class0.sum())
            ),
        })

        print("\n" + "=" * 60)
        print(method.upper())
        print("=" * 60)

        print("\nPredicted class distribution:")
        print(
            pd.Series(predicted)
            .value_counts()
            .sort_index()
        )

        print("\nCost-sensitive confusion matrix:")
        print(cm)

    summary = pd.DataFrame(results)

    output_path = (
        REPORT_DIR / "calibration_decision_audit.csv"
    )

    summary.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("CALIBRATION DECISION AUDIT")
    print("=" * 80)

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

