
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "tables"


def evaluate_binary_ranking(y_true, scores, label):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    prevalence = y_true.mean()

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    k = max(1, int(np.ceil(0.10 * len(scores))))
    top_indices = np.argsort(-scores, kind="stable")[:k]

    precision_at_10 = y_true[top_indices].mean()
    recall_at_10 = (
        y_true[top_indices].sum() / y_true.sum()
    )

    return {
        "target": label,
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_10pct": precision_at_10,
        "recall_at_10pct": recall_at_10,
    }


def main():
    results = []

    for sampling in ["fixed", "random"]:
        df = pd.read_csv(
            REPORT_DIR
            / f"validation_multiclass_{sampling}.csv"
        )

        actual = df["actual_class"].to_numpy()

        # Class 4: failure within 0-6 time units.
        y_class4 = (actual == 4).astype(int)
        score_class4 = df["prob_class_4"].to_numpy()

        # Any imminent failure within 48 time units.
        y_failure = (actual > 0).astype(int)
        score_failure = df[
            [f"prob_class_{i}" for i in range(1, 5)]
        ].sum(axis=1).to_numpy()

        for target, y, score in [
            ("class_4", y_class4, score_class4),
            ("failure_within_48", y_failure, score_failure),
        ]:
            row = evaluate_binary_ranking(
                y, score, target
            )
            row["sampling"] = sampling
            results.append(row)

    results_df = pd.DataFrame(results)

    output_path = (
        REPORT_DIR / "risk_ranking_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nRISK RANKING COMPARISON")
    print("=" * 75)
    print(results_df.round(5).to_string(index=False))
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
