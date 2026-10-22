
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_constrained_decision import COST_MATRIX


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "tables"

METHODS = ["uncalibrated", "sigmoid", "isotonic"]
CAPACITIES = [0.05, 0.10, 0.20, 0.30]


def select_by_score(scores, eligible, capacity):
    candidates = np.flatnonzero(eligible)
    ordered = candidates[
        np.argsort(-scores[candidates], kind="stable")
    ]
    return ordered[:capacity]


def evaluate(actual, decisions):
    flagged = decisions > 0
    positive = actual > 0

    true_alerts = np.sum(flagged & positive)
    false_alerts = np.sum(flagged & ~positive)

    return {
        "flagged": int(flagged.sum()),
        "true_alerts": int(true_alerts),
        "false_alerts": int(false_alerts),
        "recall": float(
            true_alerts / positive.sum()
        ),
        "precision": float(
            true_alerts / max(1, flagged.sum())
        ),
        "official_cost": float(
            COST_MATRIX[actual, decisions].sum()
        ),
    }


def main():
    rows = []

    for method in METHODS:
        df = pd.read_csv(
            REPORT_DIR
            / f"validation_calibration_{method}.csv"
        )

        actual = df["actual_class"].to_numpy(dtype=int)

        probs = df[
            [f"prob_class_{i}" for i in range(5)]
        ].to_numpy(dtype=float)

        expected_cost = probs @ COST_MATRIX

        best_action = (
            expected_cost[:, 1:].argmin(axis=1) + 1
        )

        benefit = (
            expected_cost[:, 0]
            - expected_cost[
                np.arange(len(df)), best_action
            ]
        )

        eligible = benefit > 0

        scores = {
            "expected_cost_benefit": benefit,
            "failure_risk": probs[:, 1:].sum(axis=1),
            "class4_risk": probs[:, 4],
        }

        for fraction in CAPACITIES:
            capacity = int(np.floor(len(df) * fraction))

            for policy, score in scores.items():
                selected = select_by_score(
                    score, eligible, capacity
                )

                decisions = np.zeros(
                    len(df), dtype=int
                )

                decisions[selected] = best_action[selected]

                result = evaluate(actual, decisions)

                rows.append({
                    "calibration": method,
                    "policy": policy,
                    "capacity_pct": fraction * 100,
                    **result,
                })

    comparison = pd.DataFrame(rows)

    output = (
        REPORT_DIR / "capacity_policy_benchmark.csv"
    )

    comparison.to_csv(output, index=False)

    print("\nCAPACITY POLICY BENCHMARK")
    print("=" * 90)

    print(
        comparison[
            [
                "calibration",
                "policy",
                "capacity_pct",
                "flagged",
                "recall",
                "precision",
                "official_cost",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved:", output)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
