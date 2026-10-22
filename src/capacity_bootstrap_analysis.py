
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "tables"

N_BOOTSTRAP = 3000
RANDOM_STATE = 42

COST_MATRIX = np.array([
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
], dtype=float)

COMPARISONS = [
    ("isotonic", 30, "expected_cost_benefit", "failure_risk"),
    ("isotonic", 10, "expected_cost_benefit", "class4_risk"),
]


def make_decisions(probabilities, policy, capacity):
    expected = probabilities @ COST_MATRIX

    best_action = expected[:, 1:].argmin(axis=1) + 1

    benefit = (
        expected[:, 0]
        - expected[np.arange(len(probabilities)), best_action]
    )

    if policy == "expected_cost_benefit":
        scores = benefit
    elif policy == "failure_risk":
        scores = probabilities[:, 1:].sum(axis=1)
    elif policy == "class4_risk":
        scores = probabilities[:, 4]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    eligible = np.flatnonzero(benefit > 0)
    ranking = eligible[
        np.argsort(-scores[eligible], kind="stable")
    ]

    selected = ranking[:capacity]

    decisions = np.zeros(len(probabilities), dtype=int)
    decisions[selected] = best_action[selected]

    return decisions


def run_comparison(method, capacity_pct, policy_a, policy_b):
    df = pd.read_csv(
        REPORT_DIR / f"validation_calibration_{method}.csv"
    )

    if df["vehicle_id"].duplicated().any():
        raise ValueError("Duplicate validation vehicles.")

    actual = df["actual_class"].to_numpy(dtype=int)

    probabilities = df[
        [f"prob_class_{i}" for i in range(5)]
    ].to_numpy(dtype=float)

    n = len(df)
    capacity = int(np.floor(n * capacity_pct / 100))

    decision_a = make_decisions(
        probabilities, policy_a, capacity
    )
    decision_b = make_decisions(
        probabilities, policy_b, capacity
    )

    cost_a = COST_MATRIX[actual, decision_a]
    cost_b = COST_MATRIX[actual, decision_b]

    # Positive difference means policy A is less costly.
    paired_difference = cost_b - cost_a

    observed_total_difference = paired_difference.sum()
    observed_mean_difference = paired_difference.mean()

    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_means = np.empty(N_BOOTSTRAP)

    # Resample paired vehicle-level cost differences.
    # Predictions and selected actions remain fixed.
    for i in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)

        bootstrap_means[i] = (
            paired_difference[indices].mean()
        )

    lower, upper = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    return {
        "calibration": method,
        "capacity_pct": capacity_pct,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "cost_a": float(cost_a.sum()),
        "cost_b": float(cost_b.sum()),
        "total_cost_advantage_a": float(
            observed_total_difference
        ),
        "mean_cost_advantage_a": float(
            observed_mean_difference
        ),
        "ci95_lower_mean": float(lower),
        "ci95_upper_mean": float(upper),
        "ci_excludes_zero": bool(
            lower > 0 or upper < 0
        ),
    }


def main():
    results = [
        run_comparison(*comparison)
        for comparison in COMPARISONS
    ]

    summary = pd.DataFrame(results)

    output = REPORT_DIR / "capacity_bootstrap_comparison.csv"
    summary.to_csv(output, index=False)

    print("\nPAIRED BOOTSTRAP POLICY COMPARISON")
    print("=" * 90)
    print(summary.round(5).to_string(index=False))

    print("\nPositive advantage favors policy A.")
    print("Confidence intervals are for mean cost difference per vehicle.")
    print("These intervals do not correct for validation-based policy selection.")
    print("\nSaved:", output)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
