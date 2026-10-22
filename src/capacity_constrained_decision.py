
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["uncalibrated", "sigmoid", "isotonic"]

CAPACITY_FRACTIONS = [0.05, 0.10, 0.20, 0.30]

CLASSES = np.arange(5)

# Rows = actual classes
# Columns = predicted classes / decisions

COST_MATRIX = np.array([
    [0,   7,   8,   9,  10],
    [200, 0,   7,   8,   9],
    [300, 200, 0,   7,   8],
    [400, 300, 200, 0,   7],
    [500, 400, 300, 200, 0],
], dtype=np.float64)


# ============================================================
# 2. CAPACITY-CONSTRAINED OPTIMIZATION
# ============================================================

def optimize_decisions(probabilities, capacity):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError(
            "Probabilities must have shape (N, 5)."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities contain invalid values."
        )

    if (probabilities < 0).any():
        raise ValueError(
            "Probabilities cannot be negative."
        )

    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError(
            "Probability rows must sum to one."
        )

    n_vehicles = len(probabilities)

    if not 0 <= capacity <= n_vehicles:
        raise ValueError("Invalid inspection capacity.")

    # Expected cost of every possible decision.
    expected_costs = probabilities @ COST_MATRIX

    # Expected cost if every vehicle receives action 0.
    no_action_cost = expected_costs[:, 0]

    # Best action among classes 1, 2, 3, and 4.
    best_nonzero_actions = (
        expected_costs[:, 1:].argmin(axis=1) + 1
    )

    best_nonzero_cost = expected_costs[
        np.arange(n_vehicles),
        best_nonzero_actions,
    ]

    # Predicted benefit of flagging each vehicle.
    expected_benefit = (
        no_action_cost - best_nonzero_cost
    )

    # Start with no inspections.
    decisions = np.zeros(
        n_vehicles,
        dtype=np.int8,
    )

    # Only positive-benefit inspections are eligible.
    eligible = np.flatnonzero(
        expected_benefit > 0
    )

    # Select vehicles by expected benefit, descending.
    # Stable sorting makes equal-benefit ties deterministic.
    ranking = eligible[
        np.argsort(
            -expected_benefit[eligible],
            kind="stable",
        )
    ]

    selected = ranking[:capacity]

    decisions[selected] = best_nonzero_actions[
        selected
    ]

    return decisions, expected_benefit


# ============================================================
# 3. EVALUATION
# ============================================================

def evaluate_policy(
    method,
    capacity_fraction,
    actual,
    predictions,
    expected_benefit,
):
    n_vehicles = len(actual)

    positive = actual > 0
    negative = actual == 0

    flagged = predictions > 0

    true_alerts = int(
        np.sum(positive & flagged)
    )

    false_alerts = int(
        np.sum(negative & flagged)
    )

    missed_positives = int(
        np.sum(positive & ~flagged)
    )

    vehicles_flagged = int(
        flagged.sum()
    )

    total_positive = int(
        positive.sum()
    )

    total_negative = int(
        negative.sum()
    )

    total_cost = float(
        COST_MATRIX[actual, predictions].sum()
    )

    no_action_cost = float(
        COST_MATRIX[
            actual,
            np.zeros(n_vehicles, dtype=int),
        ].sum()
    )

    return {
        "method": method,
        "capacity_pct": capacity_fraction * 100,
        "capacity_limit": int(
            np.floor(n_vehicles * capacity_fraction)
        ),
        "vehicles_flagged": vehicles_flagged,
        "true_alerts": true_alerts,
        "false_alerts": false_alerts,
        "missed_positives": missed_positives,
        "binary_precision": (
            true_alerts / vehicles_flagged
            if vehicles_flagged else 0.0
        ),
        "binary_recall": (
            true_alerts / total_positive
            if total_positive else 0.0
        ),
        "false_alert_rate": (
            false_alerts / total_negative
            if total_negative else 0.0
        ),
        "official_cost": total_cost,
        "no_action_cost": no_action_cost,
        "cost_reduction_pct": (
            100 * (no_action_cost - total_cost)
            / no_action_cost
            if no_action_cost else 0.0
        ),
        "total_selected_expected_benefit": float(
            expected_benefit[flagged].sum()
        ),
    }


# ============================================================
# 4. MAIN EXPERIMENT
# ============================================================

def main():

    results = []

    for method in METHODS:

        print("\n" + "=" * 65)
        print(f"METHOD: {method.upper()}")
        print("=" * 65)

        prediction_path = (
            REPORT_DIR
            / f"validation_calibration_{method}.csv"
        )

        df = pd.read_csv(prediction_path)

        if df["vehicle_id"].duplicated().any():
            raise ValueError(
                "Duplicate validation vehicles found."
            )

        actual = df[
            "actual_class"
        ].to_numpy(dtype=int)

        probabilities = df[
            [f"prob_class_{i}" for i in CLASSES]
        ].to_numpy(dtype=float)

        for fraction in CAPACITY_FRACTIONS:

            capacity = int(
                np.floor(len(df) * fraction)
            )

            decisions, benefits = optimize_decisions(
                probabilities,
                capacity,
            )

            summary = evaluate_policy(
                method,
                fraction,
                actual,
                decisions,
                benefits,
            )

            results.append(summary)

            print(
                f"\nCapacity: {fraction * 100:.0f}%"
            )
            print(
                "Vehicles flagged:",
                summary["vehicles_flagged"],
            )
            print(
                "True alerts:",
                summary["true_alerts"],
            )
            print(
                "False alerts:",
                summary["false_alerts"],
            )
            print(
                "Binary recall:",
                round(summary["binary_recall"], 4),
            )
            print(
                "Official cost:",
                summary["official_cost"],
            )

            # Save vehicle-level decisions for auditing.
            decisions_df = pd.DataFrame({
                "vehicle_id": df["vehicle_id"],
                "actual_class": actual,
                "decision": decisions,
                "expected_benefit": benefits,
            })

            output_name = (
                f"capacity_decisions_{method}_"
                f"{int(fraction * 100)}pct.csv"
            )

            decisions_df.to_csv(
                REPORT_DIR / output_name,
                index=False,
            )

    comparison = pd.DataFrame(results)

    output_path = (
        REPORT_DIR
        / "capacity_constrained_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 85)
    print("CAPACITY-CONSTRAINED COMPARISON")
    print("=" * 85)

    print(
        comparison[
            [
                "method",
                "capacity_pct",
                "vehicles_flagged",
                "true_alerts",
                "false_alerts",
                "binary_precision",
                "binary_recall",
                "official_cost",
                "cost_reduction_pct",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved:", output_path)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
