
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "tables"

N_BINS = 10


def analyze_class(name, class_id):
    path = REPORT_DIR / f"validation_multiclass_{name}.csv"

    df = pd.read_csv(path)

    probability_column = f"prob_class_{class_id}"

    analysis = pd.DataFrame({
        "probability": df[probability_column],
        "observed": (
            df["actual_class"] == class_id
        ).astype(int),
    })

    # Quantile bins provide approximately equal sample sizes.
    analysis["bin"] = pd.qcut(
        analysis["probability"],
        q=N_BINS,
        duplicates="drop",
    )

    reliability = (
        analysis.groupby("bin", observed=True)
        .agg(
            n_vehicles=("observed", "size"),
            mean_predicted=("probability", "mean"),
            observed_frequency=("observed", "mean"),
            actual_events=("observed", "sum"),
        )
        .reset_index(drop=True)
    )

    reliability.insert(0, "sampling", name)
    reliability.insert(1, "class_id", class_id)

    return reliability


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name in ["fixed", "random"]:
        for class_id in [0, 1, 2, 3, 4]:
            result = analyze_class(name, class_id)
            results.append(result)

    comparison = pd.concat(
        results,
        ignore_index=True,
    )

    output = (
        REPORT_DIR / "class_reliability_analysis.csv"
    )

    comparison.to_csv(output, index=False)

    for name in ["fixed", "random"]:
        print("\n" + "=" * 55)
        print(f"{name.upper()} - CLASS 4 RELIABILITY")
        print("=" * 55)

        subset = comparison[
            (comparison["sampling"] == name)
            & (comparison["class_id"] == 4)
        ]

        print(subset.to_string(index=False))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
