
from pathlib import Path

import numpy as np
import pandas as pd

from drift_monitor_agent import DriftMonitorAgent


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed/"
    "train_random_model_fit.parquet"
)

OUTPUT_PATH = (
    ROOT / "reports/tables/"
    "full_drift_robustness.csv"
)

FEATURE = "397_31_recent_std"

SEEDS = [21, 42, 84]
SHIFTS = [0.0, 0.25, 0.50, 1.0]

SAMPLE_SIZE = 1500


def get_samples(data, seed):
    # Choose one snapshot from each vehicle.
    unique = (
        data.groupby("vehicle_id", group_keys=False)
        .sample(n=1, random_state=seed)
    )

    selected = unique.sample(
        n=2 * SAMPLE_SIZE,
        random_state=seed,
    )

    reference = selected.iloc[:SAMPLE_SIZE].copy()
    current = selected.iloc[SAMPLE_SIZE:].copy()

    assert not (
        set(reference.vehicle_id)
        & set(current.vehicle_id)
    )

    return reference, current


def main():
    print("Loading feature data...", flush=True)

    data = pd.read_parquet(INPUT_PATH)

    agent = DriftMonitorAgent()
    results = []

    for seed in SEEDS:
        print(f"\nRandom seed: {seed}", flush=True)

        reference, current = get_samples(data, seed)

        values = reference[FEATURE].to_numpy(
            dtype=np.float64
        )

        values = values[np.isfinite(values)]

        if len(values) < 100:
            raise ValueError("Insufficient reference values.")

        scale = float(np.std(values))

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid reference feature scale.")

        for shift in SHIFTS:
            modified = current.copy()

            original = modified[FEATURE].to_numpy(
                dtype=np.float64,
                copy=True,
            )

            finite = np.isfinite(original)

            original[finite] += shift * scale

            modified[FEATURE] = original

            report = agent.analyze(
                reference,
                modified,
            )

            target = report.set_index("feature").loc[
                FEATURE
            ]

            results.append({
                "seed": seed,
                "shift_std": shift,
                "target_ks": float(
                    target["ks_statistic"]
                ),
                "target_detected": bool(
                    target["distribution_drift"]
                ),
                "total_flagged": int(
                    report["flagged"].sum()
                ),
                "other_flagged": int(
                    report.loc[
                        report["feature"] != FEATURE,
                        "flagged",
                    ].sum()
                ),
            })

            print(
                f"Shift={shift:.2f} SD | "
                f"KS={target['ks_statistic']:.4f} | "
                f"Detected={target['distribution_drift']} | "
                f"Other flagged={results[-1]['other_flagged']}",
                flush=True,
            )

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FULL DRIFT ROBUSTNESS SUMMARY")
    print("=" * 70)

    summary = (
        result_df.groupby("shift_std")
        .agg(
            detected=("target_detected", "sum"),
            trials=("target_detected", "count"),
            mean_ks=("target_ks", "mean"),
            mean_other_flags=("other_flagged", "mean"),
        )
    )

    summary["detection_rate"] = (
        summary["detected"] / summary["trials"]
    )

    print(summary.round(4).to_string())

    print("\nSaved:", OUTPUT_PATH)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
