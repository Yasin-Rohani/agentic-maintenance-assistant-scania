
from pathlib import Path
import json

import numpy as np
import pandas as pd

from drift_monitor_agent import DriftMonitorAgent


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed/train_random_model_fit.parquet"
)

OUTPUT_DIR = ROOT / "reports" / "tables"

RANDOM_STATE = 42
SAMPLE_SIZE = 1500
SHIFT_FEATURE = "397_31_recent_std"
SHIFT_STRENGTH = 3.0


def prepare_samples():
    print("Loading historical training features...")

    df = pd.read_parquet(INPUT_PATH)

    # Exactly one snapshot per vehicle.
    df = (
        df.groupby("vehicle_id", group_keys=False)
        .sample(n=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    required = SAMPLE_SIZE * 2

    if len(df) < required:
        raise ValueError(
            f"Need at least {required} distinct vehicles."
        )

    sampled = df.sample(
        n=required,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    reference = sampled.iloc[:SAMPLE_SIZE].copy()
    monitoring = sampled.iloc[SAMPLE_SIZE:].copy()

    reference_ids = set(reference["vehicle_id"])
    monitoring_ids = set(monitoring["vehicle_id"])

    assert not reference_ids & monitoring_ids

    print("Reference vehicles:", len(reference))
    print("Monitoring vehicles:", len(monitoring))
    print("Vehicle overlap: 0")

    return reference, monitoring


def inject_shift(reference, monitoring):
    modified = monitoring.copy(deep=True)

    if SHIFT_FEATURE not in modified.columns:
        raise ValueError(
            f"Feature not found: {SHIFT_FEATURE}"
        )

    ref_values = reference[SHIFT_FEATURE].to_numpy(
        dtype=np.float64
    )

    ref_values = ref_values[np.isfinite(ref_values)]

    if len(ref_values) < 2:
        raise ValueError(
            "Insufficient reference values for injection."
        )

    scale = float(np.std(ref_values))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Reference standard deviation must be positive."
        )

    values = modified[SHIFT_FEATURE].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    finite = np.isfinite(values)

    values[finite] += SHIFT_STRENGTH * scale

    modified[SHIFT_FEATURE] = values

    print(
        f"Injected +{SHIFT_STRENGTH} reference standard "
        f"deviations into {SHIFT_FEATURE}"
    )

    return modified


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference, monitoring = prepare_samples()

    agent = DriftMonitorAgent()

    print("\nRunning no-shift control...")

    baseline = agent.analyze(
        reference,
        monitoring,
    )

    print("Running injected-shift experiment...")

    shifted_monitoring = inject_shift(
        reference,
        monitoring,
    )

    shifted = agent.analyze(
        reference,
        shifted_monitoring,
    )

    # Align reports by exact feature name.
    baseline = baseline.set_index("feature")
    shifted = shifted.set_index("feature")

    common = baseline.index.intersection(shifted.index)

    baseline = baseline.loc[common]
    shifted = shifted.loc[common]

    baseline_ks = baseline["ks_statistic"].to_numpy()
    shifted_ks = shifted["ks_statistic"].to_numpy()

    ks_change = shifted_ks - baseline_ks

    baseline_flags = baseline["flagged"].to_numpy(
        dtype=bool
    )
    shifted_flags = shifted["flagged"].to_numpy(
        dtype=bool
    )

    new_flags = (~baseline_flags) & shifted_flags

    if SHIFT_FEATURE not in common:
        raise ValueError(
            "Injected feature missing from reports."
        )

    injected_baseline = baseline.loc[SHIFT_FEATURE]
    injected_shifted = shifted.loc[SHIFT_FEATURE]

    summary = {
        "reference_vehicles": len(reference),
        "monitoring_vehicles": len(monitoring),
        "features_compared": len(common),
        "baseline_flagged": int(baseline_flags.sum()),
        "synthetic_flagged": int(shifted_flags.sum()),
        "newly_flagged": int(new_flags.sum()),
        "ks_increased_features": int(
            np.sum(ks_change > 1e-8)
        ),
        "ks_decreased_features": int(
            np.sum(ks_change < -1e-8)
        ),
        "injected_feature": SHIFT_FEATURE,
        "injected_baseline_ks": float(
            injected_baseline["ks_statistic"]
        ),
        "injected_synthetic_ks": float(
            injected_shifted["ks_statistic"]
        ),
        "injected_baseline_flagged": bool(
            injected_baseline["flagged"]
        ),
        "injected_synthetic_flagged": bool(
            injected_shifted["flagged"]
        ),
    }

    baseline.reset_index().to_csv(
        OUTPUT_DIR / "controlled_drift_baseline.csv",
        index=False,
    )

    shifted.reset_index().to_csv(
        OUTPUT_DIR / "controlled_drift_synthetic.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "controlled_drift_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CONTROLLED DRIFT EXPERIMENT RESULTS")
    print("=" * 60)

    print(json.dumps(summary, indent=2))

    print("\nSTATUS: SUCCESS")


if __name__ == "__main__":
    main()
