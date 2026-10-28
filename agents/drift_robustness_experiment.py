
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/train_random_model_fit.parquet"
OUTPUT = ROOT / "reports/tables/drift_robustness.csv"

FEATURE = "397_31_recent_std"
SEEDS = [21, 42, 84]
SHIFT_LEVELS = [0.0, 1.0, 2.0, 3.0]
SAMPLE_SIZE = 1500
KS_THRESHOLD = 0.10
P_THRESHOLD = 0.05

def main():
    data = pd.read_parquet(
        INPUT, columns=["vehicle_id", FEATURE]
    )

    rows = []

    for seed in SEEDS:
        # One snapshot per vehicle, then two disjoint groups.
        unique = (
            data.groupby("vehicle_id", group_keys=False)
            .sample(n=1, random_state=seed)
            .sample(n=2 * SAMPLE_SIZE, random_state=seed)
        )

        reference = unique.iloc[:SAMPLE_SIZE]
        monitoring = unique.iloc[SAMPLE_SIZE:]

        assert not (
            set(reference.vehicle_id)
            & set(monitoring.vehicle_id)
        )

        ref = reference[FEATURE].to_numpy(dtype=float)
        cur = monitoring[FEATURE].to_numpy(dtype=float)

        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        scale = np.std(ref)

        if len(ref) < 100 or len(cur) < 100 or scale <= 0:
            raise ValueError("Insufficient usable feature data.")

        for shift in SHIFT_LEVELS:
            shifted = cur + shift * scale

            test = ks_2samp(
                ref, shifted, method="asymp"
            )

            detected = (
                test.statistic >= KS_THRESHOLD
                and test.pvalue < P_THRESHOLD
            )

            rows.append({
                "seed": seed,
                "shift_std": shift,
                "ks_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "detected": bool(detected),
            })

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\nDRIFT ROBUSTNESS RESULTS")
    print("=" * 65)
    print(result.round(5).to_string(index=False))

    print("\nDETECTION RATE BY SHIFT STRENGTH")
    print(
        result.groupby("shift_std")["detected"]
        .agg(["sum", "count", "mean"])
        .to_string()
    )

    print("\nSaved:", OUTPUT)
    print("STATUS: SUCCESS")

if __name__ == "__main__":
    main()
