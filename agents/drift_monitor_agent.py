
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = (
    ROOT / "data/processed/train_random_model_fit.parquet"
)

CURRENT_PATH = (
    ROOT / "data/processed/validation_temporal_features.parquet"
)

OUTPUT_DIR = ROOT / "reports" / "tables"

EXCLUDED = {
    "vehicle_id",
    "class_label",
    "binary_label",
}

RANDOM_SEED = 42
MAX_SAMPLE_SIZE = 1500

FDR_LEVEL = 0.05
MIN_KS_EFFECT = 0.10
MIN_VALID_SAMPLES = 100
MISSING_RATE_CHANGE = 0.05


def sample_reference_vehicles(df):
    """
    Select one reproducible snapshot per vehicle.

    This prevents vehicles with four snapshots from
    automatically receiving four times the weight.
    """
    return (
        df.groupby("vehicle_id", group_keys=False)
        .sample(n=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )


def sample_rows(df, n=MAX_SAMPLE_SIZE):
    if len(df) <= n:
        return df.copy()

    return df.sample(
        n=n,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)


def adjust_pvalues_bh(pvalues):
    """
    Benjamini-Hochberg FDR adjustment.

    NaN p-values are assigned 1, since insufficient
    evidence is not evidence of no drift.
    """
    p = np.asarray(pvalues, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)

    count = len(p)

    if count == 0:
        return p

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted = sorted_p * count / np.arange(
        1, count + 1
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    result = np.empty(count, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)

    return result


class DriftMonitorAgent:
    def __init__(
        self,
        min_ks_effect=MIN_KS_EFFECT,
        fdr_level=FDR_LEVEL,
    ):
        self.min_ks_effect = min_ks_effect
        self.fdr_level = fdr_level

    def analyze(self, reference, current):
        numeric_columns = [
            c for c in reference.columns
            if c not in EXCLUDED
            and pd.api.types.is_numeric_dtype(
                reference[c]
            )
            and c in current.columns
            and pd.api.types.is_numeric_dtype(
                current[c]
            )
        ]

        results = []

        for column in numeric_columns:
            ref = reference[column].to_numpy(
                dtype=np.float64
            )

            cur = current[column].to_numpy(
                dtype=np.float64
            )

            ref = np.where(np.isfinite(ref), ref, np.nan)
            cur = np.where(np.isfinite(cur), cur, np.nan)

            missing_ref = float(np.isnan(ref).mean())
            missing_cur = float(np.isnan(cur).mean())

            ref_valid = ref[np.isfinite(ref)]
            cur_valid = cur[np.isfinite(cur)]

            sufficient = (
                len(ref_valid) >= MIN_VALID_SAMPLES
                and len(cur_valid) >= MIN_VALID_SAMPLES
            )

            if sufficient:
                test = ks_2samp(
                    ref_valid,
                    cur_valid,
                    method="auto",
                )

                ks_stat = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                ks_stat = np.nan
                p_value = 1.0

            results.append({
                "feature": column,
                "n_reference": len(ref_valid),
                "n_current": len(cur_valid),
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "reference_missing_rate": missing_ref,
                "current_missing_rate": missing_cur,
                "missing_rate_difference": abs(
                    missing_cur - missing_ref
                ),
                "sufficient_samples": sufficient,
            })

        report = pd.DataFrame(results)

        if report.empty:
            raise ValueError(
                "No shared numerical features found."
            )

        report["adjusted_p_value"] = adjust_pvalues_bh(
            report["p_value"].to_numpy()
        )

        report["distribution_drift"] = (
            report["sufficient_samples"]
            & (report["ks_statistic"] >= self.min_ks_effect)
            & (report["adjusted_p_value"] < self.fdr_level)
        )

        report["missingness_shift"] = (
            report["missing_rate_difference"]
            >= MISSING_RATE_CHANGE
        )

        report["flagged"] = (
            report["distribution_drift"]
            | report["missingness_shift"]
        )

        report = report.sort_values(
            ["flagged", "ks_statistic"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        return report

    def make_summary(self, report, scenario):
        shifted = int(report["distribution_drift"].sum())
        missing = int(report["missingness_shift"].sum())
        flagged = int(report["flagged"].sum())

        insufficient = int(
            (~report["sufficient_samples"]).sum()
        )

        status = (
            "REVIEW_REQUIRED"
            if flagged > 0 or insufficient > 0
            else "NO_SHIFT_DETECTED"
        )

        return {
            "scenario": scenario,
            "status": status,
            "features_tested": int(len(report)),
            "distribution_drift_features": shifted,
            "missingness_shift_features": missing,
            "flagged_features": flagged,
            "insufficient_sample_features": insufficient,
            "recommendation": (
                "Review feature distributions, data quality "
                "and model reliability before changing "
                "maintenance policies."
                if status == "REVIEW_REQUIRED"
                else "Continue monitoring. This result does "
                     "not guarantee that the model is reliable."
            ),
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Inject a synthetic shift into one feature.",
    )

    args = parser.parse_args()

    print("Loading datasets...", flush=True)

    reference = pd.read_parquet(REFERENCE_PATH)
    current = pd.read_parquet(CURRENT_PATH)

    if reference["vehicle_id"].duplicated().any():
        reference = sample_reference_vehicles(reference)

    if current["vehicle_id"].duplicated().any():
        raise ValueError(
            "Monitoring data should contain one "
            "snapshot per vehicle in this experiment."
        )

    if set(reference["vehicle_id"]) & set(
        current["vehicle_id"]
    ):
        raise ValueError(
            "Reference and monitoring vehicles overlap."
        )

    reference = sample_rows(reference)
    current = sample_rows(current)

    scenario = "observed_population_comparison"

    if args.synthetic:
        scenario = "synthetic_shift"

        feature = "397_31_recent_std"

        if feature not in current.columns:
            raise ValueError(
                f"Missing synthetic test feature: {feature}"
            )

        current = current.copy()

        values = current[feature].to_numpy(
            dtype=np.float64, copy=True
        )

        finite = np.isfinite(values)

        # An obvious synthetic location shift,
        # with scale derived only from reference data.
        ref_values = reference[feature].to_numpy(
            dtype=np.float64
        )

        ref_values = ref_values[
            np.isfinite(ref_values)
        ]

        scale = (
            np.std(ref_values)
            if len(ref_values) > 1
            else 1.0
        )

        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        values[finite] += 3.0 * scale

        current[feature] = values

        print(
            "Injected synthetic shift into:",
            feature,
            flush=True,
        )

    print("Reference vehicles:", len(reference))
    print("Monitoring vehicles:", len(current))

    agent = DriftMonitorAgent()

    print(
        "\nRunning statistical drift analysis...",
        flush=True,
    )

    report = agent.analyze(reference, current)
    summary = agent.make_summary(report, scenario)

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    csv_path = OUTPUT_DIR / (
        f"drift_monitor_{scenario}.csv"
    )

    json_path = OUTPUT_DIR / (
        f"drift_monitor_{scenario}.json"
    )

    report.to_csv(csv_path, index=False)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("DRIFT MONITOR AGENT RESULTS")
    print("=" * 60)

    print(json.dumps(summary, indent=2))

    print("\nTop 10 flagged features:")

    print(
        report.loc[
            report["flagged"],
            [
                "feature",
                "ks_statistic",
                "adjusted_p_value",
                "missing_rate_difference",
                "distribution_drift",
                "missingness_shift",
            ],
        ].head(10).to_string(index=False)
    )

    print("\nSaved CSV:", csv_path)
    print("Saved JSON:", json_path)
    print("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
