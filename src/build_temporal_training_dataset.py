
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

READOUTS_PATH = INTERIM / "train_operational_readouts.parquet"
TTE_PATH = INTERIM / "train_tte.parquet"

OUTPUT_PATH = PROCESSED / "train_temporal_snapshots.parquet"

SNAPSHOT_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
HORIZON = 48.0
EPS = 1e-5


def get_class_label(delta: float, repaired: int):
    """
    Assign SCANIA's five-class target at a historical cutoff.

    Return None when an unrepaired vehicle has insufficient
    follow-up to establish a 48-time-unit negative label.
    """
    if repaired == 0:
        if delta > HORIZON + EPS:
            return 0
        return None

    if delta < -EPS:
        return None

    if delta <= 6 + EPS:
        return 4
    if delta <= 12 + EPS:
        return 3
    if delta <= 24 + EPS:
        return 2
    if delta <= 48 + EPS:
        return 1

    return 0


def main():
    print("Loading readout timestamps...")

    readouts = pd.read_parquet(
        READOUTS_PATH,
        columns=["vehicle_id", "time_step"],
    )

    tte = pd.read_parquet(TTE_PATH)

    readouts = readouts.sort_values(
        ["vehicle_id", "time_step"]
    ).reset_index(drop=True)

    tte_lookup = tte.set_index("vehicle_id")

    records = []
    excluded_censored = 0

    print("Building temporal snapshots...")

    for vehicle_id, group in readouts.groupby(
        "vehicle_id", sort=False
    ):
        if vehicle_id not in tte_lookup.index:
            continue

        target = tte_lookup.loc[vehicle_id]

        end_time = float(target["length_of_study_time_step"])
        repaired = int(target["in_study_repair"])

        times = np.sort(
            group["time_step"].dropna().unique()
        )

        if len(times) == 0:
            continue

        indices = sorted(set(
            max(0, int(np.ceil(len(times) * f)) - 1)
            for f in SNAPSHOT_FRACTIONS
        ))

        for index in indices:
            cutoff = float(times[index])
            delta = end_time - cutoff

            label = get_class_label(delta, repaired)

            if label is None:
                excluded_censored += 1
                continue

            records.append({
                "vehicle_id": int(vehicle_id),
                "cutoff_time": cutoff,
                "class_label": int(label),
                "binary_label": int(label > 0),
                "in_study_repair": repaired,
                "observation_end_time": end_time,
                "num_readouts_at_cutoff": int(
                    np.searchsorted(
                        group["time_step"].to_numpy(),
                        cutoff,
                        side="right",
                    )
                ),
            })

    snapshots = pd.DataFrame(records)

    PROCESSED.mkdir(parents=True, exist_ok=True)

    snapshots.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nTemporal training dataset created.")
    print("Shape:", snapshots.shape)

    print("\nClass distribution:")
    print(
        snapshots["class_label"]
        .value_counts()
        .sort_index()
    )

    print("\nBinary distribution:")
    print(
        snapshots["binary_label"]
        .value_counts()
        .sort_index()
    )

    print("\nUnique vehicles:")
    print(snapshots["vehicle_id"].nunique())

    print("\nExcluded censored snapshots:")
    print(excluded_censored)

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
