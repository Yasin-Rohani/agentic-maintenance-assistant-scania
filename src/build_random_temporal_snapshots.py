
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

READOUTS_PATH = (
    ROOT / "data/interim/train_operational_readouts.parquet"
)

TTE_PATH = ROOT / "data/interim/train_tte.parquet"

OUTPUT_PATH = (
    ROOT / "data/processed/"
    "train_temporal_snapshots_random.parquet"
)

RANDOM_STATE = 42
MAX_SNAPSHOTS_PER_VEHICLE = 4
HORIZON = 48.0
EPS = 1e-5


def assign_class(delta, repaired):
    """
    Assign the five SCANIA classes.

    A repaired vehicle has a known repair time.

    For an unrepaired vehicle, class 0 is valid only
    when observation continues beyond the prediction
    horizon. Otherwise the snapshot is censored.
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
    rng = np.random.default_rng(RANDOM_STATE)

    print("Loading readout timestamps...")

    readouts = pd.read_parquet(
        READOUTS_PATH,
        columns=["vehicle_id", "time_step"],
    )

    tte = pd.read_parquet(TTE_PATH)

    readouts = readouts.sort_values(
        ["vehicle_id", "time_step"]
    )

    tte_lookup = tte.set_index("vehicle_id")

    records = []

    excluded_censored_candidates = 0
    vehicles_without_valid_snapshot = 0

    print("Generating random temporal snapshots...")

    for vehicle_id, group in readouts.groupby(
        "vehicle_id", sort=True
    ):
        vehicle_id = int(vehicle_id)

        if vehicle_id not in tte_lookup.index:
            raise ValueError(
                f"Missing repair record: {vehicle_id}"
            )

        repair_info = tte_lookup.loc[vehicle_id]

        end_time = float(
            repair_info["length_of_study_time_step"]
        )

        repaired = int(
            repair_info["in_study_repair"]
        )

        times = np.sort(
            group["time_step"]
            .dropna()
            .unique()
        )

        # First identify all timestamps with a valid label.
        valid_candidates = []

        for cutoff in times:
            delta = end_time - float(cutoff)

            label = assign_class(delta, repaired)

            if label is None:
                excluded_censored_candidates += 1
                continue

            valid_candidates.append(
                (float(cutoff), int(label))
            )

        if not valid_candidates:
            vehicles_without_valid_snapshot += 1
            continue

        # Uniform random sampling without replacement.
        # No forced final readout.
        n = min(
            MAX_SNAPSHOTS_PER_VEHICLE,
            len(valid_candidates),
        )

        chosen_indices = np.sort(
            rng.choice(
                len(valid_candidates),
                size=n,
                replace=False,
            )
        )

        for index in chosen_indices:
            cutoff, label = valid_candidates[index]

            readout_count = int(
                np.searchsorted(
                    times,
                    cutoff + EPS,
                    side="right",
                )
            )

            records.append({
                "vehicle_id": vehicle_id,
                "cutoff_time": cutoff,
                "class_label": label,
                "binary_label": int(label > 0),
                "in_study_repair": repaired,
                "observation_end_time": end_time,
                "num_readouts_at_cutoff": readout_count,
            })

    snapshots = pd.DataFrame(records)

    snapshots = snapshots.sort_values(
        ["vehicle_id", "cutoff_time"]
    ).reset_index(drop=True)

    assert not snapshots.duplicated(
        ["vehicle_id", "cutoff_time"]
    ).any()

    assert (
        snapshots["cutoff_time"]
        <= snapshots["observation_end_time"] + EPS
    ).all()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nRandom temporal dataset created.")
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

    print("\nVehicles without valid snapshots:")
    print(vehicles_without_valid_snapshot)

    print("\nExcluded censored candidate timestamps:")
    print(excluded_censored_candidates)

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
