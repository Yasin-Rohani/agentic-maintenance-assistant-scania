
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed/"
    "train_temporal_features_random.parquet"
)

OUTPUT_DIR = ROOT / "data/processed"

RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.20


def main():
    df = pd.read_parquet(INPUT_PATH)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_idx, calibration_idx = next(
        splitter.split(
            df,
            groups=df["vehicle_id"],
        )
    )

    train = df.iloc[train_idx].copy()
    calibration = df.iloc[calibration_idx].copy()

    train_ids = set(train["vehicle_id"])
    calibration_ids = set(calibration["vehicle_id"])

    overlap = train_ids & calibration_ids

    assert not overlap, "Vehicle leakage detected!"

    for name, subset in [
        ("Model training", train),
        ("Calibration", calibration),
    ]:
        print(f"\n{name}")
        print("Snapshots:", len(subset))
        print("Vehicles:", subset["vehicle_id"].nunique())

        print("\nClass distribution:")
        print(
            subset["class_label"]
            .value_counts()
            .sort_index()
        )

    assert set(train["class_label"]) == set(range(5))
    assert set(calibration["class_label"]) == set(range(5))

    train_path = (
        OUTPUT_DIR / "train_random_model_fit.parquet"
    )

    calibration_path = (
        OUTPUT_DIR / "train_random_calibration.parquet"
    )

    train.to_parquet(train_path, index=False)
    calibration.to_parquet(calibration_path, index=False)

    print("\nVehicle overlap:", len(overlap))
    print("Training output:", train_path)
    print("Calibration output:", calibration_path)
    print("\nSTATUS: PASSED")


if __name__ == "__main__":
    main()
