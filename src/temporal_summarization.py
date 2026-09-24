from pathlib import Path

import pandas as pd


INTERIM_DATA_DIR = Path("data/interim")
PROCESSED_DATA_DIR = Path("data/processed")


PARQUET_FILES = {
    "train": {
        "operational": INTERIM_DATA_DIR / "train_operational_readouts.parquet",
        "specifications": INTERIM_DATA_DIR / "train_specifications.parquet",
        "target": INTERIM_DATA_DIR / "train_tte.parquet",
        "output": PROCESSED_DATA_DIR / "train_features.parquet",
    },
    "validation": {
        "operational": INTERIM_DATA_DIR / "validation_operational_readouts.parquet",
        "specifications": INTERIM_DATA_DIR / "validation_specifications.parquet",
        "target": INTERIM_DATA_DIR / "validation_labels.parquet",
        "output": PROCESSED_DATA_DIR / "validation_features.parquet",
    },
    "test": {
        "operational": INTERIM_DATA_DIR / "test_operational_readouts.parquet",
        "specifications": INTERIM_DATA_DIR / "test_specifications.parquet",
        "target": INTERIM_DATA_DIR / "test_labels.parquet",
        "output": PROCESSED_DATA_DIR / "test_features.parquet",
    },
}


ID_COLUMN = "vehicle_id"
TIME_COLUMN = "time_step"


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """
    Return operational sensor columns.
    Excludes vehicle_id and time_step.
    """
    return [column for column in df.columns if column not in [ID_COLUMN, TIME_COLUMN]]


def build_temporal_features(operational_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build vehicle-level temporal summary features from operational readouts.

    For each vehicle and each sensor, compute:
    - mean
    - std
    - min
    - max
    - last observed value
    """
    sensor_columns = get_sensor_columns(operational_df)

    print(f"Number of sensor columns: {len(sensor_columns)}")

    operational_df = operational_df.sort_values([ID_COLUMN, TIME_COLUMN])

    grouped = operational_df.groupby(ID_COLUMN, sort=False)

    print("Computing mean, std, min, max features...")

    summary_features = grouped[sensor_columns].agg(["mean", "std", "min", "max"])

    summary_features.columns = [
        f"{sensor}_{stat}" for sensor, stat in summary_features.columns
    ]

    summary_features = summary_features.reset_index()

    print("Computing last observed values...")

    last_values = grouped[sensor_columns].last().reset_index()
    last_values = last_values.rename(
        columns={column: f"{column}_last" for column in sensor_columns}
    )

    print("Computing number of readouts per vehicle...")

    readout_counts = grouped.size().reset_index(name="num_readouts")

    print("Merging temporal features...")

    features = summary_features.merge(last_values, on=ID_COLUMN, how="left")
    features = features.merge(readout_counts, on=ID_COLUMN, how="left")

    return features


def merge_static_and_target_features(
    temporal_features: pd.DataFrame,
    specifications_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge temporal features with static specifications and target labels.
    """
    print("Merging with vehicle specifications...")

    features = temporal_features.merge(
        specifications_df,
        on=ID_COLUMN,
        how="left",
    )

    print("Merging with target/label file...")

    features = features.merge(
        target_df,
        on=ID_COLUMN,
        how="left",
    )

    return features


def process_split(split_name: str, paths: dict[str, Path]) -> None:
    """
    Process one dataset split: train, validation, or test.
    """
    print("=" * 80)
    print(f"Processing split: {split_name}")

    print(f"Reading operational data: {paths['operational']}")
    operational_df = pd.read_parquet(paths["operational"])

    print(f"Operational shape: {operational_df.shape}")

    print(f"Reading specifications: {paths['specifications']}")
    specifications_df = pd.read_parquet(paths["specifications"])

    print(f"Specifications shape: {specifications_df.shape}")

    print(f"Reading target/labels: {paths['target']}")
    target_df = pd.read_parquet(paths["target"])

    print(f"Target shape: {target_df.shape}")

    temporal_features = build_temporal_features(operational_df)

    print(f"Temporal feature shape: {temporal_features.shape}")

    final_features = merge_static_and_target_features(
        temporal_features=temporal_features,
        specifications_df=specifications_df,
        target_df=target_df,
    )

    print(f"Final feature shape: {final_features.shape}")

    output_path = paths["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed features to: {output_path}")
    final_features.to_parquet(output_path, index=False)

    print(f"Saved: {output_path}")


def main() -> None:
    print("Starting temporal summarization...")

    for split_name, paths in PARQUET_FILES.items():
        process_split(split_name, paths)

    print("=" * 80)
    print("Temporal summarization completed successfully.")


if __name__ == "__main__":
    main()