
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

READOUTS_PATH = (
    ROOT / "data/interim/train_operational_readouts.parquet"
)

SPECIFICATIONS_PATH = (
    ROOT / "data/interim/train_specifications.parquet"
)

DATASETS = {
    "fixed": {
        "snapshots": (
            ROOT / "data/processed/train_temporal_snapshots.parquet"
        ),
        "features": (
            ROOT / "data/processed/train_temporal_features.parquet"
        ),
    },
    "random": {
        "snapshots": (
            ROOT / "data/processed/train_temporal_snapshots_random.parquet"
        ),
        "features": (
            ROOT / "data/processed/train_temporal_features_random.parquet"
        ),
    },
}

BATCH_SIZE = 50000
FLUSH_ROWS = 1500

ID_COL = "vehicle_id"
TIME_COL = "time_step"


# ============================================================
# 2. FAST NUMERICAL FEATURE SUMMARIZATION
# ============================================================

def summarize_window(frame, sensor_columns, suffix):
    """
    Calculate numerical statistics for a historical window.

    Missing values are ignored when possible.

    A sample standard deviation is undefined when fewer
    than two valid values are available.
    """

    values = frame[sensor_columns].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            RuntimeWarning,
        )

        means = np.nanmean(values, axis=0)

        stds = np.nanstd(
            values,
            axis=0,
            ddof=1,
        )

        minimums = np.nanmin(
            values,
            axis=0,
        )

        maximums = np.nanmax(
            values,
            axis=0,
        )

    result = {}

    for i, sensor in enumerate(sensor_columns):

        result[f"{sensor}_{suffix}mean"] = means[i]
        result[f"{sensor}_{suffix}std"] = stds[i]
        result[f"{sensor}_{suffix}min"] = minimums[i]
        result[f"{sensor}_{suffix}max"] = maximums[i]

    return result


# ============================================================
# 3. BUILD FEATURES FOR A SINGLE SNAPSHOT
# ============================================================

def build_snapshot_features(
    vehicle_df,
    cutoff_time,
    sensor_columns,
):
    """
    Extract features only from observations at or before
    the selected cutoff time.

    No future readouts are used.
    """

    history = vehicle_df.loc[
        vehicle_df[TIME_COL] <= cutoff_time
    ]

    if history.empty:
        return None

    count = len(history)

    # Last 20 percent of available historical observations.
    recent_size = max(
        1,
        int(np.ceil(count * 0.20)),
    )

    recent = history.tail(recent_size)

    features = {
        "num_readouts": count,
        "cutoff_time": float(cutoff_time),
    }

    # Statistics over the complete available history.
    global_features = summarize_window(
        history,
        sensor_columns,
        "",
    )

    # Statistics over recent observations.
    recent_features = summarize_window(
        recent,
        sensor_columns,
        "recent_",
    )

    features.update(global_features)
    features.update(recent_features)

    # Most recent non-missing value of each sensor.
    last_values = (
        history[sensor_columns]
        .ffill()
        .iloc[-1]
    )

    for sensor in sensor_columns:

        global_mean = features[f"{sensor}_mean"]

        recent_mean = features[
            f"{sensor}_recent_mean"
        ]

        last_value = last_values[sensor]

        features[f"{sensor}_last"] = last_value

        features[
            f"{sensor}_recent_minus_global_mean"
        ] = recent_mean - global_mean

        features[
            f"{sensor}_last_minus_global_mean"
        ] = last_value - global_mean

    return features


# ============================================================
# 4. PROCESS ALL SNAPSHOTS OF ONE VEHICLE
# ============================================================

def process_vehicle(
    vehicle_df,
    snapshot_lookup,
    spec_lookup,
    sensor_columns,
):
    vehicle_id = int(
        vehicle_df[ID_COL].iloc[0]
    )

    if vehicle_id not in snapshot_lookup:
        return []

    if vehicle_id not in spec_lookup.index:
        raise ValueError(
            f"Specifications missing for vehicle {vehicle_id}"
        )

    vehicle_df = vehicle_df.sort_values(
        TIME_COL,
        kind="stable",
    )

    specifications = spec_lookup.loc[
        vehicle_id
    ]

    results = []

    for snapshot in snapshot_lookup[vehicle_id]:

        cutoff = float(
            snapshot["cutoff_time"]
        )

        features = build_snapshot_features(
            vehicle_df,
            cutoff,
            sensor_columns,
        )

        if features is None:
            raise ValueError(
                f"No historical readouts for vehicle "
                f"{vehicle_id} at cutoff {cutoff}"
            )

        row = {
            "vehicle_id": vehicle_id,
            "class_label": int(
                snapshot["class_label"]
            ),
            "binary_label": int(
                snapshot["binary_label"]
            ),
        }

        row.update(features)

        # Add vehicle specifications.
        for col in spec_lookup.columns:

            value = specifications[col]

            row[col] = (
                str(value)
                if pd.notna(value)
                else "MISSING"
            )

        results.append(row)

    return results


# ============================================================
# 5. MAIN PROCESSING PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe temporal features "
            "for SCANIA training snapshots."
        )
    )

    parser.add_argument(
        "--sampling",
        choices=["fixed", "random"],
        required=True,
        help="Select the snapshot sampling strategy.",
    )

    args = parser.parse_args()

    snapshots_path = DATASETS[
        args.sampling
    ]["snapshots"]

    output_path = DATASETS[
        args.sampling
    ]["features"]

    print("=" * 55)
    print("SCANIA TEMPORAL FEATURE ENGINEERING")
    print("=" * 55)

    print("Sampling strategy:", args.sampling)
    print("Snapshot file:", snapshots_path)
    print("Output file:", output_path)

    # --------------------------------------------------------
    # Load snapshot labels and vehicle specifications.
    # --------------------------------------------------------

    print("\nLoading snapshots and specifications...")

    snapshots = pd.read_parquet(
        snapshots_path
    )

    specifications = pd.read_parquet(
        SPECIFICATIONS_PATH
    )

    # Validate snapshot identifiers.
    if snapshots.duplicated(
        [ID_COL, "cutoff_time"]
    ).any():
        raise ValueError(
            "Duplicate vehicle/cutoff snapshot pairs found."
        )

    snapshot_lookup = {
        int(vehicle_id): group.to_dict("records")
        for vehicle_id, group in snapshots.groupby(
            ID_COL,
            sort=True,
        )
    }

    spec_lookup = specifications.set_index(
        ID_COL
    )

    if not spec_lookup.index.is_unique:
        raise ValueError(
            "Duplicate vehicle IDs in specifications."
        )

    # --------------------------------------------------------
    # Open operational data as a streaming Parquet source.
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(
        READOUTS_PATH
    )

    sensor_columns = [
        col
        for col in parquet_file.schema_arrow.names
        if col not in [ID_COL, TIME_COL]
    ]

    print("\nSensor columns:", len(sensor_columns))
    print("Expected snapshots:", len(snapshots))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = None
    buffer = []

    pending_vehicle = pd.DataFrame()

    written_rows = 0
    processed_vehicles = 0

    # --------------------------------------------------------
    # Write buffered feature records to Parquet.
    # --------------------------------------------------------

    def flush_buffer():

        nonlocal writer
        nonlocal buffer
        nonlocal written_rows

        if not buffer:
            return

        df = pd.DataFrame(buffer)

        categorical_columns = list(
            spec_lookup.columns
        )

        non_feature_columns = [
            "vehicle_id",
            "class_label",
            "binary_label",
        ] + categorical_columns

        numerical_columns = [
            col
            for col in df.columns
            if col not in non_feature_columns
        ]

        df[numerical_columns] = df[
            numerical_columns
        ].astype("float32")

        # Preserve integer identifiers and targets.
        df["vehicle_id"] = df[
            "vehicle_id"
        ].astype("int64")

        df["class_label"] = df[
            "class_label"
        ].astype("int8")

        df["binary_label"] = df[
            "binary_label"
        ].astype("int8")

        for col in categorical_columns:
            df[col] = df[col].astype(str)

        table = pa.Table.from_pandas(
            df,
            preserve_index=False,
        )

        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="snappy",
            )

        writer.write_table(
            pa.Table.from_pandas(
                df,
                schema=writer.schema,
                preserve_index=False,
            )
        )

        written_rows += len(df)

        buffer = []

    # --------------------------------------------------------
    # Process complete vehicle histories.
    # --------------------------------------------------------

    def handle_complete_vehicles(frame):

        nonlocal processed_vehicles
        nonlocal buffer

        if frame.empty:
            return

        for _, vehicle_df in frame.groupby(
            ID_COL,
            sort=False,
        ):

            rows = process_vehicle(
                vehicle_df,
                snapshot_lookup,
                spec_lookup,
                sensor_columns,
            )

            buffer.extend(rows)

            processed_vehicles += 1

            if len(buffer) >= FLUSH_ROWS:
                flush_buffer()

            if processed_vehicles % 500 == 0:
                print(
                    f"Vehicles processed: "
                    f"{processed_vehicles}, "
                    f"rows written: "
                    f"{written_rows}",
                    flush=True,
                )

    # --------------------------------------------------------
    # Streaming feature extraction.
    # --------------------------------------------------------

    print("\nStarting batch-based feature extraction...")

    try:
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            df = batch.to_pandas()

            if not pending_vehicle.empty:

                df = pd.concat(
                    [pending_vehicle, df],
                    ignore_index=True,
                )

            # The input file is sorted by vehicle_id/time_step.
            # Keep the final vehicle in each batch because its
            # readouts may continue in the next batch.

            last_vehicle_id = df[
                ID_COL
            ].iloc[-1]

            complete = df.loc[
                df[ID_COL] != last_vehicle_id
            ]

            pending_vehicle = df.loc[
                df[ID_COL] == last_vehicle_id
            ].copy()

            handle_complete_vehicles(
                complete
            )

        # Process the final vehicle after all batches.
        handle_complete_vehicles(
            pending_vehicle
        )

        flush_buffer()

    finally:
        if writer is not None:
            writer.close()

    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 55)

    print(
        "Processed vehicles:",
        processed_vehicles,
    )

    print(
        "Saved feature rows:",
        written_rows,
    )

    print(
        "Expected feature rows:",
        len(snapshots),
    )

    if written_rows != len(snapshots):
        raise RuntimeError(
            "Feature row count does not match "
            "snapshot count!"
        )

    print("\nOutput:", output_path)
    print("Status: SUCCESS")


if __name__ == "__main__":
    main()
