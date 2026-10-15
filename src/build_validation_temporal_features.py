
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_temporal_features import build_snapshot_features


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

READOUTS_PATH = (
    ROOT / "data/interim/"
    "validation_operational_readouts.parquet"
)

SPECIFICATIONS_PATH = (
    ROOT / "data/interim/"
    "validation_specifications.parquet"
)

LABELS_PATH = (
    ROOT / "data/interim/"
    "validation_labels.parquet"
)

OUTPUT_PATH = (
    ROOT / "data/processed/"
    "validation_temporal_features.parquet"
)

BATCH_SIZE = 50000
FLUSH_ROWS = 1000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("SCANIA VALIDATION TEMPORAL FEATURES")
    print("=" * 55)

    specifications = pd.read_parquet(
        SPECIFICATIONS_PATH
    )

    labels = pd.read_parquet(
        LABELS_PATH
    )

    specifications = specifications.set_index(
        "vehicle_id"
    )

    labels = labels.set_index(
        "vehicle_id"
    )

    parquet_file = pq.ParquetFile(
        READOUTS_PATH
    )

    sensor_columns = [
        col
        for col in parquet_file.schema_arrow.names
        if col not in ["vehicle_id", "time_step"]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = None
    buffer = []
    pending_vehicle = pd.DataFrame()

    processed_vehicles = 0
    written_rows = 0

    def flush_buffer():

        nonlocal writer, buffer, written_rows

        if not buffer:
            return

        df = pd.DataFrame(buffer)

        categorical_columns = list(
            specifications.columns
        )

        non_numeric_columns = [
            "vehicle_id",
            "class_label",
            "binary_label",
        ] + categorical_columns

        numeric_columns = [
            col
            for col in df.columns
            if col not in non_numeric_columns
        ]

        df[numeric_columns] = df[
            numeric_columns
        ].astype("float32")

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
                OUTPUT_PATH,
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

    def process_vehicles(frame):

        nonlocal processed_vehicles, buffer

        if frame.empty:
            return

        for vehicle_id, vehicle_df in frame.groupby(
            "vehicle_id",
            sort=False,
        ):

            vehicle_id = int(vehicle_id)

            if vehicle_id not in specifications.index:
                raise ValueError(
                    f"Missing specifications: {vehicle_id}"
                )

            if vehicle_id not in labels.index:
                raise ValueError(
                    f"Missing label: {vehicle_id}"
                )

            vehicle_df = vehicle_df.sort_values(
                "time_step"
            )

            cutoff = float(
                vehicle_df["time_step"].max()
            )

            features = build_snapshot_features(
                vehicle_df,
                cutoff,
                sensor_columns,
            )

            if features is None:
                raise ValueError(
                    f"No features for vehicle {vehicle_id}"
                )

            class_label = int(
                labels.loc[vehicle_id, "class_label"]
            )

            row = {
                "vehicle_id": vehicle_id,
                "class_label": class_label,
                "binary_label": int(class_label > 0),
            }

            row.update(features)

            vehicle_specs = specifications.loc[
                vehicle_id
            ]

            for col in specifications.columns:

                value = vehicle_specs[col]

                row[col] = (
                    str(value)
                    if pd.notna(value)
                    else "MISSING"
                )

            buffer.append(row)

            processed_vehicles += 1

            if len(buffer) >= FLUSH_ROWS:
                flush_buffer()

            if processed_vehicles % 500 == 0:
                print(
                    f"Vehicles processed: {processed_vehicles}, "
                    f"rows written: {written_rows}",
                    flush=True,
                )

    print("\nStarting validation feature extraction...")

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

            last_vehicle_id = df[
                "vehicle_id"
            ].iloc[-1]

            complete = df.loc[
                df["vehicle_id"] != last_vehicle_id
            ]

            pending_vehicle = df.loc[
                df["vehicle_id"] == last_vehicle_id
            ].copy()

            process_vehicles(complete)

        process_vehicles(pending_vehicle)

        flush_buffer()

    finally:
        if writer is not None:
            writer.close()

    print("\n" + "=" * 55)
    print("VALIDATION FEATURE EXTRACTION COMPLETED")
    print("=" * 55)

    print("Processed vehicles:", processed_vehicles)
    print("Saved rows:", written_rows)

    if written_rows != len(labels):
        raise RuntimeError(
            "Validation row count mismatch!"
        )

    print("Output:", OUTPUT_PATH)
    print("Status: SUCCESS")


if __name__ == "__main__":
    main()

