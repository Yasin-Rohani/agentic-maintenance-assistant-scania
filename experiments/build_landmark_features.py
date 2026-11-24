"""
Build landmark features for the survival (discrete-time hazard) baseline, E03.

Difference from src/build_random_temporal_snapshots.py
------------------------------------------------------
The classifier pipeline samples snapshots only among cutoff times that have a
valid class label. For an unrepaired (censored) vehicle that excludes every
cutoff within 48 time units of the end of observation, so the late history of
censored vehicles never enters training. E01/E02 showed that this protocol
creates the cutoff-time shortcut.

Here landmarks are sampled uniformly from *all* readout times of every
training vehicle, with no label filter. Each landmark keeps its censoring
information (remaining observed time, repair indicator), so the hazard model
can use partial information from censored vehicles correctly.

Features are computed with the project's own
src/build_temporal_features.build_snapshot_features, so landmark features are
identical in definition and column order to the classifier features.

Output
------
data/processed/train_landmark_features.parquet with, per landmark:
  vehicle_id, class_label (-1 if undefined), binary_label (-1 if undefined),
  remaining_time (= length_of_study_time_step - cutoff_time),
  in_study_repair, then exactly the feature columns of the classifier file.

Memory: readouts are streamed in batches; one vehicle is processed at a time.

Usage (from the repository root, roughly 40-90 minutes):
    python experiments/build_landmark_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_random_temporal_snapshots import assign_class  # noqa: E402
from build_temporal_features import build_snapshot_features  # noqa: E402

READOUTS = ROOT / "data/interim/train_operational_readouts.parquet"
TTE = ROOT / "data/interim/train_tte.parquet"
SPECS = ROOT / "data/interim/train_specifications.parquet"
REFERENCE = ROOT / "data/processed/train_random_model_fit.parquet"
OUTPUT = ROOT / "data/processed/train_landmark_features.parquet"

ID, TIME = "vehicle_id", "time_step"
META = ["vehicle_id", "class_label", "binary_label",
        "remaining_time", "in_study_repair"]


def sample_landmarks(times, k, rng):
    times = np.unique(times[~np.isnan(times)])
    if len(times) <= k:
        return times
    return np.sort(rng.choice(times, k, replace=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-per-vehicle", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--flush-rows", type=int, default=2_000)
    parser.add_argument("--readouts", default=str(READOUTS))
    parser.add_argument("--tte", default=str(TTE))
    parser.add_argument("--specs", default=str(SPECS))
    parser.add_argument("--reference", default=str(REFERENCE))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.parquet")

    # Column order must match the classifier feature file exactly.
    ref_cols = pq.ParquetFile(args.reference).schema_arrow.names
    feature_cols = [c for c in ref_cols
                    if c not in ("vehicle_id", "class_label", "binary_label")]
    spec_cols = [c for c in feature_cols if c.startswith("Spec_")]
    numeric_cols = [c for c in feature_cols if c not in spec_cols]

    tte = pd.read_parquet(args.tte).set_index(ID)
    specs = pd.read_parquet(args.specs).set_index(ID)

    readouts = pq.ParquetFile(args.readouts)
    sensors = [c for c in readouts.schema_arrow.names if c not in (ID, TIME)]

    rng = np.random.default_rng(args.seed)
    buffer, writer = [], None
    written = vehicles = 0
    start = time.time()

    def flush():
        nonlocal buffer, writer, written
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns missing: {sorted(missing)[:5]}")
        df = df[META + feature_cols]
        df[numeric_cols] = df[numeric_cols].astype("float32")
        for c in ("class_label", "binary_label", "in_study_repair"):
            df[c] = df[c].astype("int8")
        df["vehicle_id"] = df["vehicle_id"].astype("int64")
        df["remaining_time"] = df["remaining_time"].astype("float32")
        for c in spec_cols:
            df[c] = df[c].astype(str)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(partial, table.schema, compression="snappy")
        writer.write_table(table.cast(writer.schema))
        written += len(df)
        buffer = []

    def process(vehicle_df):
        nonlocal vehicles
        vid = int(vehicle_df[ID].iloc[0])
        if vid not in tte.index:
            raise ValueError(f"Missing repair record for vehicle {vid}")
        end = float(tte.at[vid, "length_of_study_time_step"])
        repaired = int(tte.at[vid, "in_study_repair"])
        vehicle_df = vehicle_df.sort_values(TIME, kind="stable")

        for cutoff in sample_landmarks(vehicle_df[TIME].to_numpy(float),
                                       args.landmarks_per_vehicle, rng):
            features = build_snapshot_features(vehicle_df, float(cutoff), sensors)
            label = assign_class(end - float(cutoff), repaired)
            row = {
                "vehicle_id": vid,
                "class_label": -1 if label is None else label,
                "binary_label": -1 if label is None else int(label > 0),
                "remaining_time": end - float(cutoff),
                "in_study_repair": repaired,
            }
            row.update(features)
            for c in spec_cols:
                value = specs.at[vid, c] if vid in specs.index else np.nan
                row[c] = str(value) if pd.notna(value) else "MISSING"
            buffer.append(row)

        vehicles += 1
        if len(buffer) >= args.flush_rows:
            flush()
        if vehicles % 1000 == 0:
            rate = vehicles / (time.time() - start)
            remaining = (len(tte) - vehicles) / max(rate, 1e-9) / 60
            print(f"[{time.strftime('%H:%M:%S')}] vehicles {vehicles:,}/{len(tte):,} "
                  f"| landmarks {written + len(buffer):,} | ~{remaining:.0f} min left",
                  flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building landmark features "
          f"({args.landmarks_per_vehicle} per vehicle, seed {args.seed})...", flush=True)

    pending = None
    try:
        for batch in readouts.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            if pending is not None:
                df = pd.concat([pending, df], ignore_index=True)
            last = df[ID].iloc[-1]
            pending = df[df[ID] == last]
            complete = df[df[ID] != last]
            for _, group in complete.groupby(ID, sort=False):
                process(group)
        if pending is not None and len(pending):
            process(pending)
        flush()
    finally:
        if writer is not None:
            writer.close()

    partial.replace(output)

    result = pd.read_parquet(output, columns=META)
    print("\n" + "=" * 70)
    print(f"Landmarks: {len(result):,} from {result[ID].nunique():,} vehicles")
    print(f"Undefined class (censored within 48): "
          f"{(result.class_label < 0).sum():,} "
          f"({(result.class_label < 0).mean():.1%})")
    print("Class counts (defined):")
    print(result.loc[result.class_label >= 0, "class_label"].value_counts().sort_index()
          .to_string())
    print(f"Output: {output}")
    print(f"[{time.strftime('%H:%M:%S')}] STATUS: SUCCESS")


if __name__ == "__main__":
    main()
