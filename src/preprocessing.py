from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw/2024-34-3/data")
INTERIM_DATA_DIR = Path("data/interim")


FILES_TO_CONVERT = {
    "train_operational_readouts": {
        "input": RAW_DATA_DIR / "train_operational_readouts.csv",
        "output": INTERIM_DATA_DIR / "train_operational_readouts.parquet",
    },
    "validation_operational_readouts": {
        "input": RAW_DATA_DIR / "validation_operational_readouts.csv",
        "output": INTERIM_DATA_DIR / "validation_operational_readouts.parquet",
    },
    "test_operational_readouts": {
        "input": RAW_DATA_DIR / "test_operational_readouts.csv",
        "output": INTERIM_DATA_DIR / "test_operational_readouts.parquet",
    },
    "train_specifications": {
        "input": RAW_DATA_DIR / "train_specifications.csv",
        "output": INTERIM_DATA_DIR / "train_specifications.parquet",
    },
    "validation_specifications": {
        "input": RAW_DATA_DIR / "validation_specifications.csv",
        "output": INTERIM_DATA_DIR / "validation_specifications.parquet",
    },
    "test_specifications": {
        "input": RAW_DATA_DIR / "test_specifications.csv",
        "output": INTERIM_DATA_DIR / "test_specifications.parquet",
    },
    "train_tte": {
        "input": RAW_DATA_DIR / "train_tte.csv",
        "output": INTERIM_DATA_DIR / "train_tte.parquet",
    },
    "validation_labels": {
        "input": RAW_DATA_DIR / "validation_labels.csv",
        "output": INTERIM_DATA_DIR / "validation_labels.parquet",
    },
    "test_labels": {
        "input": RAW_DATA_DIR / "test_labels.csv",
        "output": INTERIM_DATA_DIR / "test_labels.parquet",
    },
}


def optimize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by converting numeric columns to smaller dtypes when possible.
    """
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="integer")

        elif pd.api.types.is_float_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], downcast="float")

    return df


def convert_csv_to_parquet(input_path: Path, output_path: Path) -> None:
    """
    Convert a CSV file to Parquet.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nReading CSV: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")

    df = optimize_dataframe_types(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing Parquet: {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"Saved successfully: {output_path}")


def main() -> None:
    print("Starting SCANIA CSV to Parquet preprocessing...")

    for file_name, paths in FILES_TO_CONVERT.items():
        print("=" * 80)
        print(f"Processing: {file_name}")
        convert_csv_to_parquet(paths["input"], paths["output"])

    print("=" * 80)
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()