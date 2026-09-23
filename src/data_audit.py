from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw/2024-34-3/data")
REPORTS_TABLES_DIR = Path("reports/tables")


DATA_FILES = {
    "train_operational_readouts": RAW_DATA_DIR / "train_operational_readouts.csv",
    "validation_operational_readouts": RAW_DATA_DIR / "validation_operational_readouts.csv",
    "test_operational_readouts": RAW_DATA_DIR / "test_operational_readouts.csv",
    "train_specifications": RAW_DATA_DIR / "train_specifications.csv",
    "validation_specifications": RAW_DATA_DIR / "validation_specifications.csv",
    "test_specifications": RAW_DATA_DIR / "test_specifications.csv",
    "train_tte": RAW_DATA_DIR / "train_tte.csv",
    "validation_labels": RAW_DATA_DIR / "validation_labels.csv",
    "test_labels": RAW_DATA_DIR / "test_labels.csv",
}


def count_csv_rows(file_path: Path) -> int:
    """
    Count CSV rows without loading the full file into memory.
    The header row is excluded.
    """
    with file_path.open("r", encoding="utf-8") as file:
        row_count = sum(1 for _ in file)

    return max(row_count - 1, 0)


def get_file_size_mb(file_path: Path) -> float:
    """Return file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def read_columns(file_path: Path) -> list[str]:
    """Read only the header row to get column names."""
    return pd.read_csv(file_path, nrows=0).columns.tolist()


def build_file_summary() -> pd.DataFrame:
    """Build a summary table for all SCANIA dataset CSV files."""
    rows = []

    for dataset_name, file_path in DATA_FILES.items():
        if not file_path.exists():
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "file_path": str(file_path),
                    "exists": False,
                    "size_mb": None,
                    "num_rows": None,
                    "num_columns": None,
                    "columns": None,
                }
            )
            continue

        columns = read_columns(file_path)

        rows.append(
            {
                "dataset_name": dataset_name,
                "file_path": str(file_path),
                "exists": True,
                "size_mb": round(get_file_size_mb(file_path), 2),
                "num_rows": count_csv_rows(file_path),
                "num_columns": len(columns),
                "columns": ", ".join(columns),
            }
        )

    return pd.DataFrame(rows)


def summarize_labels() -> dict[str, pd.DataFrame]:
    """
    Summarize target-related columns only:
    - train_tte: in_study_repair
    - validation_labels: class_label
    - test_labels: class_label
    """
    summaries = {}

    label_columns = {
        "train_tte": ["in_study_repair"],
        "validation_labels": ["class_label"],
        "test_labels": ["class_label"],
    }

    for name, columns_to_summarize in label_columns.items():
        file_path = DATA_FILES[name]

        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)

        summary_rows = []

        for column in columns_to_summarize:
            if column not in df.columns:
                continue

            value_counts = (
                df[column]
                .value_counts(dropna=False)
                .reset_index()
            )
            value_counts.columns = ["value", "count"]
            value_counts["dataset_name"] = name
            value_counts["column"] = column
            value_counts["percentage"] = (
                value_counts["count"] / value_counts["count"].sum() * 100
            ).round(2)

            summary_rows.append(value_counts)

        if summary_rows:
            summaries[name] = pd.concat(summary_rows, ignore_index=True)

    return summaries


def main() -> None:
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building SCANIA dataset file summary...")

    file_summary = build_file_summary()
    summary_output_path = REPORTS_TABLES_DIR / "dataset_file_summary.csv"
    file_summary.to_csv(summary_output_path, index=False)

    print("\nDataset file summary:")
    print(file_summary[["dataset_name", "exists", "size_mb", "num_rows", "num_columns"]])

    print(f"\nSaved file summary to: {summary_output_path}")

    print("\nBuilding label summaries...")

    label_summaries = summarize_labels()

    for name, summary_df in label_summaries.items():
        output_path = REPORTS_TABLES_DIR / f"{name}_summary.csv"
        summary_df.to_csv(output_path, index=False)

        print(f"\n{name} summary:")
        print(summary_df.head(20))
        print(f"Saved to: {output_path}")

    print("\nData audit completed successfully.")


if __name__ == "__main__":
    main()