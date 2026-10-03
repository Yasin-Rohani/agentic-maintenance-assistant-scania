from pathlib import Path

import pandas as pd


VALIDATION_FEATURES_PATH = Path("data/processed/validation_features.parquet")

ID_COLUMN = "vehicle_id"
TARGET_COLUMN = "class_label"


def load_validation_features(
    features_path: Path = VALIDATION_FEATURES_PATH,
) -> pd.DataFrame:
    """
    Load validation feature table.
    """
    if not features_path.exists():
        raise FileNotFoundError(
            f"Validation features file not found: {features_path}. "
            "Run src/temporal_summarization.py first."
        )

    return pd.read_parquet(features_path)


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric feature columns, excluding identifiers and labels.
    """
    excluded_columns = [ID_COLUMN, TARGET_COLUMN]

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    feature_columns = [
        column for column in numeric_columns if column not in excluded_columns
    ]

    return feature_columns


def compare_vehicle_to_fleet(
    vehicle_id: int,
    top_n: int = 10,
    features_path: Path = VALIDATION_FEATURES_PATH,
) -> pd.DataFrame:
    """
    Compare one vehicle's numeric feature values against fleet-level averages.

    Returns the top features with the largest absolute standardized difference.
    """
    df = load_validation_features(features_path)

    if vehicle_id not in df[ID_COLUMN].values:
        available_examples = df[ID_COLUMN].head(10).tolist()
        raise ValueError(
            f"Vehicle ID {vehicle_id} not found. "
            f"Example available vehicle IDs: {available_examples}"
        )

    feature_columns = get_numeric_feature_columns(df)

    fleet_mean = df[feature_columns].mean()
    fleet_std = df[feature_columns].std().replace(0, pd.NA)

    vehicle_row = df[df[ID_COLUMN] == vehicle_id].iloc[0]

    comparison_rows = []

    for feature in feature_columns:
        vehicle_value = vehicle_row[feature]
        mean_value = fleet_mean[feature]
        std_value = fleet_std[feature]

        if pd.isna(vehicle_value) or pd.isna(mean_value) or pd.isna(std_value):
            continue

        raw_difference = vehicle_value - mean_value
        standardized_difference = raw_difference / std_value

        comparison_rows.append(
            {
                "feature": feature,
                "vehicle_value": vehicle_value,
                "fleet_mean": mean_value,
                "fleet_std": std_value,
                "raw_difference": raw_difference,
                "standardized_difference": standardized_difference,
                "abs_standardized_difference": abs(standardized_difference),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df = comparison_df.sort_values(
        "abs_standardized_difference",
        ascending=False,
    ).head(top_n)

    return comparison_df


def print_behavior_comparison(comparison_df: pd.DataFrame, vehicle_id: int) -> None:
    """
    Print readable behavior comparison output.
    """
    print("=" * 80)
    print(f"Vehicle Behavior Comparison: vehicle_id={vehicle_id}")
    print("=" * 80)

    columns_to_print = [
        "feature",
        "vehicle_value",
        "fleet_mean",
        "raw_difference",
        "standardized_difference",
    ]

    print(comparison_df[columns_to_print])


def main() -> None:
    """
    Test behavior comparison with example vehicles.
    """
    example_vehicle_ids = [10, 16, 18, 23, 45]

    for vehicle_id in example_vehicle_ids:
        comparison_df = compare_vehicle_to_fleet(
            vehicle_id=vehicle_id,
            top_n=10,
        )
        print_behavior_comparison(
            comparison_df=comparison_df,
            vehicle_id=vehicle_id,
        )


if __name__ == "__main__":
    main()