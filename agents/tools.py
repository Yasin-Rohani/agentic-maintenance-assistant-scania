from pathlib import Path

import pandas as pd

from agents.decision_tool import decide_maintenance_action


VALIDATION_PREDICTIONS_PATH = Path(
    "reports/tables/validation_predictions_lightgbm_baseline.csv"
)

DEFAULT_SCENARIO = "safety_critical"


def load_prediction_table(
    predictions_path: Path = VALIDATION_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """
    Load vehicle-level prediction table.

    Expected columns:
    - vehicle_id
    - true_label_binary
    - failure_probability
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}. "
            "Run src/train_baselines.py first."
        )

    predictions_df = pd.read_csv(predictions_path)

    required_columns = [
        "vehicle_id",
        "true_label_binary",
        "failure_probability",
    ]

    for column in required_columns:
        if column not in predictions_df.columns:
            raise ValueError(f"Missing required column in predictions file: {column}")

    return predictions_df


def get_vehicle_prediction(
    vehicle_id: int,
    predictions_path: Path = VALIDATION_PREDICTIONS_PATH,
) -> dict:
    """
    Retrieve failure prediction for one vehicle.
    """
    predictions_df = load_prediction_table(predictions_path)

    vehicle_rows = predictions_df[predictions_df["vehicle_id"] == vehicle_id]

    if vehicle_rows.empty:
        available_examples = predictions_df["vehicle_id"].head(10).tolist()
        raise ValueError(
            f"Vehicle ID {vehicle_id} not found in prediction table. "
            f"Example available vehicle IDs: {available_examples}"
        )

    row = vehicle_rows.iloc[0]

    return {
        "vehicle_id": int(row["vehicle_id"]),
        "true_label_binary": int(row["true_label_binary"]),
        "failure_probability": float(row["failure_probability"]),
    }


def make_vehicle_maintenance_decision(
    vehicle_id: int,
    scenario_name: str = DEFAULT_SCENARIO,
    predictions_path: Path = VALIDATION_PREDICTIONS_PATH,
) -> dict:
    """
    Main tool used by the maintenance agent.

    Input:
    - vehicle_id
    - cost scenario name

    Output:
    - failure probability
    - selected threshold
    - recommended maintenance action
    """
    prediction = get_vehicle_prediction(
        vehicle_id=vehicle_id,
        predictions_path=predictions_path,
    )

    decision = decide_maintenance_action(
        failure_probability=prediction["failure_probability"],
        scenario_name=scenario_name,
    )

    return {
        "vehicle_id": prediction["vehicle_id"],
        "true_label_binary": prediction["true_label_binary"],
        "failure_probability": decision["failure_probability"],
        "scenario": decision["scenario"],
        "cost_sensitive_threshold": decision["cost_sensitive_threshold"],
        "action_key": decision["action_key"],
        "action_label": decision["action_label"],
        "severity": decision["severity"],
        "explanation": decision["explanation"],
    }


def print_vehicle_decision(result: dict) -> None:
    """Print a readable vehicle-level maintenance decision."""
    print("=" * 80)
    print("Vehicle Maintenance Decision")
    print("=" * 80)
    print(f"Vehicle ID: {result['vehicle_id']}")
    print(f"True label binary: {result['true_label_binary']}")
    print(f"Failure probability: {result['failure_probability']}")
    print(f"Scenario: {result['scenario']}")
    print(f"Cost-sensitive threshold: {result['cost_sensitive_threshold']}")
    print(f"Recommended action: {result['action_label']}")
    print(f"Severity: {result['severity']}")
    print(f"Explanation: {result['explanation']}")


def main() -> None:
    """
    Test tool with a few available vehicle IDs from validation predictions.
    """
    predictions_df = load_prediction_table()

    example_vehicle_ids = predictions_df["vehicle_id"].head(5).tolist()

    print(f"Testing vehicle IDs: {example_vehicle_ids}")

    for vehicle_id in example_vehicle_ids:
        result = make_vehicle_maintenance_decision(
            vehicle_id=int(vehicle_id),
            scenario_name=DEFAULT_SCENARIO,
        )
        print_vehicle_decision(result)


if __name__ == "__main__":
    main()