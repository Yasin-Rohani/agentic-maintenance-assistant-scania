from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix


CONFIG_PATH = Path("configs/cost_config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load cost evaluation configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_costs_for_threshold(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
    costs: dict,
    scenario_name: str,
) -> dict:
    """
    Calculate confusion matrix values and total cost for one threshold.
    """
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_cost = (
        tn * costs["true_negative"]
        + fp * costs["false_positive"]
        + fn * costs["false_negative"]
        + tp * costs["true_positive"]
    )

    average_cost_per_vehicle = total_cost / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "scenario": scenario_name,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_cost": total_cost,
        "average_cost_per_vehicle": average_cost_per_vehicle,
    }


def build_threshold_grid(config: dict) -> list[float]:
    """Build threshold grid from config."""
    start = config["threshold_grid"]["start"]
    stop = config["threshold_grid"]["stop"]
    step = config["threshold_grid"]["step"]

    thresholds = []
    current = start

    while current <= stop:
        thresholds.append(round(current, 4))
        current += step

    return thresholds


def calculate_no_action_baseline(y_true: pd.Series, costs: dict) -> dict:
    """
    Calculate cost if the system never predicts failure.

    This means:
    - all normal vehicles become true negatives
    - all failure vehicles become false negatives
    """
    tn = int((y_true == 0).sum())
    fp = 0
    fn = int((y_true == 1).sum())
    tp = 0

    total_cost = (
        tn * costs["true_negative"]
        + fp * costs["false_positive"]
        + fn * costs["false_negative"]
        + tp * costs["true_positive"]
    )

    average_cost_per_vehicle = total_cost / len(y_true)

    return {
        "no_action_total_cost": total_cost,
        "no_action_average_cost_per_vehicle": average_cost_per_vehicle,
        "no_action_tn": tn,
        "no_action_fp": fp,
        "no_action_fn": fn,
        "no_action_tp": tp,
    }


def run_cost_evaluation(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate maintenance decision cost over many thresholds and scenarios."""
    predictions_path = Path(config["input"]["validation_predictions_path"])
    cost_output_path = Path(config["output"]["cost_evaluation_path"])
    summary_output_path = Path(config["output"]["scenario_summary_path"])

    print(f"Reading validation predictions: {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)

    required_columns = [
        "true_label_binary",
        "failure_probability",
    ]

    for column in required_columns:
        if column not in predictions_df.columns:
            raise ValueError(f"Missing required column: {column}")

    y_true = predictions_df["true_label_binary"]
    y_score = predictions_df["failure_probability"]

    thresholds = build_threshold_grid(config)

    all_rows = []
    summary_rows = []

    for scenario_name, costs in config["scenarios"].items():
        print("=" * 80)
        print(f"Evaluating cost scenario: {scenario_name}")
        print(f"Costs: {costs}")

        no_action = calculate_no_action_baseline(y_true=y_true, costs=costs)

        scenario_rows = []

        for threshold in thresholds:
            row = calculate_costs_for_threshold(
                y_true=y_true,
                y_score=y_score,
                threshold=threshold,
                costs=costs,
                scenario_name=scenario_name,
            )
            scenario_rows.append(row)

        scenario_df = pd.DataFrame(scenario_rows)

        best_cost_row = scenario_df.sort_values("total_cost", ascending=True).iloc[0]
        best_f1_row = scenario_df.sort_values("f1", ascending=False).iloc[0]

        cost_saving_vs_no_action = (
            no_action["no_action_total_cost"] - best_cost_row["total_cost"]
        )

        cost_saving_percentage = (
            cost_saving_vs_no_action / no_action["no_action_total_cost"] * 100
            if no_action["no_action_total_cost"] > 0
            else 0
        )

        summary_rows.append(
            {
                "scenario": scenario_name,
                "best_cost_threshold": best_cost_row["threshold"],
                "best_cost_total_cost": best_cost_row["total_cost"],
                "best_cost_average_cost_per_vehicle": best_cost_row[
                    "average_cost_per_vehicle"
                ],
                "best_cost_precision": best_cost_row["precision"],
                "best_cost_recall": best_cost_row["recall"],
                "best_cost_f1": best_cost_row["f1"],
                "best_cost_tn": best_cost_row["tn"],
                "best_cost_fp": best_cost_row["fp"],
                "best_cost_fn": best_cost_row["fn"],
                "best_cost_tp": best_cost_row["tp"],
                "best_f1_threshold": best_f1_row["threshold"],
                "best_f1": best_f1_row["f1"],
                "best_f1_precision": best_f1_row["precision"],
                "best_f1_recall": best_f1_row["recall"],
                "no_action_total_cost": no_action["no_action_total_cost"],
                "cost_saving_vs_no_action": cost_saving_vs_no_action,
                "cost_saving_percentage": cost_saving_percentage,
            }
        )

        all_rows.extend(scenario_rows)

    cost_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    cost_output_path.parent.mkdir(parents=True, exist_ok=True)

    cost_df.to_csv(cost_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("=" * 80)
    print(f"Saved full cost evaluation to: {cost_output_path}")
    print(f"Saved scenario summary to: {summary_output_path}")

    return cost_df, summary_df


def print_scenario_summary(summary_df: pd.DataFrame) -> None:
    """Print compact scenario summary."""
    columns_to_print = [
        "scenario",
        "best_cost_threshold",
        "best_cost_total_cost",
        "best_cost_precision",
        "best_cost_recall",
        "best_cost_f1",
        "no_action_total_cost",
        "cost_saving_vs_no_action",
        "cost_saving_percentage",
    ]

    print("\nCost scenario summary:")
    print(summary_df[columns_to_print])


def main() -> None:
    config = load_config()
    _, summary_df = run_cost_evaluation(config)
    print_scenario_summary(summary_df)

    print("\nCost evaluation completed successfully.")


if __name__ == "__main__":
    main()