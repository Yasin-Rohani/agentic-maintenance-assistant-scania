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


def run_cost_evaluation(config: dict) -> pd.DataFrame:
    """Evaluate maintenance decision cost over many thresholds."""
    predictions_path = Path(config["input"]["validation_predictions_path"])
    output_path = Path(config["output"]["cost_evaluation_path"])

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

    print(f"Evaluating {len(thresholds)} thresholds...")

    rows = []

    for threshold in thresholds:
        row = calculate_costs_for_threshold(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            costs=config["costs"],
        )
        rows.append(row)

    cost_df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cost_df.to_csv(output_path, index=False)

    print(f"Saved cost evaluation to: {output_path}")

    return cost_df


def print_best_thresholds(cost_df: pd.DataFrame) -> None:
    """Print best thresholds according to cost and F1."""
    best_cost_row = cost_df.sort_values("total_cost", ascending=True).iloc[0]
    best_f1_row = cost_df.sort_values("f1", ascending=False).iloc[0]

    print("\nBest threshold by minimum total cost:")
    print(best_cost_row)

    print("\nBest threshold by maximum F1:")
    print(best_f1_row)


def main() -> None:
    config = load_config()
    cost_df = run_cost_evaluation(config)
    print_best_thresholds(cost_df)

    print("\nCost evaluation completed successfully.")


if __name__ == "__main__":
    main()