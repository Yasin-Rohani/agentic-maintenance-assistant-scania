from pathlib import Path

import pandas as pd


SCENARIO_SUMMARY_PATH = Path("reports/tables/cost_scenario_summary_lightgbm.csv")


DEFAULT_SCENARIO = "safety_critical"


ACTION_RULES = {
    "no_action": {
        "label": "No action",
        "description": "The vehicle is currently considered low risk. No maintenance action is required.",
    },
    "monitor": {
        "label": "Monitor",
        "description": "The vehicle shows some risk signals. Continue monitoring before scheduling maintenance.",
    },
    "schedule_inspection": {
        "label": "Schedule inspection",
        "description": "The vehicle risk is above the cost-sensitive threshold. A maintenance inspection should be scheduled.",
    },
    "urgent_inspection": {
        "label": "Urgent inspection",
        "description": "The vehicle has high predicted failure risk. It should be inspected urgently.",
    },
    "preventive_replacement": {
        "label": "Preventive replacement / stop operation",
        "description": "The vehicle has very high predicted failure risk. Preventive replacement or temporary stop should be considered.",
    },
}


def load_scenario_summary(path: Path = SCENARIO_SUMMARY_PATH) -> pd.DataFrame:
    """Load cost scenario summary table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario summary file not found: {path}. "
            "Run src/cost_evaluation.py first."
        )

    return pd.read_csv(path)


def get_scenario_threshold(
    scenario_name: str = DEFAULT_SCENARIO,
    summary_path: Path = SCENARIO_SUMMARY_PATH,
) -> float:
    """Return the best cost-sensitive threshold for a selected scenario."""
    summary_df = load_scenario_summary(summary_path)

    if scenario_name not in summary_df["scenario"].values:
        available_scenarios = summary_df["scenario"].tolist()
        raise ValueError(
            f"Scenario '{scenario_name}' not found. "
            f"Available scenarios: {available_scenarios}"
        )

    scenario_row = summary_df[summary_df["scenario"] == scenario_name].iloc[0]

    return float(scenario_row["best_cost_threshold"])


def decide_maintenance_action(
    failure_probability: float,
    scenario_name: str = DEFAULT_SCENARIO,
    summary_path: Path = SCENARIO_SUMMARY_PATH,
) -> dict:
    """
    Convert a failure probability into a maintenance decision.

    The selected scenario threshold comes from cost-sensitive evaluation.
    """
    if failure_probability < 0 or failure_probability > 1:
        raise ValueError("failure_probability must be between 0 and 1.")

    cost_sensitive_threshold = get_scenario_threshold(
        scenario_name=scenario_name,
        summary_path=summary_path,
    )

    if failure_probability >= 0.80:
        action_key = "preventive_replacement"
        severity = "critical"

    elif failure_probability >= 0.50:
        action_key = "urgent_inspection"
        severity = "high"

    elif failure_probability >= cost_sensitive_threshold:
        action_key = "schedule_inspection"
        severity = "medium"

    elif failure_probability >= 0.10:
        action_key = "monitor"
        severity = "low"

    else:
        action_key = "no_action"
        severity = "normal"

    action = ACTION_RULES[action_key]

    return {
        "failure_probability": round(failure_probability, 4),
        "scenario": scenario_name,
        "cost_sensitive_threshold": round(cost_sensitive_threshold, 4),
        "action_key": action_key,
        "action_label": action["label"],
        "severity": severity,
        "explanation": action["description"],
    }


def print_decision(decision: dict) -> None:
    """Print a readable decision output."""
    print("=" * 80)
    print("Maintenance Decision")
    print("=" * 80)
    print(f"Failure probability: {decision['failure_probability']}")
    print(f"Scenario: {decision['scenario']}")
    print(f"Cost-sensitive threshold: {decision['cost_sensitive_threshold']}")
    print(f"Recommended action: {decision['action_label']}")
    print(f"Severity: {decision['severity']}")
    print(f"Explanation: {decision['explanation']}")


def main() -> None:
    """
    Test examples for the decision tool.
    """
    example_probabilities = [0.03, 0.15, 0.25, 0.55, 0.85]

    for probability in example_probabilities:
        decision = decide_maintenance_action(
            failure_probability=probability,
            scenario_name=DEFAULT_SCENARIO,
        )
        print_decision(decision)


if __name__ == "__main__":
    main()