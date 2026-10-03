from agents.tools import make_vehicle_maintenance_decision


DEFAULT_SCENARIO = "safety_critical"


def generate_agent_reasoning(decision: dict) -> str:
    """
    Generate a human-readable explanation for the maintenance decision.
    """
    probability = decision["failure_probability"]
    threshold = decision["cost_sensitive_threshold"]
    action = decision["action_label"]
    severity = decision["severity"]

    if decision["action_key"] == "no_action":
        reason = (
            f"The predicted failure probability is {probability}, which is below "
            f"the monitoring boundary and below the cost-sensitive threshold "
            f"of {threshold}. No immediate maintenance action is recommended."
        )

    elif decision["action_key"] == "monitor":
        reason = (
            f"The predicted failure probability is {probability}. This is above "
            f"the low-risk monitoring boundary but still below the cost-sensitive "
            f"inspection threshold of {threshold}. The vehicle should be monitored, "
            f"but inspection is not yet recommended under the selected scenario."
        )

    elif decision["action_key"] == "schedule_inspection":
        reason = (
            f"The predicted failure probability is {probability}, which is above "
            f"the cost-sensitive threshold of {threshold}. Under the selected "
            f"scenario, the expected cost of ignoring this vehicle is higher than "
            f"scheduling an inspection."
        )

    elif decision["action_key"] == "urgent_inspection":
        reason = (
            f"The predicted failure probability is {probability}, which is well above "
            f"the cost-sensitive threshold of {threshold}. This indicates high risk, "
            f"so urgent inspection is recommended."
        )

    elif decision["action_key"] == "preventive_replacement":
        reason = (
            f"The predicted failure probability is {probability}, which is extremely "
            f"high and above the critical-risk boundary. Preventive replacement or "
            f"temporary stop of operation should be considered."
        )

    else:
        reason = (
            f"The predicted failure probability is {probability}. "
            f"The recommended action is {action} with severity level {severity}."
        )

    return reason


def build_maintenance_report(vehicle_id: int, scenario_name: str = DEFAULT_SCENARIO) -> dict:
    """
    Build a structured maintenance agent report for one vehicle.
    """
    decision = make_vehicle_maintenance_decision(
        vehicle_id=vehicle_id,
        scenario_name=scenario_name,
    )

    agent_reasoning = generate_agent_reasoning(decision)

    report = {
        "vehicle_id": decision["vehicle_id"],
        "scenario": decision["scenario"],
        "true_label_binary": decision["true_label_binary"],
        "failure_probability": decision["failure_probability"],
        "cost_sensitive_threshold": decision["cost_sensitive_threshold"],
        "recommended_action": decision["action_label"],
        "severity": decision["severity"],
        "technical_explanation": decision["explanation"],
        "agent_reasoning": agent_reasoning,
    }

    return report


def print_maintenance_report(report: dict) -> None:
    """
    Print a readable maintenance report.
    """
    print("=" * 80)
    print("Maintenance Agent Report")
    print("=" * 80)
    print(f"Vehicle ID: {report['vehicle_id']}")
    print(f"Scenario: {report['scenario']}")
    print(f"True label binary: {report['true_label_binary']}")
    print(f"Failure probability: {report['failure_probability']}")
    print(f"Cost-sensitive threshold: {report['cost_sensitive_threshold']}")
    print(f"Recommended action: {report['recommended_action']}")
    print(f"Severity: {report['severity']}")
    print("-" * 80)
    print("Agent reasoning:")
    print(report["agent_reasoning"])
    print("-" * 80)
    print("Technical explanation:")
    print(report["technical_explanation"])


def main() -> None:
    """
    Test the maintenance agent on example validation vehicles.
    """
    example_vehicle_ids = [10, 16, 18, 23, 45]

    for vehicle_id in example_vehicle_ids:
        report = build_maintenance_report(
            vehicle_id=vehicle_id,
            scenario_name=DEFAULT_SCENARIO,
        )
        print_maintenance_report(report)


if __name__ == "__main__":
    main()