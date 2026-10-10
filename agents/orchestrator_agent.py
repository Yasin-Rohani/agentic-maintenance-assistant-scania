import argparse
from pathlib import Path
from datetime import datetime

from agents.tools import make_vehicle_maintenance_decision
from agents.behavior_tool import compare_vehicle_to_fleet
from agents.report_generator import save_markdown_report


DEFAULT_SCENARIO = "safety_critical"


def initialize_task_state(vehicle_id: int, scenario_name: str) -> dict:
    """
    Initialize shared task state for the agentic workflow.
    """
    return {
        "objective": f"Analyze maintenance risk for vehicle {vehicle_id}",
        "vehicle_id": vehicle_id,
        "scenario": scenario_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": [],
        "decision": None,
        "behavior_summary": None,
        "report_path": None,
        "status": "initialized",
        "errors": [],
    }


def add_step(state: dict, step_name: str, status: str, details: str) -> None:
    """
    Add a workflow step to the shared task state.
    """
    state["steps"].append(
        {
            "step_name": step_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def run_decision_step(state: dict) -> dict:
    """
    Run maintenance decision tool.
    """
    try:
        decision = make_vehicle_maintenance_decision(
            vehicle_id=state["vehicle_id"],
            scenario_name=state["scenario"],
        )

        state["decision"] = decision

        add_step(
            state=state,
            step_name="maintenance_decision",
            status="completed",
            details=(
                f"Decision generated: {decision['action_label']} "
                f"with severity={decision['severity']}."
            ),
        )

    except Exception as error:
        state["errors"].append(str(error))
        add_step(
            state=state,
            step_name="maintenance_decision",
            status="failed",
            details=str(error),
        )

    return state


def run_behavior_analysis_step(state: dict, top_n: int = 10) -> dict:
    """
    Run vehicle behavior comparison tool.
    """
    try:
        comparison_df = compare_vehicle_to_fleet(
            vehicle_id=state["vehicle_id"],
            top_n=top_n,
        )

        top_features = comparison_df["feature"].head(top_n).tolist()

        state["behavior_summary"] = {
            "top_n": top_n,
            "top_abnormal_features": top_features,
        }

        add_step(
            state=state,
            step_name="behavior_analysis",
            status="completed",
            details=f"Top abnormal features identified: {top_features[:3]}",
        )

    except Exception as error:
        state["errors"].append(str(error))
        add_step(
            state=state,
            step_name="behavior_analysis",
            status="failed",
            details=str(error),
        )

    return state


def run_report_generation_step(state: dict) -> dict:
    """
    Generate maintenance report.
    """
    try:
        report_path = save_markdown_report(
            vehicle_id=state["vehicle_id"],
            scenario_name=state["scenario"],
        )

        state["report_path"] = str(report_path)

        add_step(
            state=state,
            step_name="report_generation",
            status="completed",
            details=f"Report generated at: {report_path}",
        )

    except Exception as error:
        state["errors"].append(str(error))
        add_step(
            state=state,
            step_name="report_generation",
            status="failed",
            details=str(error),
        )

    return state


def evaluate_progress(state: dict) -> dict:
    """
    Evaluate whether the workflow achieved the objective.
    """
    required_outputs = [
        state["decision"] is not None,
        state["behavior_summary"] is not None,
        state["report_path"] is not None,
        len(state["errors"]) == 0,
    ]

    if all(required_outputs):
        state["status"] = "completed"
        details = "Objective completed successfully."

    else:
        state["status"] = "incomplete"
        details = "Objective not fully completed. Check errors and failed steps."

    add_step(
        state=state,
        step_name="evaluate_progress",
        status=state["status"],
        details=details,
    )

    return state


def run_orchestrator(vehicle_id: int, scenario_name: str = DEFAULT_SCENARIO) -> dict:
    """
    Run the full agentic maintenance workflow.
    """
    state = initialize_task_state(
        vehicle_id=vehicle_id,
        scenario_name=scenario_name,
    )

    add_step(
        state=state,
        step_name="objective_received",
        status="completed",
        details=state["objective"],
    )

    state = run_decision_step(state)
    state = run_behavior_analysis_step(state)
    state = run_report_generation_step(state)
    state = evaluate_progress(state)

    return state


def print_orchestrator_state(state: dict) -> None:
    """
    Print readable orchestrator output.
    """
    print("=" * 80)
    print("Agentic Maintenance Orchestrator")
    print("=" * 80)
    print(f"Objective: {state['objective']}")
    print(f"Vehicle ID: {state['vehicle_id']}")
    print(f"Scenario: {state['scenario']}")
    print(f"Status: {state['status']}")

    print("\nWorkflow steps:")
    for step in state["steps"]:
        print(
            f"- [{step['status']}] {step['step_name']}: "
            f"{step['details']}"
        )

    if state["decision"] is not None:
        print("\nDecision summary:")
        print(f"- Failure probability: {state['decision']['failure_probability']}")
        print(f"- Threshold: {state['decision']['cost_sensitive_threshold']}")
        print(f"- Recommended action: {state['decision']['action_label']}")
        print(f"- Severity: {state['decision']['severity']}")

    if state["behavior_summary"] is not None:
        print("\nTop abnormal features:")
        for feature in state["behavior_summary"]["top_abnormal_features"][:5]:
            print(f"- {feature}")

    if state["report_path"] is not None:
        print(f"\nGenerated report: {state['report_path']}")

    if state["errors"]:
        print("\nErrors:")
        for error in state["errors"]:
            print(f"- {error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agentic maintenance orchestrator."
    )

    parser.add_argument(
        "--vehicle_id",
        type=int,
        default=45,
        help="Vehicle ID to analyze.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        help="Cost scenario name.",
    )

    args = parser.parse_args()

    state = run_orchestrator(
        vehicle_id=args.vehicle_id,
        scenario_name=args.scenario,
    )

    print_orchestrator_state(state)


if __name__ == "__main__":
    main()