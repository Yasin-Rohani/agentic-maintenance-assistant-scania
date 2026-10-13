from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.orchestrator_agent import run_orchestrator


PREDICTIONS_PATH = Path("reports/tables/validation_predictions_lightgbm_baseline.csv")
GENERATED_REPORTS_DIR = Path("reports/generated_reports")
FIGURES_DIR = Path("reports/figures")


SCENARIOS = [
    "safety_critical",
    "balanced_cost",
    "inspection_expensive",
    "recall_focused",
]


def load_available_vehicle_ids() -> list[int]:
    """
    Load available validation vehicle IDs from prediction table.
    """
    if not PREDICTIONS_PATH.exists():
        st.error(
            "Prediction file not found. Run src/train_baselines.py before using the dashboard."
        )
        return []

    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    if "vehicle_id" not in predictions_df.columns:
        st.error("Prediction file does not contain vehicle_id column.")
        return []

    return predictions_df["vehicle_id"].astype(int).tolist()


def display_metric_cards(dashboard_state: dict) -> None:
    """
    Display main dashboard metric cards.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Failure Probability",
            value=dashboard_state.get("failure_probability"),
        )

    with col2:
        st.metric(
            label="Cost Threshold",
            value=dashboard_state.get("cost_sensitive_threshold"),
        )

    with col3:
        st.metric(
            label="Action",
            value=dashboard_state.get("recommended_action"),
        )

    with col4:
        st.metric(
            label="Severity",
            value=dashboard_state.get("severity"),
        )


def display_top_abnormal_features(dashboard_state: dict) -> None:
    """
    Display top abnormal features.
    """
    top_features = dashboard_state.get("top_abnormal_features", [])

    if not top_features:
        st.warning("No abnormal features available.")
        return

    feature_df = pd.DataFrame(
        {
            "rank": list(range(1, len(top_features) + 1)),
            "feature": top_features,
        }
    )

    st.dataframe(feature_df, use_container_width=True)


def display_report(vehicle_id: int) -> None:
    """
    Display generated Markdown report.
    """
    report_path = GENERATED_REPORTS_DIR / f"vehicle_{vehicle_id}_report.md"

    if not report_path.exists():
        st.warning(f"Report not found: {report_path}")
        return

    report_text = report_path.read_text(encoding="utf-8")
    st.markdown(report_text)


def display_charts(vehicle_id: int) -> None:
    """
    Display generated charts.
    """
    risk_chart_path = FIGURES_DIR / f"vehicle_{vehicle_id}_risk_threshold.png"
    abnormal_chart_path = FIGURES_DIR / f"vehicle_{vehicle_id}_abnormal_features.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk vs Threshold")
        if risk_chart_path.exists():
            st.image(str(risk_chart_path), use_container_width=True)
        else:
            st.warning(f"Chart not found: {risk_chart_path}")

    with col2:
        st.subheader("Top Abnormal Features")
        if abnormal_chart_path.exists():
            st.image(str(abnormal_chart_path), use_container_width=True)
        else:
            st.warning(f"Chart not found: {abnormal_chart_path}")


def main() -> None:
    st.set_page_config(
        page_title="Agentic Maintenance Assistant",
        layout="wide",
    )

    st.title("Agentic Maintenance Assistant")
    st.caption(
        "Cost-sensitive predictive maintenance decision-support system "
        "for SCANIA Component X fleet data."
    )

    available_vehicle_ids = load_available_vehicle_ids()

    if not available_vehicle_ids:
        st.stop()

    st.sidebar.header("Input")

    vehicle_id = st.sidebar.selectbox(
        "Vehicle ID",
        available_vehicle_ids,
        index=0,
    )

    scenario = st.sidebar.selectbox(
        "Cost Scenario",
        SCENARIOS,
        index=0,
    )

    run_button = st.sidebar.button("Run Agentic Analysis")

    st.sidebar.markdown("---")
    st.sidebar.write("Selected vehicle:", vehicle_id)
    st.sidebar.write("Selected scenario:", scenario)

    if run_button:
        with st.spinner("Running agentic maintenance workflow..."):
            state = run_orchestrator(
                vehicle_id=int(vehicle_id),
                scenario_name=scenario,
            )

        st.success("Agentic workflow completed.")

        dashboard_state = {
            "vehicle_id": state["vehicle_id"],
            "scenario": state["scenario"],
            "status": state["status"],
            "failure_probability": state["decision"]["failure_probability"],
            "cost_sensitive_threshold": state["decision"]["cost_sensitive_threshold"],
            "recommended_action": state["decision"]["action_label"],
            "severity": state["decision"]["severity"],
            "true_label_binary": state["decision"]["true_label_binary"],
            "top_abnormal_features": state["behavior_summary"][
                "top_abnormal_features"
            ],
            "report_path": state["report_path"],
            "state_path": state.get("state_path"),
            "dashboard_state_path": state.get("dashboard_state_path"),
        }

        st.subheader("Maintenance Decision Summary")
        display_metric_cards(dashboard_state)

        st.markdown("---")

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("Workflow Status")
            st.write("Status:", dashboard_state["status"])
            st.write("Vehicle ID:", dashboard_state["vehicle_id"])
            st.write("Scenario:", dashboard_state["scenario"])
            st.write("True label binary:", dashboard_state["true_label_binary"])
            st.write("Report path:", dashboard_state["report_path"])
            st.write("Shared state path:", dashboard_state["state_path"])
            st.write("Dashboard state path:", dashboard_state["dashboard_state_path"])

        with right_col:
            st.subheader("Top Abnormal Features")
            display_top_abnormal_features(dashboard_state)

        st.markdown("---")

        display_charts(vehicle_id=int(vehicle_id))

        st.markdown("---")

        with st.expander("Show Full Generated Markdown Report"):
            display_report(vehicle_id=int(vehicle_id))

    else:
        st.info("Select a vehicle and scenario, then click 'Run Agentic Analysis'.")


if __name__ == "__main__":
    main()