from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from agents.maintenance_agent import build_maintenance_report
from agents.behavior_tool import compare_vehicle_to_fleet

GENERATED_REPORTS_DIR = Path("reports/generated_reports")
FIGURES_DIR = Path("reports/figures")
DEFAULT_SCENARIO = "safety_critical"


def generate_risk_threshold_chart(report: dict, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generate a simple chart comparing failure probability against
    the cost-sensitive threshold.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vehicle_id = report["vehicle_id"]
    probability = report["failure_probability"]
    threshold = report["cost_sensitive_threshold"]

    output_path = output_dir / f"vehicle_{vehicle_id}_risk_threshold.png"

    labels = ["Failure probability", "Cost-sensitive threshold"]
    values = [probability, threshold]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Vehicle {vehicle_id}: Risk vs Threshold")

    for index, value in enumerate(values):
        plt.text(index, value + 0.02, f"{value:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def generate_abnormal_features_chart(
    vehicle_id: int,
    top_n: int = 10,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Generate a horizontal bar chart for the top abnormal vehicle features.
    The chart uses absolute standardized difference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = compare_vehicle_to_fleet(
        vehicle_id=vehicle_id,
        top_n=top_n,
    )

    output_path = output_dir / f"vehicle_{vehicle_id}_abnormal_features.png"

    if comparison_df.empty:
        return output_path

    chart_df = comparison_df.sort_values(
        "abs_standardized_difference",
        ascending=True,
    )

    plt.figure(figsize=(9, 6))
    plt.barh(
        chart_df["feature"],
        chart_df["abs_standardized_difference"],
    )
    plt.xlabel("Absolute Standardized Difference")
    plt.ylabel("Feature")
    plt.title(f"Vehicle {vehicle_id}: Top Abnormal Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

def build_behavior_comparison_markdown(vehicle_id: int, top_n: int = 10) -> str:
    """
    Build a Markdown table with the top abnormal vehicle features
    compared with fleet-level averages.
    """
    comparison_df = compare_vehicle_to_fleet(
        vehicle_id=vehicle_id,
        top_n=top_n,
    )

    if comparison_df.empty:
        return "No abnormal feature comparison could be generated."

    markdown_lines = [
        "| Feature | Vehicle Value | Fleet Mean | Raw Difference | Standardized Difference |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in comparison_df.iterrows():
        markdown_lines.append(
            "| "
            f"{row['feature']} | "
            f"{row['vehicle_value']:.4f} | "
            f"{row['fleet_mean']:.4f} | "
            f"{row['raw_difference']:.4f} | "
            f"{row['standardized_difference']:.4f} |"
        )

    return "\n".join(markdown_lines)

def build_markdown_report(
    report: dict,
    risk_chart_path: Path,
    abnormal_chart_path: Path,
    ) -> str:
    """
    Convert a maintenance agent report dictionary into Markdown text.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    relative_risk_chart_path = Path("..") / "figures" / risk_chart_path.name
    relative_abnormal_chart_path = Path("..") / "figures" / abnormal_chart_path.name

    behavior_comparison_markdown = build_behavior_comparison_markdown(
    vehicle_id=report["vehicle_id"],
    top_n=10,
    )

    markdown = f"""# Maintenance Agent Report

## Report Metadata

| Field | Value |
|---|---|
| Generated at | {generated_at} |
| Vehicle ID | {report["vehicle_id"]} |
| Scenario | {report["scenario"]} |

---

## Maintenance Recommendation

| Field | Value |
|---|---|
| Failure probability | {report["failure_probability"]} |
| Cost-sensitive threshold | {report["cost_sensitive_threshold"]} |
| Recommended action | {report["recommended_action"]} |
| Severity | {report["severity"]} |
| True label binary | {report["true_label_binary"]} |

---

## Risk vs Threshold Chart

![Risk vs Threshold]({relative_risk_chart_path.as_posix()})

---

## Top Abnormal Vehicle Features

The table below compares this vehicle against fleet-level averages.  
Features with larger absolute standardized differences are more unusual compared with the validation fleet.

{behavior_comparison_markdown}

---

## Abnormal Features Chart

![Top Abnormal Features]({relative_abnormal_chart_path.as_posix()})

---

## Agent Reasoning

{report["agent_reasoning"]}

---

## Technical Explanation

{report["technical_explanation"]}

---

## Interpretation

The maintenance recommendation is generated by combining:

1. The predicted failure probability from the trained LightGBM baseline model.
2. The cost-sensitive threshold selected under the `{report["scenario"]}` scenario.
3. A rule-based maintenance action layer that maps risk levels to operational actions.

This report is intended as a decision-support output, not as an automatic replacement for human maintenance judgment.

---

## Next Development Steps

Future versions of this report should include:

- vehicle time-series trend charts
- comparison against fleet-level baseline behavior
- comparison against similar vehicles
- SHAP-based risk factor explanation
- cost comparison between action alternatives
- documentation-aware RAG explanation
"""
    return markdown


def save_markdown_report(
    vehicle_id: int,
    scenario_name: str = DEFAULT_SCENARIO,
    output_dir: Path = GENERATED_REPORTS_DIR,
) -> Path:
    """
    Generate and save a Markdown maintenance report for one vehicle.
    """
    report = build_maintenance_report(
        vehicle_id=vehicle_id,
        scenario_name=scenario_name,
    )

    risk_chart_path = generate_risk_threshold_chart(report)

    abnormal_chart_path = generate_abnormal_features_chart(
       vehicle_id=vehicle_id,
    top_n=10,
    )

    markdown_report = build_markdown_report(
    report=report,
    risk_chart_path=risk_chart_path,
    abnormal_chart_path=abnormal_chart_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"vehicle_{vehicle_id}_report.md"

    output_path.write_text(markdown_report, encoding="utf-8")

    return output_path


def main() -> None:
    """
    Generate example reports for selected validation vehicles.
    """
    example_vehicle_ids = [10, 16, 18, 23, 45]

    for vehicle_id in example_vehicle_ids:
        output_path = save_markdown_report(
            vehicle_id=vehicle_id,
            scenario_name=DEFAULT_SCENARIO,
        )

        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()