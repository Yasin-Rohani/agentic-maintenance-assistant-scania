# EAAI Experiment Plan

## Target Journal

**Engineering Applications of Artificial Intelligence**

## Working Title

**A Cost-Sensitive Agentic AI Framework for Explainable Predictive Maintenance in Heavy-Duty Vehicle Fleets**

Alternative title:

**Agentic Orchestration for Cost-Sensitive and Explainable Predictive Maintenance on the SCANIA Component X Fleet Dataset**

---

## 1. Research Motivation

Predictive maintenance systems are often evaluated only as classification models.  
However, in real engineering maintenance settings, the final objective is not only to predict failures, but also to support operational decisions under cost, uncertainty, and explainability constraints.

This project proposes an agentic AI decision-support framework that combines:

- failure risk prediction
- cost-sensitive threshold optimization
- scenario-based maintenance decisions
- fleet-level behavioral deviation analysis
- automated diagnostic report generation
- shared task state tracking for agentic workflows

The target application is predictive maintenance for heavy-duty vehicle fleets using the SCANIA Component X dataset.

---

## 2. Research Questions

### RQ1 — Predictive Modeling

Can temporal features extracted from multivariate vehicle operational readouts support failure risk prediction in a real-world fleet maintenance dataset?

### RQ2 — Cost-Sensitive Decision Making

Can cost-sensitive threshold optimization reduce maintenance decision cost compared with a no-action baseline and fixed-threshold prediction?

### RQ3 — Agentic Orchestration

Can an agentic workflow combine prediction, cost evaluation, behavioral explanation, and report generation into a structured maintenance decision-support process?

### RQ4 — Explainability

Can fleet-level behavioral deviation analysis and model explainability provide actionable explanations for individual vehicle recommendations?

---

## 3. Proposed Contributions

### Contribution 1 — Agentic Maintenance Framework

A tool-mediated agentic workflow that decomposes a maintenance objective into multiple steps:

1. risk prediction
2. cost-sensitive decision making
3. behavior analysis
4. report generation
5. shared task state tracking
6. progress evaluation

### Contribution 2 — Multi-Scenario Cost-Sensitive Decision Layer

A cost-sensitive threshold evaluation method that supports multiple operational maintenance policies:

- balanced cost
- safety-critical
- inspection-expensive
- recall-focused

### Contribution 3 — Behavior-Aware Vehicle Explanation

A vehicle-level behavior comparison module that identifies abnormal operational patterns by comparing a selected vehicle against fleet-level averages.

### Contribution 4 — Reproducible Engineering Pipeline

An end-to-end reproducible pipeline on a public real-world fleet dataset, including preprocessing, temporal summarization, model training, cost evaluation, agentic orchestration, and diagnostic reporting.

---

## 4. Dataset

Dataset:

```text
SCANIA Component X Dataset
```

Local path:

```text
data/raw/2024-34-3/
```

Main files:

| File | Purpose |
|---|---|
| `train_operational_readouts.csv` | Training time-series operational data |
| `validation_operational_readouts.csv` | Validation time-series operational data |
| `test_operational_readouts.csv` | Test time-series operational data |
| `train_specifications.csv` | Training vehicle specifications |
| `validation_specifications.csv` | Validation vehicle specifications |
| `test_specifications.csv` | Test vehicle specifications |
| `train_tte.csv` | Training target information |
| `validation_labels.csv` | Validation labels |
| `test_labels.csv` | Test labels |

Initial dataset statistics:

| Split | Vehicles | Operational Readouts |
|---|---:|---:|
| Train | 23,550 | 1,122,452 |
| Validation | 5,046 | 196,227 |
| Test | 5,045 | 198,140 |

---

## 5. Current Pipeline Status

### Completed

| Component | File |
|---|---|
| Dataset audit | `src/data_audit.py` |
| CSV to Parquet preprocessing | `src/preprocessing.py` |
| Temporal feature summarization | `src/temporal_summarization.py` |
| Baseline model training | `src/train_baselines.py` |
| Cost-sensitive evaluation | `src/cost_evaluation.py` |
| Maintenance decision tool | `agents/decision_tool.py` |
| Vehicle tool wrapper | `agents/tools.py` |
| Behavior comparison tool | `agents/behavior_tool.py` |
| Maintenance agent | `agents/maintenance_agent.py` |
| Report generator | `agents/report_generator.py` |
| Agentic orchestrator | `agents/orchestrator_agent.py` |

### Current Best Model

| Model | ROC-AUC | PR-AUC | Notes |
|---|---:|---:|---|
| Random Forest | ~0.635 | ~0.058 | baseline |
| LightGBM | ~0.664 | ~0.074 | current best ranking model |

### Current Cost Evaluation

| Scenario | Best Threshold | Cost Saving vs No Action |
|---|---:|---:|
| balanced_cost | 0.83 | 2.06% |
| safety_critical | 0.23 | 34.68% |
| inspection_expensive | 0.87 | 0.66% |
| recall_focused | 0.21 | 39.26% |

---

## 6. Experimental Protocol

### 6.1 Data Splits

The final experiments should follow this protocol:

| Split | Usage |
|---|---|
| Train | model fitting |
| Validation | hyperparameter tuning and threshold selection |
| Test | final reported performance |

Important rule:

```text
Do not report final performance only on validation.
Final paper tables must include test-set results.
```

---

## 7. Models to Compare

### Baseline Models

| Model | Purpose |
|---|---|
| Logistic Regression | simple linear baseline |
| Random Forest | tree-based baseline |
| LightGBM | current main model |
| CatBoost | categorical specification-aware model |

### Final Candidate Models

| Model | Required Work |
|---|---|
| Tuned LightGBM | hyperparameter search |
| CatBoost | include categorical specification features |
| Calibrated LightGBM | probability calibration |
| Two-stage model | optional advanced model |

---

## 8. Evaluation Metrics

### Predictive Metrics

| Metric | Reason |
|---|---|
| ROC-AUC | ranking performance |
| PR-AUC | important for imbalanced labels |
| Precision | false alarm control |
| Recall | missed failure reduction |
| F1-score | precision-recall balance |
| Confusion matrix | operational interpretation |

### Cost Metrics

| Metric | Reason |
|---|---|
| total cost | overall maintenance decision cost |
| average cost per vehicle | normalized cost |
| cost saving vs no action | operational benefit |
| best cost-sensitive threshold | deployable decision threshold |

---

## 9. Cost Scenarios

Current scenarios:

| Scenario | False Positive Cost | False Negative Cost | True Positive Cost | Use Case |
|---|---:|---:|---:|---|
| balanced_cost | 10 | 100 | 20 | general maintenance |
| safety_critical | 10 | 500 | 30 | missed failures are critical |
| inspection_expensive | 50 | 100 | 30 | inspections are costly |
| recall_focused | 5 | 300 | 20 | prioritize failure detection |

For the paper, the cost assumptions must be clearly stated as scenario-based engineering assumptions.

---

## 10. Ablation Study Plan

The ablation study should show the value added by each layer.

| Variant | Description |
|---|---|
| ML only | model probability only |
| ML + fixed threshold | model with threshold = 0.50 |
| ML + cost threshold | model with scenario-specific threshold |
| ML + cost + behavior | adds abnormal feature comparison |
| Full agentic system | orchestration, state, reports, charts |

Expected analysis:

```text
Show whether the full system improves decision quality, interpretability, and operational usefulness compared with model-only baselines.
```

---

## 11. Case Study Plan

At least four vehicle case studies are required:

| Case | Description |
|---|---|
| True Positive | failure vehicle correctly flagged |
| False Positive | normal vehicle flagged as risky |
| False Negative | failure vehicle missed |
| True Negative | normal vehicle correctly ignored |

For each case study, report:

- vehicle ID
- true label
- failure probability
- selected scenario
- threshold
- recommended action
- top abnormal features
- generated report path
- agent state summary

---

## 12. Explainability Plan

### Behavior-Based Explanation

Already implemented:

```text
vehicle vs fleet average comparison
top abnormal features by standardized difference
```

### Model-Based Explanation

To implement:

```text
SHAP local explanation
SHAP global feature importance
SHAP plots for selected case studies
```

Required file:

```text
src/explainability.py
```

---

## 13. Agentic Workflow Evaluation

The orchestrator should be evaluated as an agentic workflow.

Current workflow:

```text
Objective received
→ maintenance decision
→ behavior analysis
→ report generation
→ progress evaluation
→ shared task state saved
→ dashboard state saved
```

Required outputs:

| Output | Path |
|---|---|
| Markdown report | `reports/generated_reports/vehicle_<id>_report.md` |
| Full task state | `reports/generated_reports/vehicle_<id>_<scenario>_agent_state.json` |
| Dashboard state | `reports/generated_reports/vehicle_<id>_<scenario>_dashboard_state.json` |

---

## 14. Dashboard Plan

Required file:

```text
app/streamlit_app.py
```

Dashboard v1 should include:

- vehicle ID selector
- cost scenario selector
- run orchestrator button
- failure probability card
- threshold card
- recommended action card
- severity indicator
- top abnormal features table
- risk vs threshold chart
- abnormal features chart
- generated report preview

---

## 15. RAG Documentation Agent Plan

Required files:

```text
rag/document_loader.py
rag/retriever.py
rag/qa_tool.py
```

Purpose:

```text
Allow the agent to answer questions about dataset documentation, label definitions, challenge setup, and evaluation context.
```

Documents:

```text
data/raw/2024-34-3/documentation/2024_IDA_challenge_v2.pdf
data/raw/2024-34-3/documentation/Scania_Component_X.pdf
```

---

## 16. Paper Tables to Generate

| Table | Description |
|---|---|
| Table 1 | Dataset statistics |
| Table 2 | Model comparison |
| Table 3 | Cost scenario comparison |
| Table 4 | Ablation study |
| Table 5 | Case studies |
| Table 6 | Agentic workflow outputs |

---

## 17. Paper Figures to Generate

| Figure | Description |
|---|---|
| Figure 1 | Proposed agentic architecture |
| Figure 2 | Data processing pipeline |
| Figure 3 | Cost-sensitive threshold curves |
| Figure 4 | Risk vs threshold example |
| Figure 5 | Abnormal feature explanation |
| Figure 6 | Dashboard screenshot |
| Figure 7 | Agentic workflow state diagram |

---

## 18. Submission Readiness Checklist

Before submission to EAAI:

- [ ] dashboard v1 completed
- [ ] tuned LightGBM completed
- [ ] CatBoost baseline completed
- [ ] final test-set evaluation completed
- [ ] SHAP explanations completed
- [ ] ablation study completed
- [ ] case studies completed
- [ ] RAG documentation agent completed or clearly scoped
- [ ] architecture figure completed
- [ ] all paper tables generated
- [ ] all paper figures generated
- [ ] GitHub repository cleaned
- [ ] README updated
- [ ] paper draft completed
- [ ] references completed
- [ ] cover letter prepared

---

## 19. Target Timeline

Target journal:

```text
Engineering Applications of Artificial Intelligence
```

Target submission window:

```text
January 2027
```

Preferred internal deadline:

```text
First complete draft by mid-December 2026
```

---

## 20. Immediate Next Steps

1. Build Streamlit dashboard v1.
2. Add tuned LightGBM and CatBoost experiments.
3. Add final test-set evaluation.
4. Add SHAP explainability.
5. Generate ablation and case study tables.