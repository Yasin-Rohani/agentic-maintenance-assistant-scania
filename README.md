# Agentic Maintenance Assistant for SCANIA Fleet Data

This project builds an agentic decision-support system for predictive maintenance using the SCANIA Component X dataset.

The goal is to create an AI maintenance assistant that can analyze truck fleet data, predict component failure risk, compare current behavior against fleet and historical baselines, recommend the best maintenance action, generate visual diagnostic reports, and provide documentation-aware answers through RAG.

## Project Goals

- Build a predictive maintenance ML core for SCANIA Component X data
- Convert time-series truck readouts into model-ready temporal features
- Rank trucks by failure risk
- Compare a selected truck against historical and fleet behavior
- Evaluate maintenance actions with cost-aware decision logic
- Generate chart-rich diagnostic maintenance reports
- Build a tool-using AI agent for maintenance decision support
- Add RAG over SCANIA documentation
- Build an industrial dashboard for large-screen fleet monitoring

## Planned Architecture

```text
SCANIA raw data
    ↓
Data audit and preprocessing
    ↓
Temporal feature engineering
    ↓
Failure risk prediction
    ↓
Cost-sensitive decision policy
    ↓
Explainability and uncertainty
    ↓
Agent tools
    ↓
Maintenance agent
    ↓
Industrial dashboard and visual report