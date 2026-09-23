# SCANIA Component X Dataset Notes

## Dataset Location

The dataset is stored locally under:

```text
data/raw/2024-34-3/
```

The extracted dataset contains two main folders:

```text
data/raw/2024-34-3/
├── data/
└── documentation/
```

---

## Data Files

The `data/` folder contains the following CSV files:

| File | Description | Rows | Columns | Size |
|---|---|---:|---:|---:|
| `train_operational_readouts.csv` | Time-series operational readouts for training vehicles | 1,122,452 | 107 | 1162.73 MB |
| `validation_operational_readouts.csv` | Time-series operational readouts for validation vehicles | 196,227 | 107 | 205.61 MB |
| `test_operational_readouts.csv` | Time-series operational readouts for test vehicles | 198,140 | 107 | 204.94 MB |
| `train_specifications.csv` | Static vehicle/component specifications for training vehicles | 23,550 | 9 | 1.03 MB |
| `validation_specifications.csv` | Static vehicle/component specifications for validation vehicles | 5,046 | 9 | 0.22 MB |
| `test_specifications.csv` | Static vehicle/component specifications for test vehicles | 5,045 | 9 | 0.22 MB |
| `train_tte.csv` | Training target information including in-study repair indicator | 23,550 | 3 | 0.33 MB |
| `validation_labels.csv` | Validation class labels | 5,046 | 2 | 0.04 MB |
| `test_labels.csv` | Test class labels | 5,045 | 2 | 0.04 MB |

---

## Documentation Files

The `documentation/` folder contains:

```text
2024_IDA_challenge_v2.pdf
Scania_Component_X.pdf
```

These files will be used later for documentation-aware retrieval and RAG.

---

## Dataset Splits

The dataset is divided into three splits:

| Split | Number of Vehicles | Operational Readouts |
|---|---:|---:|
| Train | 23,550 | 1,122,452 |
| Validation | 5,046 | 196,227 |
| Test | 5,045 | 198,140 |

---

## Training Target

The training target file is:

```text
train_tte.csv
```

The relevant target-related column identified during the initial audit is:

```text
in_study_repair
```

Initial distribution:

| Value | Count | Percentage |
|---:|---:|---:|
| 0 | 21,278 | 90.35% |
| 1 | 2,272 | 9.65% |

Interpretation:

- `0`: no in-study repair
- `1`: in-study repair observed

The training target is imbalanced, but the imbalance is moderate compared with the validation and test labels.

---

## Validation Labels

The validation label file is:

```text
validation_labels.csv
```

Relevant column:

```text
class_label
```

Initial distribution:

| Class Label | Count | Percentage |
|---:|---:|---:|
| 0 | 4,910 | 97.30% |
| 4 | 76 | 1.51% |
| 3 | 30 | 0.59% |
| 1 | 16 | 0.32% |
| 2 | 14 | 0.28% |

---

## Test Labels

The test label file is:

```text
test_labels.csv
```

Relevant column:

```text
class_label
```

Initial distribution:

| Class Label | Count | Percentage |
|---:|---:|---:|
| 0 | 4,903 | 97.19% |
| 4 | 60 | 1.19% |
| 3 | 41 | 0.81% |
| 1 | 26 | 0.52% |
| 2 | 15 | 0.30% |

---

## Initial Observations

1. The operational readout files are large time-series tables.
2. Each vehicle has multiple operational readouts.
3. Static specification files are much smaller and contain one row per vehicle.
4. Training labels and validation/test labels have different structures.
5. Validation and test labels are highly imbalanced.
6. Accuracy alone will not be a reliable evaluation metric.
7. Cost-sensitive evaluation is necessary because missed failures and unnecessary inspections have different operational costs.

---

## Modeling Implications

The project should focus on:

- temporal feature engineering
- class imbalance handling
- recall and precision trade-offs
- PR-AUC and ROC-AUC
- false negative reduction
- cost-sensitive thresholding
- uncertainty-aware recommendations
- explainable risk factors

---

## Agentic System Implications

The maintenance agent should not only predict failure risk. It should also:

- compare a selected vehicle against fleet-level baselines
- compare recent behavior against historical behavior
- evaluate maintenance options using expected cost
- explain the main risk drivers
- generate chart-rich maintenance reports
- support documentation-aware Q&A using the dataset PDFs

---

## Next Steps

1. Convert large CSV files to Parquet for faster processing.
2. Build temporal features per vehicle.
3. Train baseline predictive models.
4. Add cost-sensitive decision logic.
5. Wrap data, prediction, cost, and explanation components as tools.
6. Build the first maintenance agent and dashboard.