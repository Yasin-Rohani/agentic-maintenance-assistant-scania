# E01 — Cutoff-time shortcut ablation

| | |
|---|---|
| Date | 2026-11-06 |
| Script | `experiments/cutoff_ablation.py` |
| Config | `configs/experiments/cutoff_ablation.yaml` |
| Results | `results/E01_cutoff_ablation/` |
| Status | Completed — shortcut confirmed |

## Question

Does the multiclass LightGBM rely on *when* a training snapshot was taken
rather than on vehicle condition?

## Design

Same vehicle-disjoint fit / calibration / Validation partitions and the same
LightGBM hyper-parameters as `src/train_multiclass_lightgbm.py`.

| Arm | Inputs |
|---|---|
| A_full | all 1,165 features (reference) |
| B_no_cutoff | without `cutoff_time` |
| C_no_time | without `cutoff_time`, `num_readouts` |
| D_time_only | only `cutoff_time`, `num_readouts` |
| E_spec_only | only the 8 static vehicle specifications |
| F_full_reweighted | A + density-ratio weights matching the Validation `cutoff_time` distribution |
| G_no_time_reweighted | C + the same weights |

Weights use only unlabelled Validation cutoff times (20 quantile bins, ratios
clipped to [0.1, 10]). Paired bootstrap over Validation vehicles, 1,000
resamples, sigmoid-calibrated probabilities.

### Interpretation rules (fixed before running)

1. Shortcut confirmed if the univariate AUC of `cutoff_time` is above 0.5 in
   training and near or below 0.5 in Validation.
2. Removing time columns is insufficient if non-time features predict
   `cutoff_time` with R² > 0.5.
3. Time columns are dropped if C ≥ A on Validation.
4. Observation-protocol alignment becomes part of the pipeline if F > A.

## Results

### Shortcut diagnostics

Univariate ROC-AUC of `cutoff_time` for failure within 48 units:

| fit | calibration | Validation |
|---|---|---|
| 0.727 | 0.726 | **0.430** |

Mean cutoff of class 0: 100.6 (fit) vs 191.8 (Validation).

Duration probe (LightGBM regressor, non-time features → `cutoff_time`):
R² = 0.935 (calibration), 0.827 (Validation).

### Validation performance

| Arm | AP ≤48 | ROC-AUC ≤48 | ROC-AUC class 4 | Cost (sigmoid, min-expected) | Cost @30% (isotonic) |
|---|---|---|---|---|---|
| A_full | 0.056 | 0.662 | 0.670 | 42,226 | 40,524 |
| C_no_time | 0.056 | 0.654 | 0.658 | 41,061 | 43,678 |
| D_time_only | 0.028 | 0.530 | 0.555 | 49,298 | 53,379 |
| E_spec_only | 0.052 | 0.634 | 0.618 | 46,758 | 50,259 |
| **F_full_reweighted** | **0.110** | **0.742** | **0.729** | **38,547** | **35,245** |
| G_no_time_reweighted | 0.078 | 0.690 | 0.677 | 41,974 | 41,138 |

Paired bootstrap, F − A: AP ≤48 +0.055 (95% CI 0.025 to 0.097);
cost −3,705 (95% CI −6,723 to −285).

## Interpretation

- **Rule 1 met.** The direction of the time–label association reverses between
  training and Validation. Cause: an unrepaired training snapshot can be
  labelled class 0 only if at least 48 units of observation follow it, so
  negatives are structurally earlier than positives. Validation has no such
  constraint. The shortcut is created by the snapshot-labelling protocol.
- **Rule 2 met.** Cumulative sensor counters encode elapsed time; removing the
  time columns (B, C) leaves performance essentially unchanged.
- The full model is barely better than static specifications alone (A vs E),
  meaning most sensor signal was absorbed by the shortcut.
- **Rule 4 met.** Aligning the training cutoff distribution roughly doubles AP
  and reduces cost, with bootstrap intervals excluding zero.

## Caveats

1. Marginal time reweighting also changes the effective class prior, because
   late snapshots are more often positive. Part of F's gain may come from the
   prior change rather than from removing the shortcut. Not yet separated.
2. Weights are estimated from the Validation covariate distribution. No
   labels are used, but the procedure must be frozen before Test.
3. Single seed and single fit/calibration split.
4. Pandas emits a deprecation warning for categories unseen in fit; results
   are unaffected (unseen values become missing). To be fixed.

## Next

E02: separate the shortcut effect from the prior effect —
within-class time matching, a prior-only control, ESS and effective class
distribution per weighted arm, multiple split seeds.
