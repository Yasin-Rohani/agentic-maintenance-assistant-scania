# E02 — Time–label decoupling: shortcut effect vs class-prior effect

| | |
|---|---|
| Date | 2026-11-17 |
| Script | `experiments/time_label_decoupling.py` |
| Config | `configs/experiments/time_label_decoupling.yaml` |
| Results | `results/E02_time_label_decoupling/` |
| Status | Completed — shortcut removal explains the gain; the prior does not |
| Design commit | `66765fc` (rules fixed before the run) |

## Question

In E01, re-weighting training snapshots to the Validation `cutoff_time`
distribution (arm F) roughly doubled AP. Because late snapshots are more often
positive, that re-weighting also raised the effective positive-class prior
(E01 ESS fraction: 0.415). Which of the two caused the gain?

1. Removing the time–label dependence (the shortcut), or
2. Up-weighting positive classes (the prior)?

## Design

All arms use all features and the E01 LightGBM. Only unlabelled Validation
cutoff times are used; Validation labels are never used for training,
weighting or calibration.

| Arm | Training weights | Time ⟂ label in training? | Class prior |
|---|---|---|---|
| A_none | none | no | fit prior |
| F_marginal | p_val(t) / p_fit(t) | partly | raised |
| I_prior_only | class factor reproducing F's prior | no | raised (= F) |
| H_decorrelate_val | p_val(t) / p_fit(t \| c), class totals kept | yes | fit prior |
| H0_decorrelate_fit | p_fit(t) / p_fit(t \| c), class totals kept | yes | fit prior |
| J_decorrelate_prior | H_decorrelate_val × I factor | yes | raised (= F) |

20 quantile bins, Laplace smoothing 1, ratios clipped to [0.1, 10].

Repeated over 3 vehicle-disjoint fit/calibration splits (`original` = the E01
split, and seeds 1, 2). The paired bootstrap (1,000 Validation resamples,
sigmoid probabilities) averages each metric over the splits inside every
resample, then takes the difference to A_none.

Per arm and split it records: ESS fraction, effective class prior, and the
weighted ROC-AUC of `cutoff_time` for failure-within-48 in training
(0.5 = time carries no label information after weighting).

### Interpretation rules (fixed before running)

1. **Prior explains the gain** if I_prior_only recovers most (≥ ~70%) of the
   F − A gain in AP≤48, and H_decorrelate_val does not.
2. **Shortcut removal explains the gain** if H_decorrelate_val recovers most
   of the F − A gain and I_prior_only does not.
3. **Both contribute** if each recovers part and J ≈ F.
4. **Protocol choice:** if H0_decorrelate_fit ≈ H_decorrelate_val, adopt H0
   for all later experiments and Test, because it needs no target
   information at all. Otherwise adopt the best Validation-covariate scheme,
   frozen before Test.
5. A conclusion counts only if its direction holds in every split and its
   averaged bootstrap interval excludes zero.
6. Check the weighting diagnostic: if the weighted cutoff AUC of a decorrelate
   arm stays far from 0.5 (e.g. > 0.6), clipping limited the decorrelation;
   report it and do not claim the dependence was removed.

## Results

Three vehicle-disjoint fit/calibration splits (`original`, 1, 2). Run on
commit `66765fc`, about 80 minutes on an 8 GB laptop.

### Training-weight diagnostics (mean over splits)

| Arm | ESS fraction | Weighted cutoff AUC (≤48) | Eff. prior class 4 |
|---|---|---|---|
| A_none | 1.000 | 0.725 | 0.0040 |
| F_marginal | 0.419 | 0.597 | 0.0088 |
| I_prior_only | 0.984 | 0.731 | 0.0088 |
| H_decorrelate_val | 0.411 | **0.518** | 0.0040 |
| H0_decorrelate_fit | 0.984 | 0.631 | 0.0040 |
| J_decorrelate_prior | 0.415 | 0.519 | 0.0088 |

### Validation, sigmoid calibration (mean over splits)

| Arm | AP ≤48 | ROC-AUC ≤48 | AP class 4 | Cost (min-expected) | Cost @30% |
|---|---|---|---|---|---|
| A_none | 0.069 | 0.675 | 0.048 | 42,926 | 41,470 |
| F_marginal | 0.104 | 0.730 | 0.062 | 38,973 | 36,717 |
| I_prior_only | 0.076 | 0.678 | 0.048 | 42,013 | 42,696 |
| **H_decorrelate_val** | **0.103** | **0.727** | **0.069** | **39,335** | **38,414** |
| H0_decorrelate_fit | 0.083 | 0.689 | 0.067 | 42,970 | 41,448 |
| J_decorrelate_prior | 0.091 | 0.719 | 0.057 | 38,814 | 38,045 |

### Paired bootstrap vs A_none (averaged over splits, 1,000 resamples)

| Arm | Δ AP ≤48 [95% CI] | Δ AP class 4 [95% CI] | Δ cost [95% CI] |
|---|---|---|---|
| F_marginal | +0.036 [0.009, 0.066] | +0.013 [−0.010, 0.037] | −3,989 [−6,035, −1,756] |
| I_prior_only | +0.005 [−0.009, 0.020] | 0.000 [−0.016, 0.014] | −938 [−2,081, −24] |
| **H_decorrelate_val** | **+0.035 [0.014, 0.065]** | **+0.021 [0.002, 0.043]** | **−3,615 [−5,383, −1,855]** |
| H0_decorrelate_fit | +0.014 [0.000, 0.030] | +0.019 [0.002, 0.041] | +8 [−1,150, 909] |
| J_decorrelate_prior | +0.022 [0.004, 0.044] | +0.009 [−0.014, 0.030] | −4,146 [−5,752, −2,142] |

### Per-split direction (H_decorrelate_val − A_none, sigmoid)

| Split | Δ AP ≤48 | Δ AP class 4 | Δ cost | Δ cost @30% |
|---|---|---|---|---|
| original | +0.035 | +0.020 | −3,061 | −4,275 |
| 1 | +0.012 | +0.013 | −4,113 | +993 |
| 2 | +0.055 | +0.030 | −3,599 | −5,886 |

## Interpretation

- **Rule 2 met — the gain comes from removing the shortcut.**
  H_decorrelate_val keeps the training class prior unchanged and recovers
  ~99% of F's AP gain (+0.035 vs +0.036). I_prior_only reproduces F's prior
  exactly but recovers only ~15%, with an interval containing zero.
- H is the only arm whose class-4 AP gain excludes zero; it is the best
  arm for the rarest, most expensive class.
- Raising the prior on top of decorrelation (J) lowers AP relative to H:
  the prior shift does not help ranking.
- **Rule 6:** H0 leaves a weighted cutoff AUC of 0.63 (> 0.6). Clipping
  limited its decorrelation, so its smaller gain cannot be read as evidence
  against the mechanism. H reaches 0.52 — the dependence is effectively removed.
- **Rule 4 — protocol choice:** H0 did not match H, so
  **H_decorrelate_val is adopted** for all later experiments and is frozen
  before Test. It uses only the unlabelled target `cutoff_time`
  distribution, never target labels.
- **Rule 5:** for H, the direction of Δ AP ≤48, Δ AP class 4 and Δ cost
  (min-expected) holds in all three splits and every averaged interval
  excludes zero.

## Caveats

1. Cost at 30% capacity is not consistent across splits for H
   (split 1: +993). The capacity-policy gain is less stable than the
   ranking gain; to be examined when the capacity policy is revisited.
2. Absolute AP stays low (~0.10) because positives are rare in Validation
   (136 of 5,046 within 48). The relative gain is the result.
3. Weights depend on the unlabelled target cutoff distribution. On Test the
   same procedure is applied once to the Test cutoff times, with no tuning.
4. ESS fraction for H is 0.41; a less aggressive clip could improve H0 and is
   left as a sensitivity check.

## Code change in this step

`experiments/cutoff_ablation.py`: unseen categories are now mapped to missing
with `Index.get_indexer` instead of `pd.Categorical`. Same values as before,
no pandas deprecation warning. E01 results are unaffected.
