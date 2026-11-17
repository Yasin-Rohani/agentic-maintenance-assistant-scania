# E02 — Time–label decoupling: shortcut effect vs class-prior effect

| | |
|---|---|
| Date | 2026-11-17 |
| Script | `experiments/time_label_decoupling.py` |
| Config | `configs/experiments/time_label_decoupling.yaml` |
| Results | `results/E02_time_label_decoupling/` |
| Status | Designed — awaiting run |

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

_To be filled after the run._

## Interpretation

_To be filled after the run._

## Code change in this step

`experiments/cutoff_ablation.py`: unseen categories are now mapped to missing
with `Index.get_indexer` instead of `pd.Categorical`. Same values as before,
no pandas deprecation warning. E01 results are unaffected.
