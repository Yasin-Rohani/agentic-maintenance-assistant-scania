# E03 — Discrete-time hazard baseline vs the decorrelated classifier

| | |
|---|---|
| Date | 2026-11-24 |
| Scripts | `experiments/build_landmark_features.py`, `experiments/survival_baseline.py` |
| Config | `configs/experiments/survival_baseline.yaml` |
| Results | `results/E03_survival_baseline/` |
| Status | Designed — awaiting run |

## Question

E01/E02 traced the cutoff-time shortcut to the snapshot-labelling protocol:
an unrepaired vehicle's snapshot gets class 0 only if at least 48 units of
observation follow it, so the late history of censored vehicles never enters
training. The E02 fix (H) re-weights training snapshots with the *unlabelled
target* cutoff distribution.

A survival model uses censored history correctly by construction. Can a
discrete-time hazard model match H **without any target information**?
A reviewer will ask this; the answer decides the main model of the paper.

## Design

### Landmarks
Up to 4 landmarks per training vehicle, sampled uniformly from **all** readout
times (seed 42), with no label filter. Features come from the project's own
`build_snapshot_features`, so they are identical in definition and column
order to the classifier features (1,165 inputs). Each landmark stores its
remaining observed time and repair indicator.

Training vehicles without any labelled snapshot are missing from the
classifier data; their landmarks join hazard training in every split. They are
never calibration vehicles, so this cannot leak.

### Hazard model
Remaining-time intervals (0,6], (6,12], (12,24], (24,48] → classes 4, 3, 2, 1.
One binary LightGBM per interval, trained on landmarks at risk at the start of
the interval and fully observed through it (event = repair inside the interval;
survived = observed beyond it; censored inside the interval = excluded).
P(4)=h1, P(3)=S1·h2, P(2)=S2·h3, P(1)=S3·h4, P(0)=S4.

### Arms
| Arm | Training data | Model | Weights |
|---|---|---|---|
| A_classifier | labelled snapshots | multiclass | none (E02 A_none) |
| H_classifier | labelled snapshots | multiclass | decorrelate_val (E02 H, frozen) |
| S_hazard | unfiltered landmarks | 4 interval hazards | none |
| SH_hazard | unfiltered landmarks | 4 interval hazards | decorrelate_val inside each interval |

Same LightGBM settings, the same three vehicle-disjoint splits as E02, the
same labelled calibration snapshots and the same Validation rows for every arm.
Paired bootstrap (1,000 resamples, sigmoid probabilities) averaged over splits,
**reference = H_classifier**.

### Interpretation rules (fixed before running)

0. **Reproduction check.** A_classifier and H_classifier must reproduce E02
   (mean AP≤48 within ±0.005). If not, investigate before interpreting.
1. **S matches H** if the S − H interval for AP≤48 contains zero with mean
   ≥ −0.01, and the S − H cost interval contains zero or is below it.
   → Adopt **S_hazard as the main model**: equal performance with no target
   information is strictly better for deployment and for Test.
2. **S better than H** if the S − H AP≤48 interval is above zero → adopt S.
3. **S worse than H** if the S − H AP≤48 interval is below zero → H stays the
   main model; the hazard model is reported as a principled baseline.
4. **Residual shortcut in the hazard model:** if SH − S is positive with an
   interval above zero, the hazard training still contains a time shortcut;
   otherwise correct censoring alone removed it.
5. A conclusion counts only if its direction holds in every split.
6. Report per-interval rows, events, ESS and cutoff AUC (unweighted and
   weighted). Cutoff–event association inside a hazard model is not by itself a
   shortcut (time can be genuinely informative); rule 4 is the test.

## Results

_To be filled after the run._

## Interpretation

_To be filled after the run._
