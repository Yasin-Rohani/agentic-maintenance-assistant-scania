"""
E03 — Discrete-time hazard (survival) baseline vs the decorrelated classifier.

Question
--------
E01/E02 showed that the classifier's snapshot-labelling protocol (late history
of censored vehicles is dropped) creates a cutoff-time shortcut, and that
re-weighting with the unlabelled target cutoff distribution (H) removes it.
H needs the target's cutoff distribution. A survival model that uses censored
history correctly may avoid the shortcut without any target information.

Arms (same features, same LightGBM settings, same calibration and evaluation)
----
A_classifier   multiclass on labelled snapshots, no weights        (E02 A_none)
H_classifier   multiclass on labelled snapshots, decorrelate_val   (E02 H)
S_hazard       discrete-time hazard on unfiltered landmarks, no weights
SH_hazard      S_hazard with decorrelate_val weights inside each interval model

Hazard model
------------
Remaining-time intervals (0,6], (6,12], (12,24], (24,48] map to classes
4, 3, 2, 1. One binary LightGBM per interval k, trained on landmarks at risk at
the start of the interval and fully observed through it:
    event    : repaired and remaining time in (a_k, b_k]
    survived : remaining time > b_k
    excluded : censored inside the interval, or not at risk
Class probabilities: P(4)=h1, P(3)=S1*h2, P(2)=S2*h3, P(1)=S3*h4, P(0)=S4.

Training vehicles that have no labelled snapshot at all are absent from the
classifier data; their landmarks are always added to hazard training (they are
never calibration vehicles, so this cannot leak).

All arms are calibrated on the same labelled calibration snapshots and
evaluated on the same Validation rows; repeated over the same vehicle-disjoint
splits as E02.

Usage (from the repository root, after build_landmark_features.py)
-----
    python experiments/survival_baseline.py
    python experiments/survival_baseline.py --seeds original
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cutoff_ablation import (  # noqa: E402
    ROOT,
    Partition,
    extended_metrics,
    git_commit,
    load_config,
    log,
    select_features,
)
from time_label_decoupling import (  # noqa: E402
    Pool,
    Subset,
    make_weights,
    multi_split_bootstrap,
    shared_vocabulary,
)
from calibrate_multiclass_lightgbm import (  # noqa: E402
    apply_calibrators,
    normalize_probabilities,
    train_isotonic_calibrators,
    train_sigmoid_calibrators,
    vehicle_weights,
)
from train_multiclass_lightgbm import CLASSES  # noqa: E402

EPS = 1e-5


# ============================================================
# Data
# ============================================================

class Landmarks(Partition):
    """Landmark partition plus its censoring information."""

    def __init__(self, path, cfg, categories):
        super().__init__(path, cfg, categories=categories)
        meta = pq.read_table(path, columns=["remaining_time", "in_study_repair"])
        self.remaining = meta.column("remaining_time").to_numpy().astype(float)
        self.repaired = meta.column("in_study_repair").to_numpy().astype(int)

    def rows_for(self, vehicles):
        return np.flatnonzero(np.isin(self.vehicle_id, vehicles))


def load_data(cfg):
    data = cfg["data"]
    paths = [ROOT / data["fit"], ROOT / data["calibration"]]
    vocab = shared_vocabulary(paths, cfg["categorical_columns"])

    fit = Partition(paths[0], cfg, categories=vocab)
    cal = Partition(paths[1], cfg, categories=vocab)
    val = Partition(ROOT / data["validation"], cfg, categories=vocab)
    landmarks = Landmarks(ROOT / data["landmarks"], cfg, vocab)

    assert fit.columns == cal.columns == val.columns == landmarks.columns, \
        "feature columns differ between files"

    pool = Pool(fit, cal)
    del fit, cal
    gc.collect()

    # Vehicles with no labelled snapshot at all (e.g. short censored histories)
    # are absent from the classifier pool. They are never used for
    # calibration, so they can always join hazard training without leakage.
    landmarks.extra_vehicles = np.setdiff1d(np.unique(landmarks.vehicle_id),
                                            np.unique(pool.vehicle_id))
    assert not set(pool.vehicle_id) & set(val.vehicle_id)
    assert not set(landmarks.vehicle_id) & set(val.vehicle_id)
    return pool, landmarks, val


# ============================================================
# Models
# ============================================================

def lgbm(cfg, objective, **extra):
    return LGBMClassifier(objective=objective, verbosity=-1,
                          **cfg["model"], **cfg.get("lightgbm_memory", {}), **extra)


def fit_classifier(fit, val, cal, weights, cfg):
    features = select_features(fit.columns, cfg, "all")
    cats = [c for c in fit.categorical if c in features]
    model = lgbm(cfg, "multiclass", num_class=5)
    model.fit(fit.X, fit.y, sample_weight=weights,
              feature_name=features, categorical_feature=cats or "auto")
    order = [list(model.classes_).index(c) for c in CLASSES]
    out = (normalize_probabilities(model.predict_proba(cal.X)[:, order]),
           normalize_probabilities(model.predict_proba(val.X)[:, order]))
    del model
    gc.collect()
    return out, {}


def interval_targets(remaining, repaired, a, b, first):
    at_risk = np.ones_like(remaining, bool) if first else remaining > a + EPS
    event = at_risk & (repaired == 1) & (remaining <= b + EPS)
    survived = at_risk & (remaining > b + EPS)
    include = event | survived
    return include, event


def fit_hazard(lm, rows, val, cal, weighted, cfg):
    features = select_features(lm.columns, cfg, "all")
    cats = [c for c in lm.categorical if c in features]
    remaining, repaired = lm.remaining[rows], lm.repaired[rows]
    cutoff = lm.cutoff[rows]

    hazards_cal, hazards_val, info = [], [], {}
    for k, (a, b) in enumerate(cfg["hazard"]["intervals"]):
        include, event = interval_targets(remaining, repaired, a, b, k == 0)
        idx = rows[include]
        y = event[include].astype(int)
        t = cutoff[include]

        w = (make_weights("decorrelate_val", t, y, val.cutoff, cfg)
             if weighted else np.ones(len(y)))

        tag = f"k{k + 1}"
        info[f"{tag}_rows"] = int(len(y))
        info[f"{tag}_events"] = int(y.sum())
        info[f"{tag}_cutoff_auc"] = float(roc_auc_score(y, t))
        info[f"{tag}_cutoff_auc_weighted"] = float(roc_auc_score(y, t, sample_weight=w))
        info[f"{tag}_ess_fraction"] = float(w.sum() ** 2 / np.sum(w ** 2) / len(w))

        X = np.ascontiguousarray(lm.X[idx])
        model = lgbm(cfg, "binary")
        model.fit(X, y, sample_weight=w,
                  feature_name=features, categorical_feature=cats or "auto")
        del X
        hazards_cal.append(model.predict_proba(cal.X)[:, 1])
        hazards_val.append(model.predict_proba(val.X)[:, 1])
        del model
        gc.collect()

    return (hazard_to_classes(np.column_stack(hazards_cal)),
            hazard_to_classes(np.column_stack(hazards_val))), info


def hazard_to_classes(h):
    """h[:, k] = hazard of interval k (0..3) -> probabilities of classes 0..4."""
    n = len(h)
    probs = np.zeros((n, 5))
    surv = np.ones(n)
    for k, cls in enumerate([4, 3, 2, 1]):
        probs[:, cls] = surv * h[:, k]
        surv = surv * (1 - h[:, k])
    probs[:, 0] = surv
    return normalize_probabilities(probs)


# ============================================================
# Evaluation of one arm
# ============================================================

def evaluate_arm(name, cal_probs, val_probs, cal, val, cfg, info):
    weights = vehicle_weights(cal.vehicle_id)
    variants = {"uncalibrated": val_probs}
    if "sigmoid" in cfg["calibration_methods"]:
        variants["sigmoid"] = apply_calibrators(
            val_probs, train_sigmoid_calibrators(cal_probs, cal.y, weights), "sigmoid")
    if "isotonic" in cfg["calibration_methods"]:
        variants["isotonic"] = apply_calibrators(
            val_probs, train_isotonic_calibrators(cal_probs, cal.y, weights), "isotonic")

    rows = []
    for method, probs in variants.items():
        m = extended_metrics(method, val.y, probs, cfg["capacity_fractions"])
        rows.append({"arm": name, "calibration": m.pop("method"), **m, **info})
    return rows, variants


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E03 survival baseline")
    parser.add_argument("--config", default="configs/experiments/survival_baseline.yaml")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--seeds", nargs="*")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [a for a in cfg["arms"] if not args.arms or a["name"] in args.arms]
    seeds = [str(s) for s in (args.seeds or cfg["split_seeds"])]
    ref = cfg["bootstrap"]["reference_arm"]
    assert ref in [a["name"] for a in arms], f"reference arm {ref} must be run"

    log("Loading data (streaming into float32 matrices)...")
    pool, lm, val = load_data(cfg)
    log(f"pool={pool.X.shape} landmarks={lm.X.shape} validation={val.X.shape} "
        f"| matrices: {(pool.X.nbytes + lm.X.nbytes + val.X.nbytes) / 1e6:,.0f} MB")
    log(f"Landmarks with undefined class (censored within 48): "
        f"{(lm.y < 0).sum():,} of {len(lm.y):,}; vehicles with no labelled "
        f"snapshot (hazard-only): {len(lm.extra_vehicles):,}")

    all_rows, boot_probs = [], {}
    primary = cfg["bootstrap"]["calibration"]

    for seed in seeds:
        fit_rows, cal_rows = pool.split(seed, cfg["calibration_fraction"])
        fit, cal = Subset(pool, fit_rows), Subset(pool, cal_rows)
        assert not set(fit.vehicle_id) & set(cal.vehicle_id)
        lm_rows = lm.rows_for(np.concatenate([np.unique(fit.vehicle_id),
                                              lm.extra_vehicles]))
        assert not set(lm.vehicle_id[lm_rows]) & set(cal.vehicle_id)
        log(f"Split {seed}: fit={len(fit.y)} calibration={len(cal.y)} "
            f"landmarks(fit vehicles)={len(lm_rows)}")

        boot_probs[seed] = {}
        for arm in arms:
            start = time.time()
            log(f"  Arm {arm['name']}")
            if arm["model"] == "classifier":
                w = make_weights(arm["weights"], fit.cutoff, fit.y, val.cutoff, cfg)
                (cal_p, val_p), info = fit_classifier(fit, val, cal, w, cfg)
            else:
                (cal_p, val_p), info = fit_hazard(
                    lm, lm_rows, val, cal, arm["weights"] == "decorrelate_val", cfg)
            info["fit_seconds"] = round(time.time() - start, 1)

            rows, variants = evaluate_arm(arm["name"], cal_p, val_p, cal, val, cfg, info)
            for r in rows:
                r["split"] = seed
            all_rows.extend(rows)
            boot_probs[seed][arm["name"]] = variants[primary]
            pd.DataFrame(all_rows).to_csv(out_dir / "summary.csv", index=False)
            gc.collect()

        del fit, cal
        gc.collect()

    summary = pd.DataFrame(all_rows)

    log("Paired bootstrap averaged over splits...")
    bs = cfg["bootstrap"]
    boot = multi_split_bootstrap(val.y, boot_probs, ref,
                                 bs["n_resamples"], bs["random_state"])
    boot.to_csv(out_dir / "bootstrap.csv", index=False)

    metrics = ["pr_auc_within48", "roc_auc_within48", "pr_auc_class4",
               "roc_auc_class4", "brier", "official_cost_min_expected",
               "cost_cap10", "cost_cap30"]
    agg = (summary.groupby(["arm", "calibration"], sort=False)[metrics]
           .agg(["mean", "std"]))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(out_dir / "summary_by_arm.csv", index=False)

    hazard_cols = [c for c in summary.columns if c.startswith("k") and "_" in c]
    hazard_diag = (summary[(summary.calibration == "uncalibrated")
                           & summary.arm.str.contains("hazard")]
                   .groupby("arm", sort=False)[hazard_cols].mean())
    hazard_diag.to_csv(out_dir / "hazard_diagnostics.csv")

    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "seeds_run": seeds,
        "arms_run": [a["name"] for a in arms],
        "config": cfg,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    if len(hazard_diag):
        print("\n" + "=" * 100)
        print("HAZARD TRAINING DIAGNOSTICS (per interval model, averaged over splits)")
        print("=" * 100)
        print(hazard_diag.T.round(4).to_string())

    print("\n" + "=" * 100)
    print("VALIDATION — mean over splits")
    print("=" * 100)
    show = agg.reset_index()
    cols = ["arm", "calibration"] + [c for c in show.columns if c.endswith("_mean")]
    print(show[cols].round(4).to_string(index=False))

    print(f"\nPAIRED BOOTSTRAP vs {ref} ({primary}), averaged over splits {seeds}")
    print(boot.round(4).to_string(index=False))

    print(f"\nOutputs written to: {out_dir}")
    log("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
