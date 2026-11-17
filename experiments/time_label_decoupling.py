"""
E02 — Separate the shortcut effect from the class-prior effect.

Background (E01)
----------------
Re-weighting training snapshots so that their cutoff_time distribution
matches Validation doubled average precision. But late snapshots are more
often positive, so that marginal re-weighting also raises the effective
positive-class prior. E02 asks which of the two produced the gain.

Arms (all features, same LightGBM as E01)
----
A_none              no weights
F_marginal          w = p_val(t) / p_fit(t)                       (E01 arm F)
I_prior_only        class weights reproducing F's effective prior, no time
H_decorrelate_val   w = p_val(t) / p_fit(t | c), class totals preserved
                    -> time independent of label in training, prior unchanged
H0_decorrelate_fit  w = p_fit(t) / p_fit(t | c), class totals preserved
                    -> same, but uses no Validation information at all
J_decorrelate_prior H_decorrelate_val x I_prior_only

Every arm is repeated over several vehicle-disjoint fit/calibration splits.
The paired bootstrap averages each arm-minus-reference difference over the
splits inside every Validation resample.

Usage (from the repository root)
-----
    python experiments/time_label_decoupling.py
    python experiments/time_label_decoupling.py --seeds original
    python experiments/time_label_decoupling.py --arms A_none H0_decorrelate_fit
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Shared, already-tested helpers from E01 (they also put src/ on the path).
from cutoff_ablation import (  # noqa: E402
    ROOT,
    Partition,
    bootstrap_metrics,
    extended_metrics,
    git_commit,
    load_config,
    log,
    safe_auc,
    select_features,
)
from calibrate_multiclass_lightgbm import (  # noqa: E402
    apply_calibrators,
    normalize_probabilities,
    train_isotonic_calibrators,
    train_sigmoid_calibrators,
    vehicle_weights,
)
from train_multiclass_lightgbm import CLASSES  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


# ============================================================
# Data: one training pool (fit + calibration) and Validation
# ============================================================

def shared_vocabulary(paths, categorical):
    """Category vocabularies from the training pool only (fit + calibration)."""
    vocab = {}
    for column in categorical:
        values = set()
        for path in paths:
            col = pq.read_table(path, columns=[column]).column(column).to_pandas()
            values |= set(col.fillna("MISSING").astype(str).unique())
        vocab[column] = sorted(values)
    return vocab


class Subset:
    """Partition-like view built from row indices of a pooled matrix."""

    def __init__(self, pool, rows):
        self.X = np.ascontiguousarray(pool.X[rows])
        self.y = pool.y[rows]
        self.vehicle_id = pool.vehicle_id[rows]
        self.cutoff = pool.cutoff[rows]
        self.num_readouts = pool.num_readouts[rows]
        self.columns = pool.columns
        self.categorical = pool.categorical
        self._index = pool._index

    view = Partition.view


class Pool:
    """Concatenation of the fit and calibration partitions."""

    def __init__(self, fit, cal):
        self.n_fit = len(fit.y)
        self.X = np.concatenate([fit.X, cal.X])
        self.y = np.concatenate([fit.y, cal.y])
        self.vehicle_id = np.concatenate([fit.vehicle_id, cal.vehicle_id])
        self.cutoff = np.concatenate([fit.cutoff, cal.cutoff])
        self.num_readouts = np.concatenate([fit.num_readouts, cal.num_readouts])
        self.columns, self.categorical = fit.columns, fit.categorical
        self._index = fit._index

    def split(self, seed, calibration_fraction):
        if seed == "original":
            rows = np.arange(len(self.y))
            return rows[: self.n_fit], rows[self.n_fit:]
        rng = np.random.default_rng(int(seed))
        vehicles = np.unique(self.vehicle_id)
        n_cal = int(round(len(vehicles) * calibration_fraction))
        cal_vehicles = rng.choice(vehicles, n_cal, replace=False)
        is_cal = np.isin(self.vehicle_id, cal_vehicles)
        return np.flatnonzero(~is_cal), np.flatnonzero(is_cal)


def load_data(cfg):
    data = cfg["data"]
    paths = [ROOT / data["fit"], ROOT / data["calibration"]]
    vocab = shared_vocabulary(paths, cfg["categorical_columns"])

    fit = Partition(paths[0], cfg, categories=vocab)
    cal = Partition(paths[1], cfg, categories=vocab)
    val = Partition(ROOT / data["validation"], cfg, categories=vocab)
    assert fit.columns == cal.columns == val.columns, "feature columns differ"

    pool = Pool(fit, cal)
    del fit, cal
    gc.collect()

    assert not set(pool.vehicle_id) & set(val.vehicle_id), "pool/validation overlap"
    return pool, val


# ============================================================
# Weight schemes
# ============================================================

def _bins(train_t, target_t, n_bins):
    pooled = np.concatenate([train_t, target_t])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    inner = edges[1:-1]
    return np.digitize(train_t, inner), np.digitize(target_t, inner), len(edges) - 1


def _hist(bins, n, alpha):
    counts = np.bincount(bins, minlength=n).astype(float) + alpha
    return counts / counts.sum()


def _normalise_within_class(w, y):
    """Rescale so every class keeps its original total weight (= its count)."""
    out = w.copy()
    for c in np.unique(y):
        m = y == c
        out[m] *= m.sum() / out[m].sum()
    return out


def make_weights(scheme, t_train, y, t_val, cfg):
    wc = cfg["weighting"]
    lo, hi = wc["clip"]
    alpha = wc["smoothing"]
    b_tr, b_val, n = _bins(t_train, t_val, wc["n_bins"])

    p_val = _hist(b_val, n, alpha)
    p_fit = _hist(b_tr, n, alpha)

    def marginal():
        return np.clip(p_val[b_tr] / p_fit[b_tr], lo, hi)

    def decorrelate(target):
        w = np.empty(len(y))
        for c in np.unique(y):
            m = y == c
            p_c = _hist(b_tr[m], n, alpha)
            w[m] = np.clip(target[b_tr[m]] / p_c[b_tr[m]], lo, hi)
        return _normalise_within_class(w, y)

    def prior_factor():
        w_m = marginal()
        classes = np.unique(y)
        eff = np.array([w_m[y == c].sum() for c in classes]) / w_m.sum()
        base = np.array([(y == c).mean() for c in classes])
        factor = dict(zip(classes, eff / base))
        return np.array([factor[c] for c in y])

    if scheme == "none":
        w = np.ones(len(y))
    elif scheme == "marginal":
        w = marginal()
    elif scheme == "prior_only":
        w = prior_factor()
    elif scheme == "decorrelate_val":
        w = decorrelate(p_val)
    elif scheme == "decorrelate_fit":
        w = decorrelate(p_fit)
    elif scheme == "decorrelate_val_plus_prior":
        w = decorrelate(p_val) * prior_factor()
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    return w / w.mean()


def weight_diagnostics(w, y, t):
    """ESS, effective class prior, and the weighted time-label association."""
    y48 = (y > 0).astype(int)
    info = {
        "ess_fraction": float(w.sum() ** 2 / np.sum(w ** 2) / len(w)),
        "weighted_auc_cutoff_within48": float(roc_auc_score(y48, t, sample_weight=w)),
    }
    for c in CLASSES:
        info[f"eff_prior_c{c}"] = float(w[y == c].sum() / w.sum())
    return info


# ============================================================
# One (split, arm) run
# ============================================================

def run_arm(arm, fit, cal, val, cfg):
    features = select_features(fit.columns, cfg, "all")
    cats = [c for c in fit.categorical if c in features]

    w = make_weights(arm["weights"], fit.cutoff, fit.y, val.cutoff, cfg)
    info = weight_diagnostics(w, fit.y, fit.cutoff)

    model = LGBMClassifier(objective="multiclass", num_class=5, verbosity=-1,
                           **cfg["model"], **cfg.get("lightgbm_memory", {}))
    start = time.time()
    model.fit(fit.X, fit.y, sample_weight=w,
              feature_name=features, categorical_feature=cats or "auto")
    info["fit_seconds"] = round(time.time() - start, 1)

    order = [list(model.classes_).index(c) for c in CLASSES]
    cal_probs = normalize_probabilities(model.predict_proba(cal.X)[:, order])
    val_probs = normalize_probabilities(model.predict_proba(val.X)[:, order])
    del model
    gc.collect()

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
        rows.append({"arm": arm["name"], "calibration": m.pop("method"), **m, **info})
    return rows, variants


# ============================================================
# Bootstrap averaged over splits
# ============================================================

def multi_split_bootstrap(y, probs, reference, n_resamples, seed):
    """
    probs[split][arm] -> (n, 5). For every Validation resample, compute each
    metric per split, average over splits, then difference against reference.
    """
    rng = np.random.default_rng(seed)
    splits = list(probs)
    arms = list(probs[splits[0]])
    draws = {a: [] for a in arms}
    n = len(y)

    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx] == 4)) < 2:
            continue
        for a in arms:
            per_split = [bootstrap_metrics(y[idx], probs[s][a][idx]) for s in splits]
            draws[a].append(pd.DataFrame(per_split).mean().to_dict())

    rows = []
    ref = pd.DataFrame(draws[reference])
    for a in arms:
        frame = pd.DataFrame(draws[a])
        for metric in frame.columns:
            lo, hi = np.percentile(frame[metric], [2.5, 97.5])
            row = {"arm": a, "metric": metric, "ci_low": lo, "ci_high": hi}
            if a != reference:
                d = frame[metric] - ref[metric]
                d_lo, d_hi = np.percentile(d, [2.5, 97.5])
                row.update({"diff_vs_ref_mean": d.mean(),
                            "diff_ci_low": d_lo, "diff_ci_high": d_hi})
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E02 time-label decoupling")
    parser.add_argument("--config", default="configs/experiments/time_label_decoupling.yaml")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--seeds", nargs="*", help="e.g. original 1 2")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [a for a in cfg["arms"] if not args.arms or a["name"] in args.arms]
    seeds = [str(s) for s in (args.seeds or cfg["split_seeds"])]
    ref = cfg["bootstrap"]["reference_arm"]
    assert ref in [a["name"] for a in arms], f"reference arm {ref} must be run"

    log("Loading data (streaming into float32 matrices)...")
    pool, val = load_data(cfg)
    log(f"pool={pool.X.shape} validation={val.X.shape} "
        f"| matrices: {(pool.X.nbytes + val.X.nbytes) / 1e6:,.0f} MB")

    all_rows, boot_probs = [], {}
    primary = cfg["bootstrap"]["calibration"]

    for seed in seeds:
        fit_rows, cal_rows = pool.split(seed, cfg["calibration_fraction"])
        fit, cal = Subset(pool, fit_rows), Subset(pool, cal_rows)
        assert not set(fit.vehicle_id) & set(cal.vehicle_id), "fit/cal overlap"
        assert set(np.unique(fit.y)) == set(np.unique(cal.y)) == set(CLASSES)
        log(f"Split {seed}: fit={len(fit.y)} calibration={len(cal.y)}")

        boot_probs[seed] = {}
        for arm in arms:
            log(f"  Arm {arm['name']} (weights={arm['weights']})")
            rows, variants = run_arm(arm, fit, cal, val, cfg)
            for r in rows:
                r["split"] = seed
            all_rows.extend(rows)
            boot_probs[seed][arm["name"]] = variants[primary]
            pd.DataFrame(all_rows).to_csv(out_dir / "summary.csv", index=False)

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

    diag_cols = ["ess_fraction", "weighted_auc_cutoff_within48"] + \
                [f"eff_prior_c{c}" for c in CLASSES]
    weights_table = (summary[summary.calibration == "uncalibrated"]
                     .groupby("arm", sort=False)[diag_cols].mean())
    weights_table.to_csv(out_dir / "weight_diagnostics.csv")

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
    print("\n" + "=" * 100)
    print("WEIGHT DIAGNOSTICS (training, averaged over splits)")
    print("=" * 100)
    print(weights_table.round(4).to_string())

    print("\n" + "=" * 100)
    print("VALIDATION — mean (std) over splits")
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
