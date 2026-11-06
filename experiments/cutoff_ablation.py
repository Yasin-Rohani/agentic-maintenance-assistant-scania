"""
Cutoff-time shortcut ablation for the SCANIA Component X multiclass model.

Question
--------
Does the multiclass LightGBM rely on *when* a snapshot was taken
(cutoff_time, num_readouts, and duration encoded in cumulative sensor
counters) rather than on the vehicle's condition?

Design
------
Every arm uses the same vehicle-disjoint fit / calibration / Validation
partitions and the same LightGBM hyper-parameters as
src/train_multiclass_lightgbm.py. Arms differ only in the input features
and, optionally, in importance weights that align the training cutoff_time
distribution with Validation.

Diagnostics (no model comparison needed):
  1. cutoff_time by class in fit vs Validation.
  2. Univariate ROC-AUC of cutoff_time for "failure within 48" per split.
     A flip in direction between train and Validation is direct evidence
     of a sampling shortcut.
  3. Duration probe: R^2 of predicting cutoff_time from non-time features.
     A high R^2 means removing cutoff_time alone cannot remove the shortcut.

Usage (from the repository root)
-----
    python experiments/cutoff_ablation.py
    python experiments/cutoff_ablation.py --config configs/experiments/cutoff_ablation.yaml
    python experiments/cutoff_ablation.py --arms A_full D_time_only   # subset
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm
import sklearn
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reuse the project's existing, already-validated logic so that this
# experiment is directly comparable with earlier results.
from calibrate_multiclass_lightgbm import (  # noqa: E402
    apply_calibrators,
    evaluate,
    normalize_probabilities,
    train_isotonic_calibrators,
    train_sigmoid_calibrators,
    vehicle_weights,
)
from capacity_constrained_decision import (  # noqa: E402
    evaluate_policy,
    optimize_decisions,
)
from train_multiclass_lightgbm import COST_MATRIX, CLASSES  # noqa: E402


# ============================================================
# Configuration and bookkeeping
# ============================================================

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# ============================================================
# Data
# ============================================================

class Partition:
    """
    One partition as a single C-contiguous float32 matrix plus label arrays.

    The parquet file is streamed column-chunk by column-chunk straight into a
    pre-allocated float32 array, so the wide feature table is never held as a
    pandas DataFrame. LightGBM consumes float32 NumPy without an extra
    conversion copy. Together this keeps peak memory low on 8 GB machines.

    Categorical columns are stored as integer codes (NaN = missing/unseen),
    with vocabularies learned on the fit partition only.
    """

    META = ["vehicle_id", "class_label", "cutoff_time", "num_readouts"]

    def __init__(self, path: Path, cfg: dict, categories=None,
                 keep_vehicles=None, chunk_columns: int = 64):
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        excluded = set(cfg["excluded_columns"])

        meta = parquet.read(columns=self.META).to_pandas()
        rows = np.arange(len(meta))
        if keep_vehicles is not None:
            rows = np.flatnonzero(meta["vehicle_id"].isin(keep_vehicles).to_numpy())
            meta = meta.iloc[rows].reset_index(drop=True)

        self.y = meta["class_label"].to_numpy(dtype=int)
        self.vehicle_id = meta["vehicle_id"].to_numpy()
        self.cutoff = meta["cutoff_time"].to_numpy(dtype=float)
        self.num_readouts = meta["num_readouts"].to_numpy(dtype=float)

        self.columns = [c for c in names if c not in excluded]
        self.categorical = [c for c in cfg["categorical_columns"] if c in self.columns]
        self._index = {c: i for i, c in enumerate(self.columns)}
        self.categories = categories if categories is not None else {}

        self.X = np.empty((len(rows), len(self.columns)), dtype=np.float32)
        subset = None if keep_vehicles is None else rows

        for start in range(0, len(self.columns), chunk_columns):
            chunk = self.columns[start:start + chunk_columns]
            table = parquet.read(columns=chunk)
            for column in chunk:
                values = table.column(column)
                if column in self.categorical:
                    values = pd.Series(values.to_pandas()).fillna("MISSING").astype(str)
                    if column not in self.categories:        # fit only
                        self.categories[column] = sorted(values.unique())
                    codes = pd.Categorical(
                        values, categories=self.categories[column]).codes
                    array = np.where(codes < 0, np.nan, codes).astype(np.float32)
                else:
                    array = values.to_numpy(zero_copy_only=False).astype(
                        np.float32, copy=False)
                self.X[:, self._index[column]] = array if subset is None else array[subset]
            del table
            pa.default_memory_pool().release_unused()

    def view(self, columns):
        """Full matrix without copying when every column is used."""
        if list(columns) == self.columns:
            return self.X
        return np.ascontiguousarray(self.X[:, [self._index[c] for c in columns]])

    def nbytes_mb(self) -> float:
        return self.X.nbytes / 1e6


def load_partitions(cfg: dict):
    data = cfg["data"]

    # Optional low-memory fallback: keep a random subset of fit *vehicles*.
    keep = None
    fraction = cfg.get("memory", {}).get("fit_vehicle_fraction", 1.0)
    if fraction < 1.0:
        ids = pq.read_table(ROOT / data["fit"], columns=["vehicle_id"])
        vehicles = np.unique(ids.column("vehicle_id").to_numpy())
        rng = np.random.default_rng(cfg["model"]["random_state"])
        keep = rng.choice(vehicles, int(len(vehicles) * fraction), replace=False)

    fit = Partition(ROOT / data["fit"], cfg, keep_vehicles=keep)
    cal = Partition(ROOT / data["calibration"], cfg, categories=fit.categories)
    val = Partition(ROOT / data["validation"], cfg, categories=fit.categories)

    assert fit.columns == cal.columns == val.columns, "feature columns differ"
    ids = {"fit": set(fit.vehicle_id), "cal": set(cal.vehicle_id),
           "val": set(val.vehicle_id)}
    assert not ids["fit"] & ids["cal"], "fit/calibration vehicle overlap"
    assert not ids["fit"] & ids["val"], "fit/validation vehicle overlap"
    assert not ids["cal"] & ids["val"], "calibration/validation vehicle overlap"
    for part in (fit, cal, val):
        assert set(np.unique(part.y)) == set(CLASSES)

    return fit, cal, val


def select_features(all_columns, cfg: dict, mode: str) -> list[str]:
    excluded = set(cfg["excluded_columns"])
    time_cols = list(cfg["time_columns"])
    cats = list(cfg["categorical_columns"])
    base = [c for c in all_columns if c not in excluded]

    if mode == "all":
        return base
    if mode == "no_cutoff":
        return [c for c in base if c != "cutoff_time"]
    if mode == "no_time":
        return [c for c in base if c not in time_cols]
    if mode == "time_only":
        return [c for c in time_cols if c in base]
    if mode == "spec_only":
        return [c for c in cats if c in base]
    raise ValueError(f"Unknown feature mode: {mode}")


# ============================================================
# Importance re-weighting on cutoff_time
# ============================================================

def cutoff_density_weights(train_cutoff, target_cutoff, n_bins, clip):
    """
    Histogram density-ratio weights w(t) = p_target(t) / p_train(t).

    Bins are quantiles of the pooled cutoff times. Uses only the
    target cutoff times (unlabelled covariate), never target labels.
    """
    pooled = np.concatenate([train_cutoff, target_cutoff])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    train_bins = np.digitize(train_cutoff, edges[1:-1])
    target_bins = np.digitize(target_cutoff, edges[1:-1])
    n = len(edges) - 1

    p_train = np.bincount(train_bins, minlength=n) / len(train_bins)
    p_target = np.bincount(target_bins, minlength=n) / len(target_bins)

    ratio = np.divide(p_target, p_train, out=np.zeros(n), where=p_train > 0)
    weights = np.clip(ratio[train_bins], *clip)
    weights = weights / weights.mean()

    ess = weights.sum() ** 2 / np.sum(weights ** 2)
    return weights, float(ess)


# ============================================================
# Diagnostics
# ============================================================

def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def cutoff_diagnostics(parts: dict):
    by_class, univariate = [], {}
    for name, part in parts.items():
        frame = pd.DataFrame({"class_label": part.y, "cutoff_time": part.cutoff})
        for class_id, series in frame.groupby("class_label")["cutoff_time"]:
            by_class.append({
                "partition": name,
                "class_label": int(class_id),
                "n": int(series.size),
                "mean_cutoff": float(series.mean()),
                "median_cutoff": float(series.median()),
            })

        y48 = (part.y > 0).astype(int)
        y4 = (part.y == 4).astype(int)
        univariate[name] = {
            "cutoff_time_auc_within48": safe_auc(y48, part.cutoff),
            "cutoff_time_auc_class4": safe_auc(y4, part.cutoff),
            "num_readouts_auc_within48": safe_auc(y48, part.num_readouts),
            "num_readouts_auc_class4": safe_auc(y4, part.num_readouts),
        }

    return pd.DataFrame(by_class), univariate


def duration_probe(fit, cal, val, cfg):
    """Predict cutoff_time from all non-time features."""
    features = select_features(fit.columns, cfg, "no_time")
    cats = [c for c in fit.categorical if c in features]

    params = dict(cfg["model"])
    params["n_estimators"] = cfg["duration_probe"]["n_estimators"]
    model = LGBMRegressor(objective="regression", verbosity=-1, **params)
    model.fit(fit.view(features), fit.cutoff,
              feature_name=features, categorical_feature=cats or "auto")

    result = {
        "r2_calibration": float(r2_score(cal.cutoff, model.predict(cal.view(features)))),
        "r2_validation": float(r2_score(val.cutoff, model.predict(val.view(features)))),
    }
    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=features,
    ).sort_values(ascending=False)

    del model
    gc.collect()
    return result, importance


# ============================================================
# Metrics
# ============================================================

def extended_metrics(name, y_true, probs, capacity_fractions):
    metrics, _, _ = evaluate(name, y_true, probs)
    probs = normalize_probabilities(probs)

    metrics["roc_auc_class4"] = safe_auc((y_true == 4).astype(int), probs[:, 4])

    n = len(y_true)
    for fraction in capacity_fractions:
        capacity = int(np.floor(n * fraction))
        decisions, benefit = optimize_decisions(probs, capacity)
        row = evaluate_policy(name, fraction, y_true, decisions.astype(int), benefit)
        tag = f"cap{int(round(fraction * 100))}"
        metrics[f"cost_{tag}"] = row["official_cost"]
        metrics[f"false_alerts_{tag}"] = row["false_alerts"]
        metrics[f"recall_{tag}"] = row["binary_recall"]

    metrics["cost_no_action"] = float(COST_MATRIX[y_true, 0].sum())
    return metrics


def bootstrap_metrics(y_true, probs):
    """Metrics recomputed on every bootstrap resample."""
    probs = normalize_probabilities(probs)
    y48 = (y_true > 0).astype(int)
    y4 = (y_true == 4).astype(int)
    decisions = (probs @ COST_MATRIX).argmin(axis=1)
    return {
        "ap_within48": average_precision_score(y48, probs[:, 1:].sum(axis=1)),
        "ap_class4": average_precision_score(y4, probs[:, 4]),
        "cost_min_expected": COST_MATRIX[y_true, decisions].sum(),
    }


def paired_bootstrap(y_true, arm_probs, reference, n_resamples, seed):
    """Percentile CIs for each arm and for (arm - reference), paired by vehicle."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    names = list(arm_probs)
    draws = {name: [] for name in names}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx] == 4)) < 2:
            continue
        for name in names:
            draws[name].append(bootstrap_metrics(y_true[idx], arm_probs[name][idx]))

    rows = []
    for name in names:
        frame = pd.DataFrame(draws[name])
        ref = pd.DataFrame(draws[reference]) if reference in draws else None
        for metric in frame.columns:
            low, high = np.percentile(frame[metric], [2.5, 97.5])
            row = {"arm": name, "metric": metric,
                   "ci_low": float(low), "ci_high": float(high)}
            if ref is not None and name != reference:
                diff = frame[metric] - ref[metric]
                d_low, d_high = np.percentile(diff, [2.5, 97.5])
                row.update({
                    "diff_vs_ref_mean": float(diff.mean()),
                    "diff_ci_low": float(d_low),
                    "diff_ci_high": float(d_high),
                })
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# One arm
# ============================================================

def run_arm(arm, fit, cal, val, cfg):
    features = select_features(fit.columns, cfg, arm["features"])
    cats = [c for c in fit.categorical if c in features]
    info = {"arm": arm["name"], "n_features": len(features)}

    sample_weight = None
    if arm["reweight"]:
        rw = cfg["reweighting"]
        sample_weight, ess = cutoff_density_weights(
            fit.cutoff, val.cutoff, rw["n_bins"], tuple(rw["clip"]))
        info["effective_sample_size"] = ess
        info["ess_fraction"] = ess / len(fit.y)

    model = LGBMClassifier(objective="multiclass", num_class=5, verbosity=-1,
                           **cfg["model"], **cfg.get("lightgbm_memory", {}))
    start = time.time()
    X_fit = fit.view(features)
    model.fit(X_fit, fit.y, sample_weight=sample_weight,
              feature_name=features, categorical_feature=cats or "auto")
    del X_fit
    gc.collect()
    info["fit_seconds"] = round(time.time() - start, 1)

    order = [list(model.classes_).index(c) for c in CLASSES]
    cal_probs = normalize_probabilities(model.predict_proba(cal.view(features))[:, order])
    val_probs = normalize_probabilities(model.predict_proba(val.view(features))[:, order])

    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=features,
    ).sort_values(ascending=False)
    del model
    gc.collect()

    weights = vehicle_weights(cal.vehicle_id)
    variants = {"uncalibrated": val_probs}
    methods = cfg["calibration_methods"]
    if "sigmoid" in methods:
        variants["sigmoid"] = apply_calibrators(
            val_probs, train_sigmoid_calibrators(cal_probs, cal.y, weights), "sigmoid")
    if "isotonic" in methods:
        variants["isotonic"] = apply_calibrators(
            val_probs, train_isotonic_calibrators(cal_probs, cal.y, weights), "isotonic")

    rows = []
    for method, probs in variants.items():
        metrics = extended_metrics(method, val.y, probs, cfg["capacity_fractions"])
        metrics = {"arm": arm["name"], "calibration": metrics.pop("method"), **metrics}
        metrics.update(info)
        rows.append(metrics)

    return rows, variants, importance


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default="configs/experiments/cutoff_ablation.yaml")
    parser.add_argument("--arms", nargs="*", help="Run only these arm names.")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading partitions (streaming into float32 matrices)...")
    fit, cal, val = load_partitions(cfg)
    gc.collect()
    total = fit.nbytes_mb() + cal.nbytes_mb() + val.nbytes_mb()
    log(f"fit={fit.X.shape} calibration={cal.X.shape} validation={val.X.shape} "
        f"| feature matrices: {total:,.0f} MB")

    # ---------------- diagnostics ----------------
    log("Cutoff-time diagnostics...")
    by_class, univariate = cutoff_diagnostics(
        {"fit": fit, "calibration": cal, "validation": val})
    by_class.to_csv(out_dir / "cutoff_by_class.csv", index=False)

    diagnostics = {"univariate_auc": univariate}

    if cfg["duration_probe"]["enabled"]:
        log("Duration probe: predicting cutoff_time from non-time features...")
        probe, probe_imp = duration_probe(fit, cal, val, cfg)
        diagnostics["duration_probe"] = probe
        probe_imp.head(cfg["top_importances"]).to_csv(
            out_dir / "duration_probe_importance.csv", header=["gain"])
        log(f"Duration probe R^2: {probe}")

    # ---------------- arms ----------------
    arms = cfg["arms"]
    if args.arms:
        arms = [a for a in arms if a["name"] in set(args.arms)]

    all_rows, boot_probs = [], {}
    primary = cfg["bootstrap"]["calibration"]

    for arm in arms:
        log(f"Arm {arm['name']} (features={arm['features']}, reweight={arm['reweight']})")
        rows, variants, importance = run_arm(arm, fit, cal, val, cfg)
        all_rows.extend(rows)
        boot_probs[arm["name"]] = variants[primary]
        importance.head(cfg["top_importances"]).to_csv(
            out_dir / f"importance_{arm['name']}.csv", header=["gain"])
        pd.DataFrame(all_rows).to_csv(out_dir / "summary.csv", index=False)
        gc.collect()

    summary = pd.DataFrame(all_rows)

    # ---------------- bootstrap ----------------
    log("Paired bootstrap...")
    bs = cfg["bootstrap"]
    boot = paired_bootstrap(
        val.y, boot_probs,
        bs["reference_arm"], bs["n_resamples"], bs["random_state"])
    boot.to_csv(out_dir / "bootstrap.csv", index=False)

    # ---------------- run info ----------------
    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "config": cfg,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))

    # ---------------- console report ----------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 90)
    print("CUTOFF TIME BY CLASS")
    print("=" * 90)
    print(by_class.pivot(index="class_label", columns="partition",
                         values="mean_cutoff").round(1).to_string())

    print("\nUNIVARIATE ROC-AUC (0.5 = no signal; flip across partitions = shortcut)")
    print(json.dumps(univariate, indent=2))

    if "duration_probe" in diagnostics:
        print("\nDURATION PROBE (R^2 predicting cutoff_time from non-time features)")
        print(json.dumps(diagnostics["duration_probe"], indent=2))

    keep = ["arm", "calibration", "n_features", "pr_auc_within48", "roc_auc_within48",
            "pr_auc_class4", "roc_auc_class4", "brier", "log_loss",
            "official_cost_min_expected", "cost_cap10", "cost_cap30", "false_alerts_cap30"]
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY (Validation)")
    print("=" * 90)
    print(summary[[c for c in keep if c in summary]].round(4).to_string(index=False))

    print(f"\nPAIRED BOOTSTRAP vs {bs['reference_arm']} ({primary} probabilities)")
    print(boot.round(4).to_string(index=False))

    print(f"\nOutputs written to: {out_dir}")
    log("STATUS: SUCCESS")


if __name__ == "__main__":
    main()
