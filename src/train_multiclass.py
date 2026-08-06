"""
train_multiclass.py – Multi-class LightGBM baseline for Open-Set DGA Detection.

Improvements over train_baseline.py (binary):
  1. 19-class classifier (benign + 18 DGA families) — finer-grained softmax
     makes MSP a more effective OOD scorer.
  2. KNN OOD scorer on the 35-dim feature space: unseen domains that fall
     in low-density regions of training space get high OOD scores.

OOD scorers:
  - msp          : 1 - max(p_i)           — confidence-based
  - energy       : -T * logsumexp(log p)  — energy-based
  - knn          : mean dist to k nearest training neighbours (euclidean,
                   standardized features)  — density-based

Usage:
  python src/train_multiclass.py --run_dir data/processed
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features import FEATURE_NAMES, extract_features_batch
from ood_utils import ood_metrics, print_ood_metrics
from logger import get_logger

log = get_logger(__name__)


# ── feature extraction (reuses cache written by train_baseline.py) ──────────

def _featurise(df: pd.DataFrame, cache_dir: Path | None, tag: str) -> np.ndarray:
    if cache_dir is not None:
        cf = cache_dir / f"{tag}_feats.npy"
        if cf.exists():
            log.info(f"  [cache hit] {cf}")
            return np.load(str(cf))

    domains = df["domain"].tolist()
    BATCH = 50_000
    parts = []
    for i in range(0, len(domains), BATCH):
        parts.append(extract_features_batch(domains[i:i + BATCH]))
        done = min(i + BATCH, len(domains))
        log.info(f"    featurised {done:>8,} / {len(domains):,}")
    X = np.vstack(parts)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_dir / f"{tag}_feats.npy"), X)
    return X


# ── OOD scorers ──────────────────────────────────────────────────────────────

def msp_score(proba: np.ndarray) -> np.ndarray:
    """1 - max(p_i). Higher = more OOD."""
    return 1.0 - proba.max(axis=1)


def energy_score(proba: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy-based OOD score. Higher = more OOD."""
    log_p = np.log(np.clip(proba, 1e-30, None))
    lse = T * np.log(np.sum(np.exp(log_p / T), axis=1))
    return -lse


def build_knn(X_train: np.ndarray, k: int = 5, subsample: int = 50_000, seed: int = 42):
    """Fit a KNN on a (possibly subsampled) training set after standardisation."""
    rng = np.random.default_rng(seed)
    if len(X_train) > subsample:
        idx = rng.choice(len(X_train), size=subsample, replace=False)
        X_ref = X_train[idx]
    else:
        X_ref = X_train
    scaler = StandardScaler()
    X_ref_s = scaler.fit_transform(X_ref)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1, algorithm="auto")
    nn.fit(X_ref_s)
    return nn, scaler


def knn_score(nn: NearestNeighbors, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Mean distance to k nearest training neighbours. Higher = more OOD."""
    X_s = scaler.transform(X)
    dists, _ = nn.kneighbors(X_s)
    return dists.mean(axis=1)


# ── helpers ──────────────────────────────────────────────────────────────────

def _csv_paths(run_dir: str) -> dict[str, Path]:
    rd = Path(run_dir)
    return {
        "train": rd / "known" / "train.csv",
        "val": rd / "known" / "val.csv",
        "test_known": rd / "known" / "test_known.csv",
        "unknown_family": rd / "unknown_family" / "test_unknown_family.csv",
        "unknown_ood": rd / "unknown_ood" / "test_unknown_ood.csv",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="data/processed")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_estimators", type=int, default=800,
                    help="Max boosting rounds (each round = 1 tree per class)")
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--energy_T", type=float, default=1.0)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--knn_subsample", type=int, default=50_000)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"multiclass_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.run_dir) / "_feature_cache"
    csvs = _csv_paths(args.run_dir)
    for p in csvs.values():
        if not p.exists():
            sys.exit(f"Missing CSV: {p}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    log.info("Loading CSVs …")
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}
    for k, df in dfs.items():
        log.info(f"  {k:20s}: {len(df):>8,} rows")

    # ── 2. Class label encoding (19 classes) ─────────────────────────────
    le = LabelEncoder()
    le.fit(dfs["train"]["class_label"].values)
    n_classes = len(le.classes_)
    benign_idx = int(np.where(le.classes_ == "benign")[0][0])
    log.info(f"\n  {n_classes} classes: {list(le.classes_)}")

    y = {
        "train": le.transform(dfs["train"]["class_label"].values),
        "val": le.transform(dfs["val"]["class_label"].values),
        "test_known": le.transform(dfs["test_known"]["class_label"].values),
    }

    # ── 3. Feature extraction (uses cache if available) ───────────────────
    log.info("\nExtracting features …")
    Xs = {}
    for k, df in dfs.items():
        log.info(f"  [{k}]")
        Xs[k] = _featurise(df, cache_dir, k)

    # ── 4. Train multi-class LightGBM ────────────────────────────────────
    log.info("\nTraining multi-class LightGBM …")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        metric="multi_logloss",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        class_weight="balanced",
        verbose=-1,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(
        Xs["train"], y["train"],
        eval_set=[(Xs["val"], y["val"])],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    best_iter = model.best_iteration_
    log.info(f"  Best iteration: {best_iter}")

    model.booster_.save_model(str(out_dir / "model.txt"))

    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(str(out_dir / "feature_importance.csv"), index=False)
    log.info("\n  Top-10 features:")
    log.info(imp.head(10).to_string(index=False))

    # ── 5. Classification accuracy ────────────────────────────────────────
    results = {}
    log.info("\n" + "=" * 65)
    log.info("  Classification Results")
    log.info("=" * 65)
    for split in ["val", "test_known"]:
        proba = model.predict_proba(Xs[split])
        y_pred_mc = proba.argmax(axis=1)
        mc_acc = accuracy_score(y[split], y_pred_mc)

        y_bin = (y[split] != benign_idx).astype(int)
        p_dga = 1.0 - proba[:, benign_idx]
        y_pred_bin = (p_dga >= 0.5).astype(int)
        bin_acc = accuracy_score(y_bin, y_pred_bin)
        bin_f1 = f1_score(y_bin, y_pred_bin)
        bin_auc = roc_auc_score(y_bin, p_dga)
        log.info(f"  [{split}]  MC-acc={mc_acc:.4f}  Bin-acc={bin_acc:.4f}"
              f"  F1={bin_f1:.4f}  AUC={bin_auc:.4f}")
        results[f"cls_{split}"] = {
            "mc_accuracy": float(mc_acc),
            "bin_accuracy": float(bin_acc),
            "bin_f1": float(bin_f1),
            "bin_roc_auc": float(bin_auc),
        }

    # ── 6. Build KNN index ────────────────────────────────────────────────
    log.info(f"\nBuilding KNN index (k={args.knn_k}, subsample={args.knn_subsample:,}) …")
    nn, scaler = build_knn(Xs["train"], k=args.knn_k, subsample=args.knn_subsample)
    log.info("  Done.")

    # ── 7. Pre-compute model probabilities ───────────────────────────────
    log.info("\nScoring all splits …")
    probas = {}
    for k in ["test_known", "unknown_family", "unknown_ood"]:
        probas[k] = model.predict_proba(Xs[k])

    # ── 8. OOD evaluation ────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  OOD Detection Evaluation")
    log.info("=" * 65)

    scorer_fns = {
        "msp": lambda k: msp_score(probas[k]),
        "energy": lambda k: energy_score(probas[k], T=args.energy_T),
        "knn": lambda k: knn_score(nn, scaler, Xs[k]),
    }

    ood_splits = ["unknown_family", "unknown_ood"]

    for sname, sfn in scorer_fns.items():
        log.info(f"\n── {sname} ──")
        id_scores = sfn("test_known")
        for split in ood_splits:
            ood_s = sfn(split)
            m = ood_metrics(id_scores, ood_s)
            results[f"ood_{sname}_{split}"] = m
            print_ood_metrics(m, split)

            pd.DataFrame({
                "domain": dfs[split]["domain"].values,
                "ood_score": ood_s,
            }).to_csv(str(out_dir / f"scores_{sname}_{split}.csv"), index=False)

        pd.DataFrame({
            "domain": dfs["test_known"]["domain"].values,
            "ood_score": id_scores,
        }).to_csv(str(out_dir / f"scores_{sname}_known.csv"), index=False)

    # ── 9. Save + summary ────────────────────────────────────────────────
    results["params"] = vars(args)
    results["best_iteration"] = best_iter
    results["n_classes"] = n_classes
    results["classes"] = list(le.classes_)
    with open(str(out_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("\n" + "=" * 70)
    log.info("  SUMMARY")
    log.info("=" * 70)
    ck = results["cls_test_known"]
    log.info(f"  Binary  (test_known): Acc={ck['bin_accuracy']:.4f}  F1={ck['bin_f1']:.4f}"
          f"  AUC={ck['bin_roc_auc']:.4f}")
    log.info(f"  19-class(test_known): Acc={ck['mc_accuracy']:.4f}")
    for sn in ["msp", "energy", "knn"]:
        log.info(f"\n  OOD ({sn}):")
        for sl in ood_splits:
            m = results[f"ood_{sn}_{sl}"]
            log.info(f"    {sl:20s}: AUROC={m['auroc']:.4f}  AUPR-OUT={m['aupr_out']:.4f}"
                  f"  FPR@95={m['fpr_at_tpr']:.4f}")

    log.info(f"\nAll results → {out_dir}")


if __name__ == "__main__":
    main()
