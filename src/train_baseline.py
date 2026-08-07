"""
train_baseline.py – Binary LightGBM baseline for Open-Set DGA Detection.

Pipeline:
  1. Load train / val / test_known / unknown_family / unknown_ood CSVs.
  2. Extract 35 lexical features from domain strings.
  3. Train a binary LightGBM classifier  (benign=0, dga=1).
  4. Compute OOD scores on all evaluation splits:
       - MSP  (Max Softmax Probability) : 1 - max(p)
       - Energy score                   : -T * log(sum(exp(logit/T)))
  5. Save scored CSVs and run evaluate_ood.py metrics.
  6. Save model + feature importance.

Usage:
  python src/train_baseline.py --run_dir data/processed
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from features import FEATURE_NAMES, extract_features_batch
from logger import get_logger

log = get_logger(__name__)

# ── paths ───────────────────────────────────────────────────────────────────

def _csv_paths(run_dir: str) -> dict[str, Path]:
    rd = Path(run_dir)
    return {
        "train": rd / "known" / "train.csv",
        "val": rd / "known" / "val.csv",
        "test_known": rd / "known" / "test_known.csv",
        "unknown_family": rd / "unknown_family" / "test_unknown_family.csv",
        "unknown_ood": rd / "unknown_ood" / "test_unknown_ood.csv",
    }


# ── feature extraction with caching ────────────────────────────────────────

def _featurise(df: pd.DataFrame, cache_dir: Path | None, tag: str) -> np.ndarray:
    """Extract features; optionally cache as .npy for reuse."""
    if cache_dir is not None:
        cache_file = cache_dir / f"{tag}_feats.npy"
        if cache_file.exists():
            log.info(f"  [cache hit] {cache_file}")
            return np.load(str(cache_file))

    domains = df["domain"].tolist()
    BATCH = 50_000
    parts = []
    for i in range(0, len(domains), BATCH):
        batch = domains[i:i + BATCH]
        parts.append(extract_features_batch(batch))
        done = min(i + BATCH, len(domains))
        log.info(f"    featurised {done:>8,} / {len(domains):,}")
    X = np.vstack(parts)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_dir / f"{tag}_feats.npy"), X)
    return X


# ── OOD scoring functions ──────────────────────────────────────────────────

def _msp_score(proba: np.ndarray) -> np.ndarray:
    """Max Softmax Probability OOD score: 1 - max(p).
    Higher = more likely OOD."""
    return 1.0 - proba.max(axis=1)


def _energy_score(raw_logit: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score from raw LightGBM log-odds. Higher = more OOD.
    Symmetric two-class logits [-z/2, z/2]: E = -T·log(2·cosh(z/(2T))).
    Numerically stable via: log(2·cosh(w)) = |w| + log1p(exp(-2|w|))."""
    w = np.abs(raw_logit) / (2.0 * T)
    return -T * (w + np.log1p(np.exp(-2.0 * w)))


# ── evaluation helpers ──────────────────────────────────────────────────────

def _evaluate_binary(y_true, y_pred, y_proba, label: str):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_proba)
    log.info(f"\n{'=' * 60}")
    log.info(f"  Binary classification – {label}")
    log.info(f"{'=' * 60}")
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  F1       : {f1:.4f}")
    log.info(f"  ROC-AUC  : {auc:.4f}")
    log.info(classification_report(y_true, y_pred, target_names=["benign", "dga"]))
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def _ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_target: float = 0.95):
    """Compute AUROC, AUPR-OUT, AUPR-IN, FPR@TPR using pandas (same logic as evaluate_ood.py)."""
    id_s = pd.Series(id_scores)
    ood_s = pd.Series(ood_scores)

    # AUROC via Mann-Whitney U
    n_id, n_ood = len(id_s), len(ood_s)
    all_df = pd.concat([
        pd.DataFrame({"score": id_s.values, "is_ood": 0}),
        pd.DataFrame({"score": ood_s.values, "is_ood": 1}),
    ], ignore_index=True)
    all_df["rank"] = all_df["score"].rank(method="average")
    rank_sum_ood = all_df.loc[all_df["is_ood"] == 1, "rank"].sum()
    auroc = (rank_sum_ood - n_ood * (n_ood + 1) / 2) / (n_id * n_ood)

    # FPR@TPR
    threshold = float(ood_s.quantile(1.0 - tpr_target, interpolation="linear"))
    realized_tpr = float((ood_s >= threshold).mean())
    fpr = float((id_s >= threshold).mean())

    # AUPR-OUT / AUPR-IN  (sklearn handles ties correctly via threshold averaging)
    scores_all = np.concatenate([id_scores, ood_scores])
    labels_ood = np.array([0] * n_id + [1] * n_ood, dtype=int)
    aupr_out = float(average_precision_score(labels_ood, scores_all))
    aupr_in  = float(average_precision_score(1 - labels_ood, -scores_all))

    # Precision@TPR: precision at the operating point used for FPR@TPR
    tp = realized_tpr * n_ood
    fp = fpr * n_id
    precision_at_tpr = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "auroc": float(auroc),
        "aupr_out": float(aupr_out),
        "aupr_in": float(aupr_in),
        "fpr_at_tpr": float(fpr),
        "precision_at_tpr": float(precision_at_tpr),
        "tpr_target": tpr_target,
        "realized_tpr": float(realized_tpr),
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="data/processed",
                    help="Path to processed dataset directory")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: baseline_out/<timestamp>)")
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--energy_T", type=float, default=1.0, help="Temperature for energy score")
    ap.add_argument("--no_cache", action="store_true", help="Disable feature caching")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.run_dir) / "_feature_cache"

    csvs = _csv_paths(args.run_dir)
    for k, p in csvs.items():
        if not p.exists():
            sys.exit(f"Missing CSV: {p}")

    # ── 1. Load data ────────────────────────────────────────────────────────
    log.info("Loading CSVs …")
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}
    for k, df in dfs.items():
        log.info(f"  {k:20s}: {len(df):>8,} rows")

    # ── 2. Feature extraction ───────────────────────────────────────────────
    log.info("\nExtracting features …")
    Xs, ys = {}, {}
    for k, df in dfs.items():
        log.info(f"  [{k}]")
        Xs[k] = _featurise(df, cache_dir, k)
        # binary label: benign=0, dga/ood=1
        ys[k] = (df["label"] != "benign").astype(int).values

    # ── 3. Train LightGBM ──────────────────────────────────────────────────
    log.info("\nTraining LightGBM …")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "class_weight": "balanced",
        "verbose": -1,
        "n_jobs": -1,
        "random_state": args.seed,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        Xs["train"], ys["train"],
        eval_set=[(Xs["val"], ys["val"])],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    best_iter = model.best_iteration_
    log.info(f"  Best iteration: {best_iter}")

    # save model
    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))
    log.info(f"  Model saved → {model_path}")

    # feature importance
    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(str(out_dir / "feature_importance.csv"), index=False)
    log.info(f"\n  Top-10 features:")
    log.info(imp.head(10).to_string(index=False))

    # ── 4. Evaluate binary classification ───────────────────────────────────
    results = {}
    for split in ["val", "test_known"]:
        proba = model.predict_proba(Xs[split])[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[f"binary_{split}"] = _evaluate_binary(ys[split], pred, proba, split)

    # ── 5. OOD scoring ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  OOD Detection Evaluation")
    log.info("=" * 60)

    # ID scores come from test_known
    proba_known = model.predict_proba(Xs["test_known"])
    raw_known = model.booster_.predict(Xs["test_known"], raw_score=True)  # raw log-odds for energy

    ood_splits = {
        "unknown_family": "unknown_family",
        "unknown_ood": "unknown_ood",
    }

    for score_name in ["msp", "energy"]:
        log.info(f"\n── Score: {score_name} ──")
        id_scores = (_msp_score(proba_known) if score_name == "msp"
                     else _energy_score(raw_known, T=args.energy_T))

        for split_label, split_key in ood_splits.items():
            if score_name == "msp":
                proba_ood = model.predict_proba(Xs[split_key])
                ood_scores = _msp_score(proba_ood)
            else:
                raw_ood = model.booster_.predict(Xs[split_key], raw_score=True)
                ood_scores = _energy_score(raw_ood, T=args.energy_T)

            metrics = _ood_metrics(id_scores, ood_scores)
            results[f"ood_{score_name}_{split_label}"] = metrics

            log.info(f"\n  {split_label}:")
            log.info(f"    AUROC      : {metrics['auroc']:.6f}")
            log.info(f"    AUPR-OUT   : {metrics['aupr_out']:.6f}")
            log.info(f"    AUPR-IN    : {metrics['aupr_in']:.6f}")
            log.info(f"    FPR@TPR=0.95: {metrics['fpr_at_tpr']:.6f} (realized TPR={metrics['realized_tpr']:.6f})")

            # save scored CSV for evaluate_ood.py compatibility
            csv_out = out_dir / f"scores_{score_name}_{split_label}.csv"
            pd.DataFrame({
                "domain": dfs[split_key]["domain"].values,
                "ood_score": ood_scores,
            }).to_csv(str(csv_out), index=False)

        # save known scores too
        csv_known_out = out_dir / f"scores_{score_name}_known.csv"
        pd.DataFrame({
            "domain": dfs["test_known"]["domain"].values,
            "ood_score": id_scores,
        }).to_csv(str(csv_known_out), index=False)

    # ── 6. Save results JSON ────────────────────────────────────────────────
    results["params"] = {k: v for k, v in vars(args).items()}
    results["best_iteration"] = best_iter
    results["n_features"] = len(FEATURE_NAMES)
    results_path = out_dir / "results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nAll results saved → {out_dir}")

    # ── 7. Summary table ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  SUMMARY")
    log.info("=" * 70)
    log.info(f"  Binary classification (test_known):")
    bk = results["binary_test_known"]
    log.info(f"    Accuracy={bk['accuracy']:.4f}  F1={bk['f1']:.4f}  AUC={bk['roc_auc']:.4f}")

    for sn in ["msp", "energy"]:
        log.info(f"\n  OOD detection ({sn}):")
        for sl in ["unknown_family", "unknown_ood"]:
            key = f"ood_{sn}_{sl}"
            if key in results:
                m = results[key]
                log.info(f"    {sl:20s}: AUROC={m['auroc']:.4f}  AUPR-OUT={m['aupr_out']:.4f}  FPR@95={m['fpr_at_tpr']:.4f}")



if __name__ == "__main__":
    main()
