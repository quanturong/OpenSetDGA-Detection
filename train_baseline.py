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
  python train_baseline.py --run_dir dataset_out/run_20260222_193219
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from features import FEATURE_NAMES, extract_features_batch

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
            print(f"  [cache hit] {cache_file}")
            return np.load(str(cache_file))

    domains = df["domain"].tolist()
    BATCH = 50_000
    parts = []
    for i in range(0, len(domains), BATCH):
        batch = domains[i:i + BATCH]
        parts.append(extract_features_batch(batch))
        done = min(i + BATCH, len(domains))
        print(f"    featurised {done:>8,} / {len(domains):,}", end="\r")
    print()
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


def _energy_score(proba: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy-based OOD score.  Higher = more likely OOD.
    For a probability model we use -T*log(sum(exp(log(p)/T)))
    which reduces to -T*logsumexp(log(p)/T)."""
    log_p = np.log(np.clip(proba, 1e-30, None))
    lse = np.log(np.sum(np.exp(log_p / T), axis=1))
    return -T * lse  # more negative for confident → we negate so higher = OOD


# ── evaluation helpers ──────────────────────────────────────────────────────

def _evaluate_binary(y_true, y_pred, y_proba, label: str):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_proba)
    print(f"\n{'=' * 60}")
    print(f"  Binary classification – {label}")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["benign", "dga"]))
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

    # AUPR-OUT
    scores_all = pd.concat([id_s, ood_s], ignore_index=True)
    labels_out = pd.Series([0] * n_id + [1] * n_ood, dtype=int)
    aupr_out = _aupr(scores_all, labels_out)

    # AUPR-IN
    labels_in = pd.Series([1] * n_id + [0] * n_ood, dtype=int)
    aupr_in = _aupr(-scores_all, labels_in)

    return {
        "auroc": float(auroc),
        "aupr_out": float(aupr_out),
        "aupr_in": float(aupr_in),
        "fpr_at_tpr": float(fpr),
        "tpr_target": tpr_target,
        "realized_tpr": float(realized_tpr),
    }


def _aupr(scores: pd.Series, labels: pd.Series) -> float:
    df = pd.DataFrame({"score": scores.values, "label": labels.values}).sort_values("score", ascending=False)
    tp = fp = 0
    p = int((df["label"] == 1).sum())
    precisions = [1.0]
    recalls = [0.0]
    for _, row in df.iterrows():
        if row["label"] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / p if p > 0 else 0.0)
    area = sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in range(1, len(recalls)))
    return float(max(0.0, min(1.0, area)))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="dataset_out/run_20260222_193219",
                    help="Path to a build_dataset.py run directory")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: baseline_out/<timestamp>)")
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--energy_T", type=float, default=1.0, help="Temperature for energy score")
    ap.add_argument("--no_cache", action="store_true", help="Disable feature caching")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else Path(args.run_dir) / "_feature_cache"

    csvs = _csv_paths(args.run_dir)
    for k, p in csvs.items():
        if not p.exists():
            sys.exit(f"Missing CSV: {p}")

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("Loading CSVs …")
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}
    for k, df in dfs.items():
        print(f"  {k:20s}: {len(df):>8,} rows")

    # ── 2. Feature extraction ───────────────────────────────────────────────
    print("\nExtracting features …")
    Xs, ys = {}, {}
    for k, df in dfs.items():
        print(f"  [{k}]")
        Xs[k] = _featurise(df, cache_dir, k)
        # binary label: benign=0, dga/ood=1
        ys[k] = (df["label"] != "benign").astype(int).values

    # ── 3. Train LightGBM ──────────────────────────────────────────────────
    print("\nTraining LightGBM …")
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
        "random_state": 42,
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
    print(f"  Best iteration: {best_iter}")

    # save model
    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    # feature importance
    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(str(out_dir / "feature_importance.csv"), index=False)
    print(f"\n  Top-10 features:")
    print(imp.head(10).to_string(index=False))

    # ── 4. Evaluate binary classification ───────────────────────────────────
    results = {}
    for split in ["val", "test_known"]:
        proba = model.predict_proba(Xs[split])[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[f"binary_{split}"] = _evaluate_binary(ys[split], pred, proba, split)

    # ── 5. OOD scoring ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OOD Detection Evaluation")
    print("=" * 60)

    # ID scores come from test_known
    proba_known = model.predict_proba(Xs["test_known"])
    msp_known = _msp_score(proba_known)
    energy_known = _energy_score(proba_known, T=args.energy_T)

    ood_splits = {
        "unknown_family": "unknown_family",
        "unknown_ood": "unknown_ood",
    }

    for score_name, score_fn in [("msp", _msp_score), ("energy", lambda p: _energy_score(p, T=args.energy_T))]:
        print(f"\n── Score: {score_name} ──")
        id_scores = score_fn(proba_known)

        for split_label, split_key in ood_splits.items():
            proba_ood = model.predict_proba(Xs[split_key])
            ood_scores = score_fn(proba_ood)

            metrics = _ood_metrics(id_scores, ood_scores)
            results[f"ood_{score_name}_{split_label}"] = metrics

            print(f"\n  {split_label}:")
            print(f"    AUROC      : {metrics['auroc']:.6f}")
            print(f"    AUPR-OUT   : {metrics['aupr_out']:.6f}")
            print(f"    AUPR-IN    : {metrics['aupr_in']:.6f}")
            print(f"    FPR@TPR=0.95: {metrics['fpr_at_tpr']:.6f} (realized TPR={metrics['realized_tpr']:.6f})")

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
    print(f"\nAll results saved → {out_dir}")

    # ── 7. Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Binary classification (test_known):")
    bk = results["binary_test_known"]
    print(f"    Accuracy={bk['accuracy']:.4f}  F1={bk['f1']:.4f}  AUC={bk['roc_auc']:.4f}")

    for sn in ["msp", "energy"]:
        print(f"\n  OOD detection ({sn}):")
        for sl in ["unknown_family", "unknown_ood"]:
            key = f"ood_{sn}_{sl}"
            if key in results:
                m = results[key]
                print(f"    {sl:20s}: AUROC={m['auroc']:.4f}  AUPR-OUT={m['aupr_out']:.4f}  FPR@95={m['fpr_at_tpr']:.4f}")

    print()


if __name__ == "__main__":
    main()
