"""
hybrid_scorer.py — Energy + KNN Hybrid OOD Scorer

Trains a logistic regression on [s_energy, s_knn] using a calibration split
derived from existing score CSVs, then evaluates on both test splits.

Methodology:
  - ID source  : BiLSTM energy scores on test_known  (40k samples)
  - OOD source : 50% of unknown_ood scores           (50k samples, stratified)
  - LogReg trained on 80% of above, validated on remaining 20%
  - Final evaluation on:
      * test_known  (full, as ID reference)
      * unknown_family  (completely held-out — no contamination)
      * unknown_ood holdout (the 50% not used for training)

Score sources:
  - Energy : baseline_out/bilstm_*/scores_energy_*.csv
  - KNN    : baseline_out/multiclass_*/scores_knn_*.csv

Usage:
  python hybrid_scorer.py
  python hybrid_scorer.py --bilstm_dir baseline_out/bilstm_20260414_222503
                          --multiclass_dir baseline_out/multiclass_20260414_210513
                          --out_dir baseline_out/hybrid_<timestamp>
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ood_utils import ood_metrics, print_ood_metrics


# ── helpers ──────────────────────────────────────────────────────────────────

def load_scores(path: Path, tag: str) -> pd.Series:
    df = pd.read_csv(path)
    if "ood_score" not in df.columns:
        raise ValueError(f"No 'ood_score' column in {path}")
    return df.set_index("domain")["ood_score"].rename(tag)


def merge_scores(energy_path: Path, knn_path: Path) -> pd.DataFrame:
    e = load_scores(energy_path, "energy")
    k = load_scores(knn_path,   "knn")
    df = pd.concat([e, k], axis=1).dropna()
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilstm_dir",     default=None)
    ap.add_argument("--multiclass_dir", default=None)
    ap.add_argument("--out_dir",        default=None)
    ap.add_argument("--ood_cal_frac",   type=float, default=0.5,
                    help="Fraction of unknown_ood used for LogReg calibration")
    ap.add_argument("--logreg_C",       type=float, default=1.0)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    # ── locate run dirs ──────────────────────────────────────────────────────
    baseline = Path("baseline_out")

    def _find_latest(prefix: str) -> Path:
        runs = sorted(baseline.glob(f"{prefix}_*"))
        if not runs:
            raise FileNotFoundError(f"No run matching '{prefix}_*' in {baseline}")
        return runs[-1]

    bilstm_dir     = Path(args.bilstm_dir)     if args.bilstm_dir     else _find_latest("bilstm")
    multiclass_dir = Path(args.multiclass_dir) if args.multiclass_dir else _find_latest("multiclass")

    # Prefer BiLSTM KNN scores (neural features) over LightGBM KNN if available
    extra_ood_dir = _find_latest("extra_ood") if not args.bilstm_dir else None
    bilstm_knn_dir = None
    if extra_ood_dir and (extra_ood_dir / "scores_bilstm_knn_k5_known.csv").exists():
        bilstm_knn_dir = extra_ood_dir
        print(f"Using BiLSTM KNN scores from : {bilstm_knn_dir}")
    else:
        print(f"BiLSTM KNN scores not found, using LightGBM KNN from: {multiclass_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else baseline / f"hybrid_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"BiLSTM dir      : {bilstm_dir}")
    print(f"Multiclass dir  : {multiclass_dir}")
    print(f"Output dir      : {out_dir}")
    print()

    # ── load score pairs ─────────────────────────────────────────────────────
    print("Loading score CSVs...")

    if bilstm_knn_dir:
        knn_known_path = bilstm_knn_dir / "scores_bilstm_knn_k5_known.csv"
        knn_uf_path    = bilstm_knn_dir / "scores_bilstm_knn_k5_unknown_family.csv"
        knn_ood_path   = bilstm_knn_dir / "scores_bilstm_knn_k5_unknown_ood.csv"
        knn_label = "bilstm_knn_k5"
    else:
        knn_known_path = multiclass_dir / "scores_knn_known.csv"
        knn_uf_path    = multiclass_dir / "scores_knn_unknown_family.csv"
        knn_ood_path   = multiclass_dir / "scores_knn_unknown_ood.csv"
        knn_label = "lgbm_knn"

    def _merge(energy_path, knn_path):
        e = load_scores(energy_path, "energy")
        k = load_scores(knn_path,   "knn")
        return pd.concat([e, k], axis=1).dropna()

    known_df = _merge(bilstm_dir / "scores_energy_known.csv", knn_known_path)
    uf_df    = _merge(bilstm_dir / "scores_energy_unknown_family.csv", knn_uf_path)
    ood_df   = _merge(bilstm_dir / "scores_energy_unknown_ood.csv",   knn_ood_path)
    print(f"  known:          {len(known_df):,}")
    print(f"  unknown_family: {len(uf_df):,}")
    print(f"  unknown_ood:    {len(ood_df):,}")
    print()

    # ── build calibration set ────────────────────────────────────────────────
    # Label: 0 = ID (known), 1 = OOD
    # Mixed calibration: use BOTH unknown_family and unknown_ood as OOD examples
    # so LogReg learns to weight Energy (good for family) and KNN (good for ood).
    rng = np.random.default_rng(args.seed)

    # Split unknown_family: half calibration, half test
    uf_n_cal = len(uf_df) // 2
    uf_idx   = rng.choice(len(uf_df), size=uf_n_cal, replace=False)
    uf_mask  = np.zeros(len(uf_df), dtype=bool)
    uf_mask[uf_idx] = True
    uf_cal     = uf_df.iloc[uf_idx]
    uf_holdout = uf_df.iloc[~uf_mask]

    # Split unknown_ood: half calibration, half test
    ood_cal_n = len(ood_df) // 2
    ood_idx   = rng.choice(len(ood_df), size=ood_cal_n, replace=False)
    ood_mask  = np.zeros(len(ood_df), dtype=bool)
    ood_mask[ood_idx] = True
    ood_cal     = ood_df.iloc[ood_idx]
    ood_holdout = ood_df.iloc[~ood_mask]

    X_id      = known_df[["energy", "knn"]].values
    X_ood_mix = np.vstack([uf_cal[["energy", "knn"]].values,
                           ood_cal[["energy", "knn"]].values])
    y_id      = np.zeros(len(X_id),      dtype=int)
    y_ood_mix = np.ones( len(X_ood_mix), dtype=int)

    X_cal = np.vstack([X_id, X_ood_mix])
    y_cal = np.concatenate([y_id, y_ood_mix])

    X_train, X_val, y_train, y_val = train_test_split(
        X_cal, y_cal, test_size=0.2, random_state=args.seed, stratify=y_cal
    )

    print(f"Calibration split — train: {len(X_train):,}  val: {len(X_val):,}")
    print(f"  ID train: {(y_train==0).sum():,}  OOD train: {(y_train==1).sum():,}")
    print(f"  OOD mix : {uf_n_cal:,} family + {ood_cal_n:,} ood")
    print()

    # ── train logistic regression ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegression(C=args.logreg_C, max_iter=1000, random_state=args.seed)
    clf.fit(X_train_s, y_train)

    val_auroc = roc_auc_score(y_val, clf.predict_proba(X_val_s)[:, 1])
    print(f"LogReg coef      : energy={clf.coef_[0][0]:.4f}  knn={clf.coef_[0][1]:.4f}")
    print(f"LogReg intercept : {clf.intercept_[0]:.4f}")
    print(f"LogReg val AUROC : {val_auroc:.6f}")
    print()

    # ── score test splits ─────────────────────────────────────────────────────
    def hybrid_score(df: pd.DataFrame) -> np.ndarray:
        X = scaler.transform(df[["energy", "knn"]].values)
        return clf.predict_proba(X)[:, 1]  # P(OOD)

    known_scores    = hybrid_score(known_df)
    uf_scores       = hybrid_score(uf_holdout)
    ood_hld_scores  = hybrid_score(ood_holdout)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Evaluation results")
    print("=" * 60)

    results = {}

    m_uf = ood_metrics(known_scores, uf_scores)
    results["hybrid_unknown_family"] = m_uf
    print("vs unknown_family  (holdout 50%, never seen in calibration):")
    print_ood_metrics(m_uf)
    print()

    m_ood = ood_metrics(known_scores, ood_hld_scores)
    results["hybrid_unknown_ood_holdout"] = m_ood
    print("vs unknown_ood     (holdout 50%):")
    print_ood_metrics(m_ood)
    print()

    # Compare: individual scores on same splits
    print("-" * 60)
    print("Comparison — individual scorers on unknown_family (holdout):")
    for name, col in [("BiLSTM Energy", "energy"), (f"KNN ({knn_label})", "knn")]:
        m = ood_metrics(known_df[col].values, uf_holdout[col].values)
        print(f"  {name:25s}: AUROC={m['auroc']:.6f}  FPR@95={m['fpr_at_tpr']:.4f}")
    print(f"  {'Hybrid (LogReg)':25s}: AUROC={m_uf['auroc']:.6f}  FPR@95={m_uf['fpr_at_tpr']:.4f}")
    print()

    print("-" * 60)
    print("Comparison — individual scorers on unknown_ood (holdout):")
    for name, col in [("BiLSTM Energy", "energy"), (f"KNN ({knn_label})", "knn")]:
        m = ood_metrics(known_df[col].values, ood_holdout[col].values)
        print(f"  {name:25s}: AUROC={m['auroc']:.6f}  FPR@95={m['fpr_at_tpr']:.4f}")
    print(f"  {'Hybrid (LogReg)':25s}: AUROC={m_ood['auroc']:.6f}  FPR@95={m_ood['fpr_at_tpr']:.4f}")
    print()

    # ── save score CSVs ───────────────────────────────────────────────────────
    pd.DataFrame({"domain": known_df.index, "ood_score": known_scores}).to_csv(
        out_dir / "scores_hybrid_known.csv", index=False)
    pd.DataFrame({"domain": uf_holdout.index, "ood_score": uf_scores}).to_csv(
        out_dir / "scores_hybrid_unknown_family.csv", index=False)
    pd.DataFrame({"domain": ood_holdout.index, "ood_score": ood_hld_scores}).to_csv(
        out_dir / "scores_hybrid_unknown_ood.csv", index=False)

    # ── save results ──────────────────────────────────────────────────────────
    results["logreg"] = {
        "coef_energy":    float(clf.coef_[0][0]),
        "coef_knn":       float(clf.coef_[0][1]),
        "intercept":      float(clf.intercept_[0]),
        "val_auroc":      val_auroc,
        "C":              args.logreg_C,
        "ood_cal_frac":   args.ood_cal_frac,
        "seed":           args.seed,
    }
    results["sources"] = {
        "bilstm_dir":     str(bilstm_dir),
        "multiclass_dir": str(multiclass_dir),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Scores and results saved to: {out_dir}")


if __name__ == "__main__":
    main()
