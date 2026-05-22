import argparse
import math
from typing import Tuple

import pandas as pd


def _validate_score_series(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if out.empty:
        raise ValueError(f"No valid numeric scores found for {name}")
    return out


def _auroc(id_scores: pd.Series, ood_scores: pd.Series) -> float:
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    all_scores = pd.concat([
        pd.DataFrame({"score": id_scores.values, "is_ood": 0}),
        pd.DataFrame({"score": ood_scores.values, "is_ood": 1}),
    ], ignore_index=True)
    all_scores["rank"] = all_scores["score"].rank(method="average")
    rank_sum_ood = all_scores.loc[all_scores["is_ood"] == 1, "rank"].sum()
    u = rank_sum_ood - (n_ood * (n_ood + 1) / 2)
    return float(u / (n_ood * n_id))


def _fpr_at_tpr(id_scores: pd.Series, ood_scores: pd.Series, tpr_target: float = 0.95) -> Tuple[float, float]:
    if not (0 < tpr_target <= 1):
        raise ValueError("tpr_target must be in (0, 1]")
    threshold = float(ood_scores.quantile(1.0 - tpr_target, interpolation="linear"))
    tpr = float((ood_scores >= threshold).mean())
    fpr = float((id_scores >= threshold).mean())
    return fpr, tpr


def _precision_recall_curve(scores: pd.Series, labels: pd.Series):
    df = pd.DataFrame({"score": scores.values, "label": labels.values}).sort_values("score", ascending=False)
    tp = 0
    fp = 0
    p = int((df["label"] == 1).sum())

    precisions = []
    recalls = []

    for _, row in df.iterrows():
        if row["label"] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / p if p > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    precisions = [1.0] + precisions
    recalls = [0.0] + recalls
    return precisions, recalls


def _aupr(scores: pd.Series, labels: pd.Series) -> float:
    precisions, recalls = _precision_recall_curve(scores, labels)
    area = 0.0
    for i in range(1, len(recalls)):
        dx = recalls[i] - recalls[i - 1]
        area += dx * precisions[i]
    return float(max(0.0, min(1.0, area)))


def main():
    ap = argparse.ArgumentParser(description="Evaluate OOD detection metrics from known/OOD score CSV files")
    ap.add_argument("--known_csv", required=True, help="CSV of known(ID) samples")
    ap.add_argument("--ood_csv", required=True, help="CSV of OOD samples")
    ap.add_argument("--score_col", default="ood_score", help="Column name with OOD score (higher means more OOD)")
    ap.add_argument("--tpr_target", type=float, default=0.95, help="TPR target for FPR@TPR")
    args = ap.parse_args()

    known_df = pd.read_csv(args.known_csv)
    ood_df = pd.read_csv(args.ood_csv)

    if args.score_col not in known_df.columns:
        raise ValueError(f"{args.score_col} not found in known CSV")
    if args.score_col not in ood_df.columns:
        raise ValueError(f"{args.score_col} not found in OOD CSV")

    id_scores = _validate_score_series(known_df[args.score_col], "known_csv")
    ood_scores = _validate_score_series(ood_df[args.score_col], "ood_csv")

    auroc = _auroc(id_scores, ood_scores)
    fpr, realized_tpr = _fpr_at_tpr(id_scores, ood_scores, tpr_target=args.tpr_target)

    scores_all = pd.concat([id_scores, ood_scores], ignore_index=True)
    labels_out = pd.Series([0] * len(id_scores) + [1] * len(ood_scores), dtype=int)
    aupr_out = _aupr(scores_all, labels_out)

    labels_in = pd.Series([1] * len(id_scores) + [0] * len(ood_scores), dtype=int)
    aupr_in = _aupr(-scores_all, labels_in)

    print("OOD evaluation results")
    print(f"known_count: {len(id_scores):,}")
    print(f"ood_count:   {len(ood_scores):,}")
    print(f"AUROC:       {auroc:.6f}")
    print(f"AUPR-OUT:    {aupr_out:.6f}")
    print(f"AUPR-IN:     {aupr_in:.6f}")
    print(f"FPR@TPR={args.tpr_target:.2f}: {fpr:.6f} (realized TPR={realized_tpr:.6f})")


if __name__ == "__main__":
    main()
