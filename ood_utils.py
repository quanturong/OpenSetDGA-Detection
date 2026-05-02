"""
Shared OOD evaluation utilities.
Fast vectorised implementations using sklearn — much faster than the
row-iteration approach in evaluate_ood.py for large datasets.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def msp_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    MSP OOD score with temperature scaling.
    Dividing logits by T > 1 softens the distribution → reduces overconfidence.
    Returns 1 - max(softmax(logits / T)), shape (N,).
    """
    scaled = logits / T
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    return 1.0 - probs.max(axis=1)


def energy_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    Energy OOD score: -T * log sum_k exp(logit_k / T).
    T > 1 separates in-distribution from OOD better than raw logits.
    Returns shape (N,).
    """
    scaled = logits / T
    return -T * np.log(np.exp(scaled).sum(axis=1))


def find_best_temperature(
    id_logits: np.ndarray,
    ood_logits: np.ndarray,
    scorer: str = "energy",
    temps: tuple = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
) -> tuple[float, dict]:
    """
    Grid-search over temperature values; pick T that maximises AUROC.
    scorer: 'energy' | 'msp'
    Returns (best_T, {T: auroc}).
    """
    results = {}
    for T in temps:
        if scorer == "energy":
            id_s  = energy_with_temperature(id_logits,  T)
            ood_s = energy_with_temperature(ood_logits, T)
        else:
            id_s  = msp_with_temperature(id_logits,  T)
            ood_s = msp_with_temperature(ood_logits, T)
        scores = np.concatenate([id_s, ood_s])
        labels = np.array([0] * len(id_s) + [1] * len(ood_s))
        results[T] = float(roc_auc_score(labels, scores))
    best_T = max(results, key=results.__getitem__)
    return best_T, results


def ood_metrics(id_scores, ood_scores, tpr_target: float = 0.95) -> dict:
    """
    Compute AUROC, AUPR-OUT, AUPR-IN, FPR@TPR.

    id_scores  : 1-D array-like — OOD scores for in-distribution (known) samples
    ood_scores : 1-D array-like — OOD scores for OOD samples
    Higher score = more OOD.
    """
    id_s = np.asarray(id_scores, dtype=np.float64).ravel()
    ood_s = np.asarray(ood_scores, dtype=np.float64).ravel()

    scores = np.concatenate([id_s, ood_s])
    labels_ood = np.array([0] * len(id_s) + [1] * len(ood_s), dtype=int)

    auroc = float(roc_auc_score(labels_ood, scores))
    aupr_out = float(average_precision_score(labels_ood, scores))
    aupr_in = float(average_precision_score(1 - labels_ood, -scores))

    threshold = float(np.quantile(ood_s, 1.0 - tpr_target))
    realized_tpr = float((ood_s >= threshold).mean())
    fpr = float((id_s >= threshold).mean())

    tp = int((ood_s >= threshold).sum())
    fp = int((id_s >= threshold).sum())
    precision_at_tpr = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "auroc": auroc,
        "aupr_out": aupr_out,
        "aupr_in": aupr_in,
        "fpr_at_tpr": fpr,
        "tpr_target": tpr_target,
        "realized_tpr": realized_tpr,
        "precision_at_tpr": precision_at_tpr,
    }


def print_ood_metrics(metrics: dict, label: str = ""):
    prefix = f"  {label:20s}" if label else "  "
    print(f"{prefix}: AUROC={metrics['auroc']:.6f}  AUPR-OUT={metrics['aupr_out']:.6f}"
          f"  AUPR-IN={metrics['aupr_in']:.6f}  FPR@95={metrics['fpr_at_tpr']:.6f}"
          f"  Prec@TPR={metrics.get('precision_at_tpr', float('nan')):.4f}"
          f"  (TPR={metrics['realized_tpr']:.4f})")
