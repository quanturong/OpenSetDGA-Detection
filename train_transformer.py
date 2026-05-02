"""
train_transformer.py – Character-level Transformer Encoder OOD for Open-Set DGA Detection.

Architecture:
  [CLS] + Embedding(vocab=40, dim=64) + SinCos PositionalEncoding
  → TransformerEncoder(nhead=4, d_model=64, dim_ff=256, layers=2, dropout=0.1)
  → CLS token → Dense(128, relu) [penultimate, used for KNN / Mahalanobis]
  → Dense(n_classes)

OOD scorers:
  - MSP  (1 − max softmax prob)
  - Energy score: −T·log Σ exp(logit_k / T)
  - Mahalanobis distance on penultimate features

Output files (compatible with eval_extra_ood.py):
  model.pt, results.json
  scores_{msp,energy,mahalanobis}_{known,unknown_family,unknown_ood}.csv

Usage:
  python train_transformer.py
  python train_transformer.py --epochs 20 --device cuda
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from ood_utils import ood_metrics, print_ood_metrics

# ── tokenization (shared with BiLSTM / CNN) ───────────────────────────────────

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS) + 2   # 0=PAD, 1=UNK
MAX_LEN = 75
CLS_IDX = VOCAB_SIZE           # extra token id for [CLS]; vocab extended by 1


def tokenize_batch(domains: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    """(N, 1 + max_len): column 0 = CLS_IDX, rest = char tokens."""
    out = np.zeros((len(domains), 1 + max_len), dtype=np.int32)
    out[:, 0] = CLS_IDX
    for i, d in enumerate(domains):
        d = d.lower().strip()
        for j, ch in enumerate(d[:max_len]):
            out[i, j + 1] = _CHAR2IDX.get(ch, 1)
    return out


# ── model ─────────────────────────────────────────────────────────────────────

class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


class DomainTransformer(nn.Module):
    """Character-level Transformer encoder for domain classification."""

    def __init__(
        self,
        vocab_size: int,          # includes CLS token → VOCAB_SIZE + 1
        embed_dim: int,
        n_classes: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        feat_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = MAX_LEN + 1,  # +1 for CLS
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos   = SinCosPositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        self.pre = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) long — first position is CLS token.
        Returns CLS representation (B, embed_dim).
        """
        # padding mask: True = ignore position
        pad_mask = (x == 0)  # (B, L) — PAD positions
        e = self.pos(self.embed(x))                   # (B, L, E)
        h = self.encoder(e, src_key_padding_mask=pad_mask)  # (B, L, E)
        return h[:, 0, :]                              # CLS token

    def forward(self, x: torch.Tensor):
        cls = self._encode(x)           # (B, E)
        feat = self.pre(cls)            # (B, feat_dim)
        logit = self.head(feat)         # (B, n_classes)
        return logit, feat

    def features(self, x: torch.Tensor) -> torch.Tensor:
        _, feat = self.forward(x)
        return feat


# ── Mahalanobis OOD scorer ────────────────────────────────────────────────────

class MahalanobisScorer:
    def __init__(self):
        self.means = None
        self.sigma_inv = None

    def fit(self, feats: np.ndarray, labels: np.ndarray,
            min_samples: int = 10, reg: float = 1e-4):
        n_classes = int(labels.max()) + 1
        # Store mean for every class index so self.means[label] works
        self.means = np.zeros((n_classes, feats.shape[1]))
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() >= min_samples:
                self.means[c] = feats[mask].mean(axis=0)

        centered = feats - self.means[labels]
        sigma = (centered.T @ centered) / len(feats)
        sigma += reg * np.eye(sigma.shape[0])
        self.sigma_inv = np.linalg.inv(sigma)

    def score(self, feats: np.ndarray) -> np.ndarray:
        diff = feats[:, None, :] - self.means[None, :, :]  # (N, C, D)
        left = diff @ self.sigma_inv                         # (N, C, D)
        dists = (left * diff).sum(-1)                        # (N, C)
        return dists.min(axis=1)                             # (N,) min over classes


# ── data loaders ──────────────────────────────────────────────────────────────

def _make_loader_labeled(tokens: np.ndarray, labels: np.ndarray,
                         batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(tokens).long()
    yt = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def _make_loader(tokens: np.ndarray, batch_size: int = 1024) -> DataLoader:
    xt = torch.from_numpy(tokens).long()
    return DataLoader(TensorDataset(xt), batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=False)


@torch.no_grad()
def _extract_features(model: DomainTransformer, loader: DataLoader,
                      device: torch.device):
    model.eval()
    all_feats, all_logits = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        logit, feat = model(xb)
        all_feats.append(feat.cpu().numpy())
        all_logits.append(logit.cpu().numpy())
    return np.vstack(all_feats), np.vstack(all_logits)


# ── CSV paths ─────────────────────────────────────────────────────────────────

def _csv_paths(run_dir: str) -> dict[str, Path]:
    rd = Path(run_dir)
    return {
        "train":          rd / "known"          / "train.csv",
        "val":            rd / "known"          / "val.csv",
        "test_known":     rd / "known"          / "test_known.csv",
        "unknown_family": rd / "unknown_family" / "test_unknown_family.csv",
        "unknown_ood":    rd / "unknown_ood"    / "test_unknown_ood.csv",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",        default="dataset_out/run_20260222_193219")
    ap.add_argument("--out_dir",        default=None)
    ap.add_argument("--epochs",         type=int,   default=20)
    ap.add_argument("--batch",          type=int,   default=512)
    ap.add_argument("--lr",             type=float, default=3e-4)
    ap.add_argument("--embed_dim",      type=int,   default=64)
    ap.add_argument("--nhead",          type=int,   default=4)
    ap.add_argument("--num_layers",     type=int,   default=2)
    ap.add_argument("--dim_feedforward",type=int,   default=256)
    ap.add_argument("--feat_dim",       type=int,   default=128)
    ap.add_argument("--dropout",        type=float, default=0.1)
    ap.add_argument("--patience",       type=int,   default=5)
    ap.add_argument("--energy_T",       type=float, default=1.0)
    ap.add_argument("--device",         default="cpu")
    ap.add_argument("--model_path",     default=None,
                    help="Skip training and load an existing model.pt")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"transformer_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    csvs   = _csv_paths(args.run_dir)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading CSVs ...")
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}
    for k, df in dfs.items():
        print(f"  {k:20s}: {len(df):>8,} rows")

    # ── 2. Label encoding ─────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(dfs["train"]["class_label"].values)
    n_classes  = len(le.classes_)
    benign_idx = int(np.where(le.classes_ == "benign")[0][0])
    print(f"\n  {n_classes} classes: {list(le.classes_)}")

    y_train = le.transform(dfs["train"]["class_label"].values)
    y_val   = le.transform(dfs["val"]["class_label"].values)
    y_test  = le.transform(dfs["test_known"]["class_label"].values)

    # ── 3. Tokenize ───────────────────────────────────────────────────────
    print("\nTokenizing domains ...")
    tok = {}
    for k, df in dfs.items():
        tok[k] = tokenize_batch(df["domain"].tolist())
        print(f"  {k:20s}: {tok[k].shape}")

    # ── 4. Build data loaders ─────────────────────────────────────────────
    train_loader = _make_loader_labeled(tok["train"], y_train, args.batch, shuffle=True)
    val_loader   = _make_loader_labeled(tok["val"],   y_val,   args.batch, shuffle=False)

    # ── 5. Build model ────────────────────────────────────────────────────
    model = DomainTransformer(
        vocab_size       = CLS_IDX + 1,   # 0..VOCAB_SIZE (CLS_IDX = VOCAB_SIZE)
        embed_dim        = args.embed_dim,
        n_classes        = n_classes,
        nhead            = args.nhead,
        num_layers       = args.num_layers,
        dim_feedforward  = args.dim_feedforward,
        feat_dim         = args.feat_dim,
        dropout          = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDomainTransformer: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # ── 6. Training loop ──────────────────────────────────────────────────
    if args.model_path:
        print(f"\nLoading model from {args.model_path} (skipping training) ...")
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        torch.save(state, str(out_dir / "model.pt"))
    else:
        print("\nTraining ...")
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss, correct, n_train = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logit, _ = model(xb)
                loss = criterion(logit, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (logit.argmax(-1) == yb).sum().item()
                n_train    += len(yb)

            train_loss = total_loss / n_train
            train_acc  = correct   / n_train

            model.eval()
            val_loss, val_correct, n_val = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logit, _ = model(xb)
                    val_loss    += criterion(logit, yb).item() * len(yb)
                    val_correct += (logit.argmax(-1) == yb).sum().item()
                    n_val       += len(yb)
            val_loss /= n_val
            val_acc   = val_correct / n_val

            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  E{epoch:02d}: train_loss={train_loss:.4f} acc={train_acc:.4f}"
                  f"  val_loss={val_loss:.4f} acc={val_acc:.4f}"
                  f"  lr={lr_now:.2e}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, str(out_dir / "model.pt"))
        print(f"  Best model saved -> {out_dir / 'model.pt'}")

    # ── 7. Extract features for all splits ────────────────────────────────
    print("\nExtracting penultimate features ...")
    feat_loaders = {k: _make_loader(tok[k], batch_size=1024)
                    for k in ["train", "test_known", "unknown_family", "unknown_ood"]}
    feats, logits_all = {}, {}
    for k, loader in feat_loaders.items():
        feats[k], logits_all[k] = _extract_features(model, loader, device)
        print(f"  {k:20s}: feats={feats[k].shape}")

    # ── 8. Fit Mahalanobis scorer ─────────────────────────────────────────
    print("\nFitting Mahalanobis scorer ...")
    scorer = MahalanobisScorer()
    scorer.fit(feats["train"], y_train, min_samples=10, reg=1e-4)
    print("  Done.")

    # ── 9. Classification eval ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Classification Results (test_known)")
    print("=" * 65)
    results = {}

    proba_test = F.softmax(torch.tensor(logits_all["test_known"]), dim=-1).numpy()
    y_pred_mc  = proba_test.argmax(axis=1)
    mc_acc     = accuracy_score(y_test, y_pred_mc)

    y_bin   = (y_test != benign_idx).astype(int)
    p_dga   = 1.0 - proba_test[:, benign_idx]
    bin_acc = accuracy_score(y_bin, (p_dga >= 0.5).astype(int))
    bin_f1  = f1_score(y_bin, (p_dga >= 0.5).astype(int))
    bin_auc = roc_auc_score(y_bin, p_dga)
    print(f"  MC-acc={mc_acc:.4f}  Bin-acc={bin_acc:.4f}  F1={bin_f1:.4f}  AUC={bin_auc:.4f}")
    results["cls_test_known"] = {
        "mc_accuracy": float(mc_acc),  "bin_accuracy": float(bin_acc),
        "bin_f1":      float(bin_f1),  "bin_roc_auc":  float(bin_auc),
    }

    # ── 10. OOD evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  OOD Detection Evaluation")
    print("=" * 65)

    T = args.energy_T

    def _msp_score(split):
        p = F.softmax(torch.tensor(logits_all[split]), dim=-1).numpy()
        return 1.0 - p.max(axis=1)

    def _energy_score(split):
        lg = torch.tensor(logits_all[split])
        return -(T * torch.logsumexp(lg / T, dim=-1)).numpy()

    scorer_fns = {
        "msp":         _msp_score,
        "energy":      _energy_score,
        "mahalanobis": lambda split: scorer.score(feats[split]),
    }
    id_scores = {
        "msp":         _msp_score("test_known"),
        "energy":      _energy_score("test_known"),
        "mahalanobis": scorer.score(feats["test_known"]),
    }
    ood_splits = ["unknown_family", "unknown_ood"]

    for sname, sfn in scorer_fns.items():
        print(f"\n-- {sname} --")
        for split in ood_splits:
            ood_s = sfn(split)
            m     = ood_metrics(id_scores[sname], ood_s)
            results[f"ood_{sname}_{split}"] = m
            print_ood_metrics(m, split)

            pd.DataFrame({
                "domain":    dfs[split]["domain"].values,
                "ood_score": ood_s,
            }).to_csv(str(out_dir / f"scores_{sname}_{split}.csv"), index=False)

        pd.DataFrame({
            "domain":    dfs["test_known"]["domain"].values,
            "ood_score": id_scores[sname],
        }).to_csv(str(out_dir / f"scores_{sname}_known.csv"), index=False)

    # ── 11. Save + Summary ────────────────────────────────────────────────
    results["params"]    = vars(args)
    results["n_classes"] = n_classes
    results["classes"]   = list(le.classes_)
    with open(str(out_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    ck = results["cls_test_known"]
    print(f"  Binary  (test_known): Acc={ck['bin_accuracy']:.4f}  F1={ck['bin_f1']:.4f}"
          f"  AUC={ck['bin_roc_auc']:.4f}")
    print(f"  20-class(test_known): Acc={ck['mc_accuracy']:.4f}")
    for sn in ["msp", "energy", "mahalanobis"]:
        print(f"\n  OOD ({sn}):")
        for sl in ood_splits:
            m = results[f"ood_{sn}_{sl}"]
            print(f"    {sl:20s}: AUROC={m['auroc']:.4f}  AUPR-OUT={m['aupr_out']:.4f}"
                  f"  FPR@95={m['fpr_at_tpr']:.4f}")

    print(f"\nAll results -> {out_dir}")


if __name__ == "__main__":
    main()
