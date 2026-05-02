"""
train_bilstm.py – Character-level BiLSTM + Mahalanobis OOD for Open-Set DGA Detection.

Architecture:
  Embedding(vocab=40, dim=32) → BiLSTM(hidden=128, layers=2, dropout=0.3)
  → concat last hidden (fwd + bwd) → Dense(128, relu) [penultimate]
  → Dense(n_classes) [20-class head]

OOD scorers:
  - MSP (1 − max softmax prob)
  - Mahalanobis distance on penultimate features
  - Energy score on raw logits: −T·log Σ exp(logit_k / T)

Usage:
  python train_bilstm.py --run_dir dataset_out/run_20260222_193219
"""

import argparse
import json
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

# ── tokenization (shared with train_neural.py) ──────────────────────────────

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS) + 2  # 0=PAD, 1=UNK
MAX_LEN = 75


def tokenize_batch(domains: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    out = np.zeros((len(domains), max_len), dtype=np.int32)
    for i, d in enumerate(domains):
        d = d.lower().strip()
        for j, ch in enumerate(d[:max_len]):
            out[i, j] = _CHAR2IDX.get(ch, 1)
    return out


# ── model ─────────────────────────────────────────────────────────────────────

class DomainBiLSTM(nn.Module):
    """Character-level bidirectional LSTM domain classifier."""

    def __init__(self, vocab_size: int, embed_dim: int, n_classes: int,
                 hidden_dim: int = 128, num_layers: int = 2,
                 feat_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        # BiLSTM output: 2 * hidden_dim (fwd + bwd)
        self.pre = nn.Sequential(
            nn.Linear(hidden_dim * 2, feat_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Run LSTM with pack_padded_sequence for speed; return fwd_last ‖ bwd_first."""
        e = self.embed(x)  # (B, L, E)

        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )  # (B, L', 2*H)

            B = out.size(0)
            H = out.size(2) // 2
            idx = (lengths - 1).clamp(min=0).long()
            fwd = out[torch.arange(B, device=out.device), idx, :H]
            bwd = out[:, 0, H:]
            pooled = torch.cat([fwd, bwd], dim=-1)
        else:
            out, _ = self.lstm(e)
            pooled = out[:, -1, :]

        return self.drop(pooled)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        pooled = self._pool(x, lengths)
        feat = self.pre(pooled)        # (B, feat_dim)
        logit = self.head(feat)        # (B, n_classes)
        return logit, feat

    def features(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        _, feat = self.forward(x, lengths)
        return feat


# ── Mahalanobis OOD scorer ────────────────────────────────────────────────────

class MahalanobisScorer:
    def __init__(self):
        self.means = None
        self.sigma_inv = None

    def fit(self, features: np.ndarray, labels: np.ndarray,
            min_samples: int = 10, reg: float = 1e-4):
        """Fit class-conditional Gaussians.  Skip classes with < min_samples."""
        N, D = features.shape
        classes = np.unique(labels)

        # Filter degenerate classes
        valid = [k for k in classes if (labels == k).sum() >= min_samples]
        if len(valid) < len(classes):
            skipped = len(classes) - len(valid)
            print(f"  Mahalanobis: skipping {skipped} classes with < {min_samples} samples")

        means = np.zeros((len(classes), D))
        cov = np.zeros((D, D))
        count = 0
        for k in valid:
            mask = labels == k
            feat_k = features[mask]
            means[k] = feat_k.mean(axis=0)
            diff = feat_k - means[k]
            cov += diff.T @ diff
            count += len(feat_k)
        # Classes not in valid keep zero mean (will produce large distances → fine)
        cov /= max(count, 1)

        sigma_inv = np.linalg.inv(cov + reg * np.eye(D))
        self.means = means
        self.sigma_inv = sigma_inv

    def score(self, features: np.ndarray, batch_size: int = 2000) -> np.ndarray:
        n_classes = len(self.means)
        N = len(features)
        min_dists = np.full(N, np.inf)
        for k in range(n_classes):
            for i in range(0, N, batch_size):
                diff = features[i:i + batch_size] - self.means[k]
                tmp = diff @ self.sigma_inv
                dists = np.sum(tmp * diff, axis=1)
                min_dists[i:i + batch_size] = np.minimum(
                    min_dists[i:i + batch_size], dists)
        return min_dists.astype(np.float32)


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_lengths(tokens: np.ndarray) -> np.ndarray:
    """Number of non-padding tokens per row."""
    return (tokens != 0).sum(axis=1).astype(np.int64)


def _extract_features(model: DomainBiLSTM, loader: DataLoader,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_feat, all_logit = [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            # Second tensor is always lengths (from _make_loader)
            lb = batch[1].to(device) if len(batch) >= 2 else None
            logit, feat = model(xb, lb)
            all_feat.append(feat.cpu().numpy())
            all_logit.append(logit.cpu().numpy())
    return np.vstack(all_feat), np.vstack(all_logit)


def _make_loader(tokens: np.ndarray, batch_size: int,
                 shuffle: bool = False) -> DataLoader:
    xt = torch.tensor(tokens, dtype=torch.long)
    lengths = torch.tensor(_compute_lengths(tokens), dtype=torch.long)
    return DataLoader(TensorDataset(xt, lengths), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def _make_loader_labeled(tokens: np.ndarray, labels: np.ndarray,
                         batch_size: int, shuffle: bool = True) -> DataLoader:
    xt = torch.tensor(tokens, dtype=torch.long)
    yt = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(_compute_lengths(tokens), dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt, lengths), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def _csv_paths(run_dir: str) -> dict[str, Path]:
    rd = Path(run_dir)
    return {
        "train": rd / "known" / "train.csv",
        "val": rd / "known" / "val.csv",
        "test_known": rd / "known" / "test_known.csv",
        "unknown_family": rd / "unknown_family" / "test_unknown_family.csv",
        "unknown_ood": rd / "unknown_ood" / "test_unknown_ood.csv",
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="dataset_out/run_20260222_193219")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--feat_dim", type=int, default=128,
                    help="Penultimate layer dimension")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--energy_T", type=float, default=1.0,
                    help="Temperature for energy score")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"bilstm_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    csvs = _csv_paths(args.run_dir)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading CSVs …")
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}
    for k, df in dfs.items():
        print(f"  {k:20s}: {len(df):>8,} rows")

    # ── 2. Label encoding ────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(dfs["train"]["class_label"].values)
    n_classes = len(le.classes_)
    benign_idx = int(np.where(le.classes_ == "benign")[0][0])
    print(f"\n  {n_classes} classes: {list(le.classes_)}")

    y_train = le.transform(dfs["train"]["class_label"].values)
    y_val = le.transform(dfs["val"]["class_label"].values)
    y_test = le.transform(dfs["test_known"]["class_label"].values)

    # ── 3. Tokenize ───────────────────────────────────────────────────────
    print("\nTokenizing domains …")
    tok = {}
    for k, df in dfs.items():
        tok[k] = tokenize_batch(df["domain"].tolist())
        print(f"  {k:20s}: {tok[k].shape}")

    # ── 4. Build data loaders ─────────────────────────────────────────────
    train_loader = _make_loader_labeled(tok["train"], y_train, args.batch, shuffle=True)
    val_loader = _make_loader_labeled(tok["val"], y_val, args.batch, shuffle=False)

    # ── 5. Build model ────────────────────────────────────────────────────
    model = DomainBiLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDomainBiLSTM: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # ── 6. Training loop ──────────────────────────────────────────────────
    print("\nTraining …")
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n_train = 0
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad()
            logit, _ = model(xb, lb)
            loss = criterion(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logit.argmax(-1) == yb).sum().item()
            n_train += len(yb)

        train_loss = total_loss / n_train
        train_acc = correct / n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        n_val = 0
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                logit, _ = model(xb, lb)
                val_loss += criterion(logit, yb).item() * len(yb)
                val_correct += (logit.argmax(-1) == yb).sum().item()
                n_val += len(yb)
        val_loss /= n_val
        val_acc = val_correct / n_val

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  E{epoch:02d}: train_loss={train_loss:.4f} acc={train_acc:.4f}"
              f"  val_loss={val_loss:.4f} acc={val_acc:.4f}  lr={lr_now:.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, str(out_dir / "model.pt"))
    print(f"  Best model saved → {out_dir / 'model.pt'}")

    # ── 7. Extract penultimate features for all splits ────────────────────
    print("\nExtracting penultimate features …")
    feat_loaders = {
        k: _make_loader(tok[k], batch_size=1024)
        for k in ["train", "test_known", "unknown_family", "unknown_ood"]
    }
    feats, logits = {}, {}
    for k, loader in feat_loaders.items():
        feats[k], logits[k] = _extract_features(model, loader, device)
        print(f"  {k:20s}: feats={feats[k].shape}")

    # ── 8. Fit Mahalanobis scorer ─────────────────────────────────────────
    print("\nFitting Mahalanobis scorer …")
    scorer = MahalanobisScorer()
    scorer.fit(feats["train"], y_train, min_samples=10, reg=1e-4)
    print("  Done.")

    # ── 9. Classification eval ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Classification Results (test_known)")
    print("=" * 65)
    results = {}

    proba_test = F.softmax(torch.tensor(logits["test_known"]), dim=-1).numpy()
    y_pred_mc = proba_test.argmax(axis=1)
    mc_acc = accuracy_score(y_test, y_pred_mc)

    y_bin = (y_test != benign_idx).astype(int)
    p_dga = 1.0 - proba_test[:, benign_idx]
    bin_acc = accuracy_score(y_bin, (p_dga >= 0.5).astype(int))
    bin_f1 = f1_score(y_bin, (p_dga >= 0.5).astype(int))
    bin_auc = roc_auc_score(y_bin, p_dga)
    print(f"  MC-acc={mc_acc:.4f}  Bin-acc={bin_acc:.4f}  F1={bin_f1:.4f}  AUC={bin_auc:.4f}")
    results["cls_test_known"] = {
        "mc_accuracy": float(mc_acc), "bin_accuracy": float(bin_acc),
        "bin_f1": float(bin_f1), "bin_roc_auc": float(bin_auc),
    }

    # ── 10. OOD evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  OOD Detection Evaluation")
    print("=" * 65)

    T = args.energy_T

    def _msp_score(split):
        p = F.softmax(torch.tensor(logits[split]), dim=-1).numpy()
        return 1.0 - p.max(axis=1)

    def _energy_score(split):
        lg = torch.tensor(logits[split])
        return -(T * torch.logsumexp(lg / T, dim=-1)).numpy()

    scorer_fns = {
        "msp": _msp_score,
        "energy": _energy_score,
        "mahalanobis": lambda split: scorer.score(feats[split]),
    }
    id_scores = {
        "msp": _msp_score("test_known"),
        "energy": _energy_score("test_known"),
        "mahalanobis": scorer.score(feats["test_known"]),
    }
    ood_splits = ["unknown_family", "unknown_ood"]

    for sname, sfn in scorer_fns.items():
        print(f"\n── {sname} ──")
        for split in ood_splits:
            ood_s = sfn(split)
            m = ood_metrics(id_scores[sname], ood_s)
            results[f"ood_{sname}_{split}"] = m
            print_ood_metrics(m, split)

            pd.DataFrame({
                "domain": dfs[split]["domain"].values,
                "ood_score": ood_s,
            }).to_csv(str(out_dir / f"scores_{sname}_{split}.csv"), index=False)

        pd.DataFrame({
            "domain": dfs["test_known"]["domain"].values,
            "ood_score": id_scores[sname],
        }).to_csv(str(out_dir / f"scores_{sname}_known.csv"), index=False)

    # ── 11. Save + Summary ────────────────────────────────────────────────
    results["params"] = vars(args)
    results["n_classes"] = n_classes
    results["classes"] = list(le.classes_)
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

    print(f"\nAll results → {out_dir}")


if __name__ == "__main__":
    main()
