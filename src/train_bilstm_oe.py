"""
train_bilstm_oe.py – BiLSTM trained with Outlier Exposure (OE) for Open-Set DGA Detection.

Same BiLSTM architecture as train_bilstm.py, but trained with an auxiliary OE loss
that pushes unknown_ood samples toward uniform softmax output:

  loss = CE(ID_batch) + lambda_oe * (-mean(log_softmax(OE_batch)))

The unknown_ood set is split 50/50 into oe_train (used during training) and
oe_test (held-out, used for evaluation instead of the full unknown_ood set).

OOD scorers: MSP, Energy, KNN

Usage:
  python src/train_bilstm_oe.py --run_dir data/processed
"""

import argparse
import json
import random
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from ood_utils import ood_metrics, print_ood_metrics

# ── logging ───────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── tokenization ─────────────────────────────────────────────────────────────

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS) + 2   # 0=PAD, 1=UNK
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
                 hidden_dim: int = 64, num_layers: int = 1,
                 feat_dim: int = 128, dropout: float = 0.2):
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
        self.pre = nn.Sequential(
            nn.Linear(hidden_dim * 2, feat_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        e = self.embed(x)
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None,
                return_feats: bool = True):
        pooled = self._pool(x, lengths)
        feat = self.pre(pooled)
        logit = self.head(feat)
        if return_feats:
            return logit, feat
        return logit


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_lengths(tokens: np.ndarray) -> np.ndarray:
    return (tokens != 0).sum(axis=1).astype(np.int64)


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


def _extract_features(model: DomainBiLSTM, loader: DataLoader,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_feat, all_logit = [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            lb = batch[1].to(device) if len(batch) >= 2 else None
            logit, feat = model(xb, lb)
            all_feat.append(feat.cpu().numpy())
            all_logit.append(logit.cpu().numpy())
    return np.vstack(all_feat), np.vstack(all_logit)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="data/processed")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed_dim", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--feat_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lambda_oe", type=float, default=0.5,
                    help="Weight for Outlier Exposure loss term")
    ap.add_argument("--oe_train_size", type=int, default=50_000,
                    help="Max samples from unknown_ood used as OE training data")
    ap.add_argument("--energy_T", type=float, default=1.0)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"bilstm_oe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)

    # ── 1. Load data ──────────────────────────────────────────────────────
    log.info("Loading CSVs ...")
    csvs = {
        "train":          run_dir / "known" / "train.csv",
        "val":            run_dir / "known" / "val.csv",
        "test_known":     run_dir / "known" / "test_known.csv",
        "unknown_family": run_dir / "unknown_family" / "test_unknown_family.csv",
    }
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}

    # Split unknown_ood 50/50: oe_train (for training) vs oe_test (held-out eval)
    unk_ood_all = pd.read_csv(str(run_dir / "unknown_ood" / "test_unknown_ood.csv"))
    unk_ood_all = unk_ood_all.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    mid = min(args.oe_train_size, len(unk_ood_all) // 2)
    dfs["oe_train"] = unk_ood_all.iloc[:mid].reset_index(drop=True)
    dfs["oe_test"]  = unk_ood_all.iloc[mid:mid * 2].reset_index(drop=True)

    for k, df in dfs.items():
        tag = "  [NOT in test]" if k == "oe_train" else ("  [held-out]" if k == "oe_test" else "")
        log.info(f"  {k:20s}: {len(df):>8,} rows{tag}")

    # ── 2. Label encoding ────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(dfs["train"]["class_label"].values)
    n_classes = len(le.classes_)
    benign_idx = int(np.where(le.classes_ == "benign")[0][0])
    log.info(f"\n  {n_classes} classes")

    y_train = le.transform(dfs["train"]["class_label"].values)
    y_val   = le.transform(dfs["val"]["class_label"].values)
    y_test  = le.transform(dfs["test_known"]["class_label"].values)

    # ── 3. Tokenize ───────────────────────────────────────────────────────
    log.info("\nTokenizing ...")
    tok = {k: tokenize_batch(df["domain"].tolist()) for k, df in dfs.items()}

    # ── 4. Data loaders ───────────────────────────────────────────────────
    train_loader = _make_loader_labeled(tok["train"], y_train, args.batch, shuffle=True)
    val_loader   = _make_loader_labeled(tok["val"],   y_val,   args.batch, shuffle=False)
    oe_loader    = _make_loader(tok["oe_train"], args.batch, shuffle=True)
    oe_iter      = cycle(oe_loader)

    # ── 5. Model ──────────────────────────────────────────────────────────
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
    log.info(f"\nDomainBiLSTM+OE: {n_params:,} trainable params")
    log.info(f"  lambda_oe={args.lambda_oe}  oe_train_size={len(dfs['oe_train'])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # ── 6. Training with OE ───────────────────────────────────────────────
    log.info("\nTraining with Outlier Exposure ...")
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ce = total_oe = 0.0
        correct = n_seen = 0

        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)

            # OE batch (cycle to match ID batch count)
            oe_batch = next(oe_iter)
            oe_x  = oe_batch[0].to(device)
            oe_lb = oe_batch[1].to(device)

            optimizer.zero_grad()

            # ID loss
            logit, _ = model(xb, lb)
            loss_ce = criterion(logit, yb)

            # OE loss: push OE toward uniform
            oe_logit = model(oe_x, oe_lb, return_feats=False)
            loss_oe = -F.log_softmax(oe_logit, dim=1).mean()

            loss = loss_ce + args.lambda_oe * loss_oe
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_ce += loss_ce.item() * len(yb)
            total_oe += loss_oe.item() * len(yb)
            correct  += (logit.argmax(-1) == yb).sum().item()
            n_seen   += len(yb)

        # Validation
        model.eval()
        val_loss = val_correct = n_val = 0
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                logit, _ = model(xb, lb)
                val_loss    += criterion(logit, yb).item() * len(yb)
                val_correct += (logit.argmax(-1) == yb).sum().item()
                n_val       += len(yb)
        val_loss /= n_val
        val_acc   = val_correct / n_val

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"  E{epoch:02d}: ce={total_ce/n_seen:.4f} oe={total_oe/n_seen:.4f} "
                 f"train_acc={correct/n_seen:.4f}  "
                 f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  lr={lr_now:.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, str(out_dir / "model.pt"))
    log.info("  Best model saved.")

    # ── 7. Extract features ───────────────────────────────────────────────
    log.info("\nExtracting features ...")
    feat_splits = ["train", "test_known", "unknown_family", "oe_test"]
    feat_loaders = {k: _make_loader(tok[k], batch_size=1024) for k in feat_splits}
    feats, logits_d = {}, {}
    for k, loader in feat_loaders.items():
        feats[k], logits_d[k] = _extract_features(model, loader, device)
        log.info(f"  {k:20s}: {feats[k].shape}")

    # ── 8. KNN scorer ─────────────────────────────────────────────────────
    log.info(f"\nBuilding KNN index (k={args.knn_k}) ...")
    knn = NearestNeighbors(n_neighbors=args.knn_k, algorithm="auto", n_jobs=-1)
    knn.fit(feats["train"])
    log.info("  Done.")

    # ── 9. OOD evaluation ─────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  OOD Detection Results")
    log.info("=" * 65)

    T = args.energy_T

    def _msp_score(split):
        p = F.softmax(torch.tensor(logits_d[split]), dim=-1).numpy()
        return 1.0 - p.max(axis=1)

    def _energy_score(split):
        lg = torch.tensor(logits_d[split])
        return -(T * torch.logsumexp(lg / T, dim=-1)).numpy()

    def _knn_score(split):
        dists, _ = knn.kneighbors(feats[split])
        return dists.mean(axis=1)

    results = {}
    ood_splits_eval = ["unknown_family", "oe_test"]
    scorer_fns = {"energy": _energy_score, "msp": _msp_score, "knn": _knn_score}

    for sname, sfn in scorer_fns.items():
        log.info(f"\n-- {sname} --")
        id_score = sfn("test_known")
        for split in ood_splits_eval:
            ood_s = sfn(split)
            m = ood_metrics(id_score, ood_s)
            label = "unknown_ood (held-out)" if split == "oe_test" else split
            results[f"ood_{sname}_{split}"] = m
            print_ood_metrics(m, label)

    # ── 10. Classification eval ───────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  Classification (test_known)")
    log.info("=" * 65)
    proba = F.softmax(torch.tensor(logits_d["test_known"]), dim=-1).numpy()
    mc_acc  = accuracy_score(y_test, proba.argmax(1))
    y_bin   = (y_test != benign_idx).astype(int)
    p_dga   = 1.0 - proba[:, benign_idx]
    bin_acc = accuracy_score(y_bin, (p_dga >= 0.5).astype(int))
    bin_f1  = f1_score(y_bin, (p_dga >= 0.5).astype(int))
    bin_auc = roc_auc_score(y_bin, p_dga)
    log.info(f"  MC-acc={mc_acc:.4f}  Bin-acc={bin_acc:.4f}  "
             f"F1={bin_f1:.4f}  AUC={bin_auc:.4f}")
    results["cls"] = {
        "mc_accuracy":  float(mc_acc),
        "bin_accuracy": float(bin_acc),
        "bin_f1":       float(bin_f1),
        "bin_roc_auc":  float(bin_auc),
    }

    # ── 11. Save results ──────────────────────────────────────────────────
    results["params"] = vars(args)
    with open(str(out_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("\n" + "=" * 65)
    log.info("  SUMMARY")
    log.info("=" * 65)
    log.info(f"  Classification (test_known): Bin-acc={bin_acc:.4f}  AUC={bin_auc:.4f}")
    for sname in ["energy", "msp", "knn"]:
        log.info(f"\n  OOD ({sname}):")
        for split in ood_splits_eval:
            m = results[f"ood_{sname}_{split}"]
            label = "unknown_ood (held-out)" if split == "oe_test" else split
            log.info(f"    {label:25s}: AUROC={m['auroc']:.4f}  FPR@95={m['fpr_at_tpr']:.4f}")

    log.info(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
