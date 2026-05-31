"""
Compute precision / recall / F1 per model on test_known and
append them into each model's meta JSON under "cls_detailed".
"""
import gzip, json, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS) + 2
MAX_LEN = 75
_CLASSES = [
    "benign","conficker","cryptolocker","dircrypt","emotet","gozi",
    "kraken","matsnu","murofet","necurs","nymaim","padcrypt",
    "pykspa","ramdo","ramnit","rovnix","simda","suppobox","tinba",
]
N_CLASSES = len(_CLASSES)


def tokenize_batch(domains, max_len=MAX_LEN):
    out = np.zeros((len(domains), max_len), dtype=np.int32)
    for i, d in enumerate(domains):
        d = d.lower().strip()
        for j, ch in enumerate(d[:max_len]):
            out[i, j] = _CHAR2IDX.get(ch, 1)
    return out


class DomainBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, hidden_dim=128,
                 num_layers=2, feat_dim=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.pre = nn.Sequential(nn.Linear(hidden_dim * 2, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        e = self.embed(x)
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        B, _, DH = out.shape; H = DH // 2
        idx = (lengths - 1).clamp(min=0).long()
        fwd = out[torch.arange(B), idx, :H]
        bwd = out[:, 0, H:]
        pooled = self.drop(torch.cat([fwd, bwd], dim=-1))
        feat = self.pre(pooled)
        return self.head(feat), feat


class DomainCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, feat_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Sequential(nn.Conv1d(embed_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU())
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.pre = nn.Sequential(nn.Linear(256, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)
        h = self.conv3(self.conv2(self.conv1(e)))
        feat = self.pre(self.pool(h).squeeze(-1))
        return self.head(feat), feat


def _neural_infer(model, domains, batch_size=512):
    model.eval()
    all_probs = []
    for i in range(0, len(domains), batch_size):
        batch = domains[i:i + batch_size]
        x = torch.tensor(tokenize_batch(batch), dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        all_probs.append(F.softmax(logits, dim=-1).numpy())
    return np.vstack(all_probs)


def _metrics(y_true_mc, y_pred_mc, y_true_bin, y_pred_bin):
    return {
        "bin_accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
        "bin_precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "bin_recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "bin_f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "mc_accuracy":   float(accuracy_score(y_true_mc, y_pred_mc)),
        "mc_precision_macro": float(precision_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)),
        "mc_recall_macro":    float(recall_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)),
        "mc_f1_macro":        float(f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)),
    }


def _update_meta(fname, key, data):
    path = MODELS / fname
    with open(path) as f:
        meta = json.load(f)
    meta[key] = data
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated {fname} [{key}]")


# ── load test data ──────────────────────────────────────────────────────────
test = pd.read_csv("data/processed/known/test_known.csv")
domains = test["domain"].tolist()
y_true_mc = test["class_label"].values
y_true_bin = (y_true_mc != "benign").astype(int)
benign_idx = _CLASSES.index("benign")
print(f"Test set: {len(domains)} domains")


# ── 1. BiLSTM ──────────────────────────────────────────────────────────────
print("\n[1/5] BiLSTM")
with open(MODELS / "bilstm_meta.json") as f:
    p = json.load(f)["params"]
m = DomainBiLSTM(VOCAB_SIZE, p["embed_dim"], N_CLASSES, p["hidden_dim"], p["num_layers"], p["feat_dim"], p["dropout"])
m.load_state_dict(torch.load(MODELS / "bilstm_model.pt", map_location="cpu", weights_only=True))
probs = _neural_infer(m, domains)
y_pred_mc = [_CLASSES[i] for i in probs.argmax(axis=1)]
y_pred_bin = (probs.argmax(axis=1) != benign_idx).astype(int)
_update_meta("bilstm_meta.json", "cls_detailed", _metrics(y_true_mc, y_pred_mc, y_true_bin, y_pred_bin))


# ── 2. BiLSTM OE ──────────────────────────────────────────────────────────
print("\n[2/5] BiLSTM OE")
with open(MODELS / "bilstm_oe_meta.json") as f:
    p = json.load(f)["params"]
m = DomainBiLSTM(VOCAB_SIZE, p["embed_dim"], N_CLASSES, p["hidden_dim"], p["num_layers"], p["feat_dim"], p["dropout"])
m.load_state_dict(torch.load(MODELS / "bilstm_oe_model.pt", map_location="cpu", weights_only=True))
probs = _neural_infer(m, domains)
y_pred_mc = [_CLASSES[i] for i in probs.argmax(axis=1)]
y_pred_bin = (probs.argmax(axis=1) != benign_idx).astype(int)
_update_meta("bilstm_oe_meta.json", "cls_detailed", _metrics(y_true_mc, y_pred_mc, y_true_bin, y_pred_bin))


# ── 3. CNN ──────────────────────────────────────────────────────────────────
print("\n[3/5] CNN")
with open(MODELS / "neural_meta.json") as f:
    p = json.load(f)["params"]
m = DomainCNN(VOCAB_SIZE, p["embed_dim"], N_CLASSES, p["feat_dim"])
m.load_state_dict(torch.load(MODELS / "neural_model.pt", map_location="cpu", weights_only=True))
probs = _neural_infer(m, domains)
y_pred_mc = [_CLASSES[i] for i in probs.argmax(axis=1)]
y_pred_bin = (probs.argmax(axis=1) != benign_idx).astype(int)
_update_meta("neural_meta.json", "cls_detailed", _metrics(y_true_mc, y_pred_mc, y_true_bin, y_pred_bin))


# ── 4. LGBM Binary ──────────────────────────────────────────────────────────
print("\n[4/5] LGBM Binary")
import lightgbm as lgb
from features import extract_features_batch
booster = lgb.Booster(model_file=str(MODELS / "lgbm_binary.txt"))
feats = extract_features_batch(domains)
scores = booster.predict(feats)
y_pred_bin_lgbm = (scores >= 0.5).astype(int)
y_pred_mc_lgbm = ["DGA" if p >= 0.5 else "benign" for p in scores]
det = _metrics(y_true_mc, y_pred_mc_lgbm, y_true_bin, y_pred_bin_lgbm)
_update_meta("lgbm_binary_meta.json", "cls_detailed", det)


# ── 5. LGBM Multiclass ──────────────────────────────────────────────────────
print("\n[5/5] LGBM Multiclass")
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
    tmp_path = tmp.name
    with gzip.open(MODELS / "lgbm_multiclass.txt.gz", "rb") as gz:
        tmp.write(gz.read())
booster_mc = lgb.Booster(model_file=tmp_path)
Path(tmp_path).unlink(missing_ok=True)
probs_mc = booster_mc.predict(feats)
y_pred_mc_lgbm2 = [_CLASSES[i] for i in probs_mc.argmax(axis=1)]
y_pred_bin_lgbm2 = (probs_mc.argmax(axis=1) != benign_idx).astype(int)
det = _metrics(y_true_mc, y_pred_mc_lgbm2, y_true_bin, y_pred_bin_lgbm2)
_update_meta("lgbm_multiclass_meta.json", "cls_detailed", det)

print("\nDone. Summary:")
for fname in ["bilstm_meta.json","bilstm_oe_meta.json","neural_meta.json","lgbm_binary_meta.json","lgbm_multiclass_meta.json"]:
    with open(MODELS / fname) as f:
        m = json.load(f)
    d = m.get("cls_detailed", {})
    name = fname.replace("_meta.json","")
    print(f"  {name:20s}: bin_P={d.get('bin_precision',0):.4f}  bin_R={d.get('bin_recall',0):.4f}  bin_F1={d.get('bin_f1',0):.4f}  mc_F1={d.get('mc_f1_macro',0):.4f}")
