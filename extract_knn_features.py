"""Extract BiLSTM penultimate features for KNN OOD scorer."""
import gzip, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS) + 2
MAX_LEN = 75
N_CLASSES = 19
N_PER_CLASS = 1000


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


# load model
with open(ROOT / "models/bilstm_meta.json") as f:
    meta = json.load(f)
p = meta["params"]
model = DomainBiLSTM(VOCAB_SIZE, p["embed_dim"], N_CLASSES,
                     p["hidden_dim"], p["num_layers"], p["feat_dim"], p["dropout"])
model.load_state_dict(torch.load(ROOT / "models/bilstm_model.pt", map_location="cpu", weights_only=True))
model.eval()

# sample training data
df = pd.read_csv(ROOT / "data/processed/known/train.csv")
parts = [g.sample(min(N_PER_CLASS, len(g)), random_state=42)
         for _, g in df.groupby("class_label")]
sample = pd.concat(parts).reset_index(drop=True)
print(f"Sample: {len(sample)} domains across {sample['class_label'].nunique()} classes")

# extract features
all_feats = []
batch_size = 512
domains = sample["domain"].tolist()
for i in range(0, len(domains), batch_size):
    batch = domains[i:i + batch_size]
    x = torch.tensor(tokenize_batch(batch), dtype=torch.long)
    with torch.no_grad():
        _, feat = model(x)
    all_feats.append(feat.numpy())
    if i % 5000 == 0:
        print(f"  {i}/{len(domains)}")

feats = np.vstack(all_feats).astype(np.float32)
labels = sample["class_label"].values
print(f"Features shape: {feats.shape}")

# save compressed
import io
out_path = ROOT / "models/bilstm_knn_features.npz.gz"
buf = io.BytesIO()
np.savez_compressed(buf, features=feats, labels=labels)
with gzip.open(out_path, "wb", compresslevel=9) as gz:
    gz.write(buf.getvalue())

size_mb = out_path.stat().st_size / 1024 / 1024
print(f"Saved {out_path}  ({size_mb:.1f} MB)")
