"""
app.py – Streamlit demo for Open-Set DGA Detection.

Usage:
    streamlit run app.py
"""

import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
EDA_DATA_PATH = ROOT / "data" / "eda" / "sample_features.csv"

_CLASSES = [
    "benign", "conficker", "cryptolocker", "dircrypt", "emotet", "gozi",
    "kraken", "matsnu", "murofet", "necurs", "nymaim", "padcrypt",
    "pykspa", "ramdo", "ramnit", "rovnix", "simda", "suppobox", "tinba",
]
N_CLASSES = len(_CLASSES)

# ── tokenization ──────────────────────────────────────────────────────────────

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS) + 2
MAX_LEN = 75


def tokenize_batch(domains: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    out = np.zeros((len(domains), max_len), dtype=np.int32)
    for i, d in enumerate(domains):
        d = d.lower().strip()
        for j, ch in enumerate(d[:max_len]):
            out[i, j] = _CHAR2IDX.get(ch, 1)
    return out


# ── neural model definitions ──────────────────────────────────────────────────

class DomainBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, hidden_dim=128,
                 num_layers=2, feat_dim=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.pre = nn.Sequential(nn.Linear(hidden_dim * 2, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor):
        e = self.embed(x)
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        B, _, DH = out.shape
        H = DH // 2
        idx = (lengths - 1).clamp(min=0).long()
        fwd = out[torch.arange(B, device=out.device), idx, :H]
        bwd = out[:, 0, H:]
        pooled = self.drop(torch.cat([fwd, bwd], dim=-1))
        feat = self.pre(pooled)
        return self.head(feat), feat


class DomainCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, feat_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.pre = nn.Sequential(nn.Linear(256, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)
        h = self.conv3(self.conv2(self.conv1(e)))
        feat = self.pre(self.pool(h).squeeze(-1))
        return self.head(feat), feat


# ── model loaders (cached) ────────────────────────────────────────────────────

def _load_state(path, model):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading BiLSTM…")
def _load_bilstm():
    with open(MODELS_DIR / "bilstm_meta.json") as f:
        p = json.load(f)["params"]
    m = DomainBiLSTM(VOCAB_SIZE, p["embed_dim"], N_CLASSES,
                     p["hidden_dim"], p["num_layers"], p["feat_dim"], p["dropout"])
    return _load_state(MODELS_DIR / "bilstm_model.pt", m), float(p["energy_T"])


@st.cache_resource(show_spinner="Loading BiLSTM-OE…")
def _load_bilstm_oe():
    with open(MODELS_DIR / "bilstm_oe_meta.json") as f:
        p = json.load(f)["params"]
    m = DomainBiLSTM(VOCAB_SIZE, p["embed_dim"], N_CLASSES,
                     p["hidden_dim"], p["num_layers"], p["feat_dim"], p["dropout"])
    return _load_state(MODELS_DIR / "bilstm_oe_model.pt", m), float(p["energy_T"])


@st.cache_resource(show_spinner="Loading CNN…")
def _load_cnn():
    with open(MODELS_DIR / "neural_meta.json") as f:
        p = json.load(f)["params"]
    m = DomainCNN(VOCAB_SIZE, p["embed_dim"], N_CLASSES, p["feat_dim"])
    return _load_state(MODELS_DIR / "neural_model.pt", m)


@st.cache_resource(show_spinner="Loading LGBM Binary…")
def _load_lgbm():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODELS_DIR / "lgbm_binary.txt"))


@st.cache_resource(show_spinner="Loading LGBM Multiclass (decompress ~42 MB)…")
def _load_lgbm_mc():
    import lightgbm as lgb
    gz_path = MODELS_DIR / "lgbm_multiclass.txt.gz"
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name
        with gzip.open(gz_path, "rb") as gz:
            tmp.write(gz.read())
    booster = lgb.Booster(model_file=tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return booster


@st.cache_data(show_spinner="Loading EDA data…")
def load_eda_data() -> pd.DataFrame:
    return pd.read_csv(EDA_DATA_PATH)


# ── inference ─────────────────────────────────────────────────────────────────

def _neural_predict(model, domains, energy_T=1.0):
    x = torch.tensor(tokenize_batch(domains), dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    probs = F.softmax(logits, dim=-1).numpy()
    energy = -(energy_T * torch.logsumexp(logits / energy_T, dim=-1)).numpy()
    msp = 1.0 - probs.max(axis=1)
    benign_idx = _CLASSES.index("benign")
    rows = []
    for i, domain in enumerate(domains):
        pred_idx = int(probs[i].argmax())
        rows.append({
            "domain": domain,
            "predicted_class": _CLASSES[pred_idx],
            "confidence": float(probs[i, pred_idx]),
            "dga_prob": float(1.0 - probs[i, benign_idx]),
            "msp_score": float(msp[i]),
            "energy_score": float(energy[i]),
            "probabilities": probs[i].tolist(),
        })
    return rows


def _lgbm_predict(booster, domains):
    from features import extract_features_batch
    feats = extract_features_batch(domains)
    scores = booster.predict(feats)         # P(DGA)
    rows = []
    for i, domain in enumerate(domains):
        dga_prob = float(scores[i])
        pred = "DGA" if dga_prob >= 0.5 else "benign"
        rows.append({
            "domain": domain,
            "predicted_class": pred,
            "confidence": float(max(dga_prob, 1 - dga_prob)),
            "dga_prob": dga_prob,
            "msp_score": float(1.0 - max(dga_prob, 1 - dga_prob)),
            "energy_score": float("nan"),
            "probabilities": None,
        })
    return rows


def _lgbm_mc_predict(booster, domains):
    from features import extract_features_batch
    feats = extract_features_batch(domains)
    probs = booster.predict(feats)          # (N, 19) softmax probabilities
    benign_idx = _CLASSES.index("benign")
    rows = []
    for i, domain in enumerate(domains):
        pred_idx = int(probs[i].argmax())
        dga_prob = float(1.0 - probs[i, benign_idx])
        conf = float(probs[i, pred_idx])
        rows.append({
            "domain": domain,
            "predicted_class": _CLASSES[pred_idx],
            "confidence": conf,
            "dga_prob": dga_prob,
            "msp_score": float(1.0 - conf),
            "energy_score": float("nan"),
            "probabilities": probs[i].tolist(),
        })
    return rows


MODEL_OPTIONS = {
    "BiLSTM (best)": "bilstm",
    "BiLSTM + Outlier Exposure": "bilstm_oe",
    "CNN": "cnn",
    "LGBM Multiclass (19-class)": "lgbm_mc",
    "LGBM Binary": "lgbm",
}


def run_inference(model_key: str, domains: list[str]) -> list[dict]:
    if model_key == "bilstm":
        model, energy_T = _load_bilstm()
        return _neural_predict(model, domains, energy_T)
    if model_key == "bilstm_oe":
        model, energy_T = _load_bilstm_oe()
        return _neural_predict(model, domains, energy_T)
    if model_key == "cnn":
        model = _load_cnn()
        return _neural_predict(model, domains)
    if model_key == "lgbm":
        booster = _load_lgbm()
        return _lgbm_predict(booster, domains)
    if model_key == "lgbm_mc":
        booster = _load_lgbm_mc()
        return _lgbm_mc_predict(booster, domains)
    raise ValueError(model_key)


# ── UI helpers ────────────────────────────────────────────────────────────────

_DGA_FAMILIES = {
    "conficker", "cryptolocker", "dircrypt", "emotet", "gozi", "kraken",
    "matsnu", "murofet", "necurs", "nymaim", "padcrypt", "pykspa",
    "ramdo", "ramnit", "rovnix", "simda", "suppobox", "tinba",
}


def _ood_score(row: dict, scorer: str) -> float:
    """Return the OOD score for the chosen scorer."""
    if scorer == "Energy":
        v = row.get("energy_score", float("nan"))
        # Energy is negative: more negative = more in-dist.
        # Normalise to [0,1]-ish by negating and shifting so higher = more OOD.
        return float(-v) if not (v != v) else 0.0   # nan → 0
    return float(row["msp_score"])   # MSP: already 0-1, higher = more OOD


def _verdict(row: dict, threshold: float, scorer: str = "MSP") -> tuple[str, str]:
    score = _ood_score(row, scorer)
    # For Energy we need a different scale — use quantile-free comparison:
    # threshold slider is always 0-1; Energy raw values are negative floats.
    # We map: threshold on slider → same threshold on MSP; for Energy we
    # remap threshold to energy domain: energy_thresh = -threshold * 20
    if scorer == "Energy":
        energy_val = row.get("energy_score", 0.0)
        energy_thresh = -threshold * 20          # e.g. 0.5 → -10
        is_ood = float(energy_val) > energy_thresh
    else:
        is_ood = score > threshold

    cls = row["predicted_class"]
    if is_ood:
        return "Unknown / OOD", "#e74c3c"
    if cls == "benign":
        return "Benign", "#27ae60"
    return f"DGA – {cls}", "#e67e22"


def _badge(label, color):
    return (f'<span style="background:{color};color:white;padding:3px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600">{label}</span>')


def _style_verdict(val):
    if val == "Benign":
        return "background-color:#d5f5e3;color:#1a5c35"
    if val == "Unknown / OOD":
        return "background-color:#fadbd8;color:#922b21"
    return "background-color:#fdebd0;color:#7d4608"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Open-Set DGA Detection", page_icon="🔍", layout="wide")
st.title("Open-Set DGA Detection")
st.caption(
    "Phát hiện tên miền DGA sử dụng nhiều mô hình học sâu và cây quyết định · "
    "Open-set: phát hiện cả DGA chưa thấy trong huấn luyện"
)

tab_detect, tab_compare, tab_eda = st.tabs(["Domain Detection", "Model Comparison", "EDA Dashboard"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DOMAIN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_SCORER_OPTIONS = {
    "bilstm":    ["MSP", "Energy"],
    "bilstm_oe": ["MSP", "Energy"],
    "cnn":       ["MSP", "Energy"],
    "lgbm_mc":   ["MSP"],
    "lgbm":      ["MSP"],
}

with tab_detect:
    with st.sidebar:
        st.header("Settings")
        selected_model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        selected_model_key = MODEL_OPTIONS[selected_model_label]

        available_scorers = _SCORER_OPTIONS[selected_model_key]
        selected_scorer = st.selectbox(
            "OOD Scorer",
            options=available_scorers,
            help="MSP: 1 − max softmax prob · Energy: −T·log Σ exp(logit/T). "
                 "Higher score → more likely OOD.",
        )

        ood_threshold = st.slider(
            f"OOD threshold ({selected_scorer})",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Domains whose OOD score exceeds this are flagged as Unknown/OOD.",
        )
        st.divider()
        st.subheader("Known DGA families (19)")
        st.markdown(", ".join(sorted(_DGA_FAMILIES)))

    st.subheader("Enter domain name(s)")
    col_input, col_ex = st.columns([3, 1])

    with col_ex:
        st.markdown("**Quick examples**")
        for label, examples in {
            "Benign": ["google.com", "microsoft.com", "github.com"],
            "DGA (known)": ["iecbmjqs.com", "qnhpfstlkr.net", "tkzxymqvwj.biz"],
            "Mixed": ["amazon.com", "iecbmjqs.com", "facebook.com"],
        }.items():
            if st.button(label, use_container_width=True):
                st.session_state["domain_input"] = "\n".join(examples)

    with col_input:
        domain_text = st.text_area(
            "One domain per line",
            value=st.session_state.get("domain_input", "google.com\niecbmjqs.com"),
            height=150,
            key="domain_input",
        )

    if st.button("Analyze", type="primary"):
        domains = [d.strip() for d in domain_text.strip().splitlines() if d.strip()][:50]
        if not domains:
            st.warning("Enter at least one domain.")
        else:
            with st.spinner(f"Running {selected_model_label}…"):
                results = run_inference(selected_model_key, domains)

            st.divider()
            st.subheader(f"Results — {len(results)} domain(s)  ·  {selected_model_label}  ·  scorer: **{selected_scorer}**")

            is_lgbm = selected_model_key in ("lgbm", "lgbm_mc")
            table_rows = []
            for r in results:
                verdict, _ = _verdict(r, ood_threshold, selected_scorer)
                row = {
                    "Domain": r["domain"],
                    "Prediction": r["predicted_class"],
                    "Confidence": f"{r['confidence']:.1%}",
                    "DGA probability": f"{r['dga_prob']:.1%}",
                    "Verdict": verdict if not is_lgbm else ("Benign" if r["predicted_class"] == "benign" else "DGA"),
                }
                if not is_lgbm:
                    row["MSP score"] = f"{r['msp_score']:.4f}"
                    ev = r.get("energy_score", float("nan"))
                    row["Energy score"] = f"{ev:.4f}" if ev == ev else "—"
                    row[f"↳ Active ({selected_scorer})"] = f"{_ood_score(r, selected_scorer):.4f}"
                table_rows.append(row)

            df_res = pd.DataFrame(table_rows)
            st.dataframe(
                df_res.style.map(_style_verdict, subset=["Verdict"]),
                use_container_width=True, hide_index=True,
            )

            show_charts = selected_model_key in ("bilstm", "bilstm_oe", "cnn", "lgbm_mc")
            if show_charts and (
                len(results) == 1 or st.checkbox("Show probability charts", value=len(results) <= 5)
            ):
                for r in results:
                    verdict, color = _verdict(r, ood_threshold, selected_scorer)
                    icon = "🟢" if verdict == "Benign" else ("🔴" if verdict == "Unknown / OOD" else "🟠")
                    with st.expander(f"{icon} {r['domain']}  —  {verdict}", expanded=len(results) == 1):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Predicted class", r["predicted_class"].upper())
                        c2.metric("Confidence", f"{r['confidence']:.1%}")
                        c3.metric("DGA probability", f"{r['dga_prob']:.1%}")
                        c4.metric(f"{selected_scorer} score", f"{_ood_score(r, selected_scorer):.4f}")
                        prob_df = pd.DataFrame({
                            "Class": _CLASSES, "Probability": r["probabilities"],
                        }).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Class"), height=240)

            if len(results) > 1:
                st.divider()
                n_benign = sum(1 for r in results if r["predicted_class"] == "benign")
                n_dga = len(results) - n_benign
                a1, a2 = st.columns(2)
                a1.metric("Benign", n_benign)
                a2.metric("DGA / Unknown", n_dga)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.subheader("Model performance comparison")
    st.caption("Evaluated on test_known (40,000 domains) · threshold = 0.5 · OOD: unknown_family split")

    def _read_meta(fname):
        path = MODELS_DIR / fname
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    bilstm_m    = _read_meta("bilstm_meta.json")
    cnn_m       = _read_meta("neural_meta.json")
    bilstm_oe_m = _read_meta("bilstm_oe_meta.json")
    lgbm_m      = _read_meta("lgbm_binary_meta.json")
    lgbm_mc_m   = _read_meta("lgbm_multiclass_meta.json")

    def _g(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return default
        return d if d != {} else default

    def _fmt(v):
        if v is None or v == "—":
            return "—"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    # ── 1. Binary classification (benign vs DGA) ──────────────────────────
    st.markdown("#### Binary classification — benign vs DGA (test_known)")

    rows_bin = [
        {"Model": "LGBM Binary",
         "Accuracy":  _g(lgbm_m,     "cls_detailed", "bin_accuracy"),
         "Precision": _g(lgbm_m,     "cls_detailed", "bin_precision"),
         "Recall":    _g(lgbm_m,     "cls_detailed", "bin_recall"),
         "F1":        _g(lgbm_m,     "cls_detailed", "bin_f1"),
         "AUC":       _g(lgbm_m,     "binary_test_known", "roc_auc")},
        {"Model": "LGBM Multiclass",
         "Accuracy":  _g(lgbm_mc_m,  "cls_detailed", "bin_accuracy"),
         "Precision": _g(lgbm_mc_m,  "cls_detailed", "bin_precision"),
         "Recall":    _g(lgbm_mc_m,  "cls_detailed", "bin_recall"),
         "F1":        _g(lgbm_mc_m,  "cls_detailed", "bin_f1"),
         "AUC":       _g(lgbm_mc_m,  "cls_test_known", "bin_roc_auc")},
        {"Model": "CNN",
         "Accuracy":  _g(cnn_m,      "cls_detailed", "bin_accuracy"),
         "Precision": _g(cnn_m,      "cls_detailed", "bin_precision"),
         "Recall":    _g(cnn_m,      "cls_detailed", "bin_recall"),
         "F1":        _g(cnn_m,      "cls_detailed", "bin_f1"),
         "AUC":       _g(cnn_m,      "cls_test_known", "bin_roc_auc")},
        {"Model": "BiLSTM",
         "Accuracy":  _g(bilstm_m,   "cls_detailed", "bin_accuracy"),
         "Precision": _g(bilstm_m,   "cls_detailed", "bin_precision"),
         "Recall":    _g(bilstm_m,   "cls_detailed", "bin_recall"),
         "F1":        _g(bilstm_m,   "cls_detailed", "bin_f1"),
         "AUC":       _g(bilstm_m,   "cls_test_known", "bin_roc_auc")},
        {"Model": "BiLSTM + OE",
         "Accuracy":  _g(bilstm_oe_m,"cls_detailed", "bin_accuracy"),
         "Precision": _g(bilstm_oe_m,"cls_detailed", "bin_precision"),
         "Recall":    _g(bilstm_oe_m,"cls_detailed", "bin_recall"),
         "F1":        _g(bilstm_oe_m,"cls_detailed", "bin_f1"),
         "AUC":       _g(bilstm_oe_m,"cls", "bin_roc_auc")},
    ]
    df_bin = pd.DataFrame(rows_bin)
    for col in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        df_bin[col] = df_bin[col].apply(_fmt)
    st.dataframe(df_bin, use_container_width=True, hide_index=True)

    # bar charts for Precision / Recall / F1
    metric_chart = st.selectbox("Visualise metric", ["F1", "Precision", "Recall", "Accuracy", "AUC"], key="bin_metric")
    chart_data = df_bin.set_index("Model")[metric_chart].apply(
        lambda x: float(x) if x != "—" else None
    ).dropna().sort_values(ascending=False)
    st.bar_chart(chart_data.rename(metric_chart), height=240)

    # ── 2. Multiclass classification ──────────────────────────────────────
    st.markdown("#### Multiclass classification — 19 classes (macro-averaged)")

    rows_mc = [
        {"Model": "LGBM Binary",    "MC Accuracy": "—",                                            "Macro Precision": "—",                                                 "Macro Recall": "—",                                               "Macro F1": "—"},
        {"Model": "LGBM Multiclass","MC Accuracy": _g(lgbm_mc_m, "cls_detailed","mc_accuracy"),    "Macro Precision": _g(lgbm_mc_m,"cls_detailed","mc_precision_macro"),   "Macro Recall": _g(lgbm_mc_m,"cls_detailed","mc_recall_macro"),    "Macro F1": _g(lgbm_mc_m,"cls_detailed","mc_f1_macro")},
        {"Model": "CNN",            "MC Accuracy": _g(cnn_m,     "cls_detailed","mc_accuracy"),    "Macro Precision": _g(cnn_m,    "cls_detailed","mc_precision_macro"),   "Macro Recall": _g(cnn_m,    "cls_detailed","mc_recall_macro"),    "Macro F1": _g(cnn_m,    "cls_detailed","mc_f1_macro")},
        {"Model": "BiLSTM",         "MC Accuracy": _g(bilstm_m,  "cls_detailed","mc_accuracy"),    "Macro Precision": _g(bilstm_m, "cls_detailed","mc_precision_macro"),   "Macro Recall": _g(bilstm_m, "cls_detailed","mc_recall_macro"),    "Macro F1": _g(bilstm_m, "cls_detailed","mc_f1_macro")},
        {"Model": "BiLSTM + OE",    "MC Accuracy": _g(bilstm_oe_m,"cls_detailed","mc_accuracy"),   "Macro Precision": _g(bilstm_oe_m,"cls_detailed","mc_precision_macro"), "Macro Recall": _g(bilstm_oe_m,"cls_detailed","mc_recall_macro"),  "Macro F1": _g(bilstm_oe_m,"cls_detailed","mc_f1_macro")},
    ]
    df_mc = pd.DataFrame(rows_mc)
    for col in ["MC Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
        df_mc[col] = df_mc[col].apply(_fmt)
    st.dataframe(df_mc, use_container_width=True, hide_index=True)

    # ── 3. OOD detection ──────────────────────────────────────────────────
    st.markdown("#### OOD detection — unknown_family (best scorer per model)")
    st.caption("AUPR-out = precision-recall AUC treating OOD as positive · Precision@TPR95 = precision when recall=95%")

    rows_ood = [
        {"Model": "LGBM Binary",    "Scorer": "MSP",    "AUROC": _g(lgbm_m,      "ood_msp_unknown_family",    "auroc"), "AUPR-out": _g(lgbm_m,     "ood_msp_unknown_family","aupr_out"), "FPR@95TPR": _g(lgbm_m,     "ood_msp_unknown_family","fpr_at_tpr"), "Prec@TPR95": _g(lgbm_m,    "ood_msp_unknown_family","precision_at_tpr")},
        {"Model": "LGBM Multiclass","Scorer": "MSP",    "AUROC": _g(lgbm_mc_m,   "ood_msp_unknown_family",    "auroc"), "AUPR-out": _g(lgbm_mc_m,  "ood_msp_unknown_family","aupr_out"), "FPR@95TPR": _g(lgbm_mc_m,  "ood_msp_unknown_family","fpr_at_tpr"), "Prec@TPR95": _g(lgbm_mc_m, "ood_msp_unknown_family","precision_at_tpr")},
        {"Model": "CNN",            "Scorer": "MSP",    "AUROC": _g(cnn_m,        "ood_msp_unknown_family",    "auroc"), "AUPR-out": _g(cnn_m,      "ood_msp_unknown_family","aupr_out"), "FPR@95TPR": _g(cnn_m,      "ood_msp_unknown_family","fpr_at_tpr"), "Prec@TPR95": _g(cnn_m,     "ood_msp_unknown_family","precision_at_tpr")},
        {"Model": "BiLSTM",         "Scorer": "Energy", "AUROC": _g(bilstm_m,     "ood_energy_unknown_family", "auroc"), "AUPR-out": _g(bilstm_m,   "ood_energy_unknown_family","aupr_out"), "FPR@95TPR": _g(bilstm_m,   "ood_energy_unknown_family","fpr_at_tpr"), "Prec@TPR95": _g(bilstm_m,  "ood_energy_unknown_family","precision_at_tpr")},
        {"Model": "BiLSTM + OE",    "Scorer": "Energy", "AUROC": _g(bilstm_oe_m,  "ood_energy_unknown_family", "auroc"), "AUPR-out": _g(bilstm_oe_m,"ood_energy_unknown_family","aupr_out"), "FPR@95TPR": _g(bilstm_oe_m,"ood_energy_unknown_family","fpr_at_tpr"), "Prec@TPR95": _g(bilstm_oe_m,"ood_energy_unknown_family","precision_at_tpr")},
    ]
    df_ood = pd.DataFrame(rows_ood)
    for col in ["AUROC", "AUPR-out", "FPR@95TPR", "Prec@TPR95"]:
        df_ood[col] = df_ood[col].apply(_fmt)
    st.dataframe(df_ood, use_container_width=True, hide_index=True)

    auroc_series = pd.Series({
        r["Model"]: float(r["AUROC"]) for r in rows_ood if r["AUROC"] != "—"
    }).sort_values(ascending=False)
    st.markdown("**AUROC (unknown_family)**")
    st.bar_chart(auroc_series.rename("AUROC"), height=240)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.caption("200 samples per class · 38 lexical features extracted from domain names")

    if not EDA_DATA_PATH.exists():
        st.error("EDA data not found. Run `python extract_eda_sample.py` first.")
        st.stop()

    eda = load_eda_data()
    all_classes = sorted(eda["class_label"].unique())
    all_features = [c for c in eda.columns if c not in ("domain", "class_label")]

    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            selected_classes = st.multiselect("Filter by class", all_classes, default=all_classes)
        with f2:
            selected_feature = st.selectbox(
                "Primary feature",
                options=all_features,
                index=all_features.index("char_entropy") if "char_entropy" in all_features else 0,
            )

    if not selected_classes:
        st.warning("Select at least one class.")
        st.stop()

    eda_f = eda[eda["class_label"].isin(selected_classes)]

    st.markdown("### Class distribution")
    cc = eda_f["class_label"].value_counts().sort_values(ascending=False)
    st.bar_chart(cc.rename("count"), height=280)

    st.markdown(f"### `{selected_feature}` distribution by class")
    pivot = (
        eda_f.groupby("class_label")[selected_feature]
        .describe()[["mean", "50%", "std"]]
        .rename(columns={"50%": "median"})
        .sort_values("mean", ascending=False)
    )
    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)
    st.bar_chart(
        eda_f.groupby("class_label")[selected_feature].mean().sort_values(ascending=False).rename("mean"),
        height=260,
    )

    st.markdown("### Feature comparison: benign vs DGA")
    compare_features = st.multiselect(
        "Select features to compare",
        options=all_features,
        default=[f for f in ["char_entropy", "length", "digit_ratio", "vowel_ratio",
                              "markov_log_likelihood", "compression_ratio"] if f in all_features],
    )
    if compare_features:
        tmp = eda_f.copy()
        tmp["group"] = tmp["class_label"].apply(lambda x: "benign" if x == "benign" else "DGA")
        grp = tmp.groupby("group")[compare_features].mean()
        st.dataframe(grp.T.style.format("{:.4f}"), use_container_width=True)
        norm = grp.T
        norm_s = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
        st.bar_chart(norm_s, height=280)

    st.markdown("### Top features separating benign vs DGA")
    eda_all = eda.copy()
    eda_all["group"] = eda_all["class_label"].apply(lambda x: "benign" if x == "benign" else "DGA")
    means = eda_all.groupby("group")[all_features].mean()
    if "benign" in means.index and "DGA" in means.index:
        diff = (means.loc["DGA"] - means.loc["benign"]).abs().sort_values(ascending=False)
        st.bar_chart(diff.head(10).rename("|Mean diff| DGA vs Benign"), height=260)
        st.dataframe(diff.head(10).reset_index().rename(columns={"index": "Feature", 0: "|Mean diff|"})
                     .style.format({"|Mean diff|": "{:.4f}"}), use_container_width=True, hide_index=True)

    st.markdown("### Raw data sample")
    n_show = st.slider("Rows to display", 10, 200, 50, step=10)
    show_cols = ["domain", "class_label"] + (compare_features if compare_features else all_features[:8])
    st.dataframe(
        eda_f[show_cols].sample(min(n_show, len(eda_f)), random_state=0),
        use_container_width=True, hide_index=True,
    )
