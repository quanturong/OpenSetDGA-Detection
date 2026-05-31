"""
app.py – Streamlit demo for Open-Set DGA Detection.

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent

BILSTM_MODEL_PATH = ROOT / "models" / "bilstm_model.pt"
BILSTM_META_PATH = ROOT / "models" / "bilstm_meta.json"
EDA_DATA_PATH = ROOT / "data" / "eda" / "sample_features.csv"

# ── tokenization ──────────────────────────────────────────────────────────────

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
    def __init__(self, vocab_size, embed_dim, n_classes, hidden_dim=128,
                 num_layers=2, feat_dim=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
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


# ── loaders (cached) ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(BILSTM_META_PATH) as f:
        meta = json.load(f)
    p = meta["params"]
    model = DomainBiLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=p["embed_dim"],
        n_classes=meta["n_classes"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        feat_dim=p["feat_dim"],
        dropout=p["dropout"],
    )
    state = torch.load(BILSTM_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta["classes"], float(p["energy_T"])


@st.cache_data(show_spinner="Loading EDA data…")
def load_eda_data() -> pd.DataFrame:
    return pd.read_csv(EDA_DATA_PATH)


# ── inference ─────────────────────────────────────────────────────────────────

def predict(model: DomainBiLSTM, classes: list[str], energy_T: float,
            domains: list[str]) -> list[dict]:
    tokens = tokenize_batch(domains)
    x = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    probs = F.softmax(logits, dim=-1).numpy()
    energy = -(energy_T * torch.logsumexp(logits / energy_T, dim=-1)).numpy()
    msp = 1.0 - probs.max(axis=1)

    benign_idx = classes.index("benign")
    rows = []
    for i, domain in enumerate(domains):
        pred_idx = int(probs[i].argmax())
        rows.append({
            "domain": domain,
            "predicted_class": classes[pred_idx],
            "confidence": float(probs[i, pred_idx]),
            "dga_prob": float(1.0 - probs[i, benign_idx]),
            "msp_score": float(msp[i]),
            "energy_score": float(energy[i]),
            "probabilities": probs[i].tolist(),
        })
    return rows


# ── UI helpers ────────────────────────────────────────────────────────────────

_DGA_FAMILIES = {
    "conficker", "cryptolocker", "dircrypt", "emotet", "gozi", "kraken",
    "matsnu", "murofet", "necurs", "nymaim", "padcrypt", "pykspa",
    "ramdo", "ramnit", "rovnix", "simda", "suppobox", "tinba",
}


def _verdict(row: dict, ood_threshold: float) -> tuple[str, str]:
    is_ood = row["msp_score"] > ood_threshold
    cls = row["predicted_class"]
    if is_ood:
        return "Unknown / OOD", "#e74c3c"
    if cls == "benign":
        return "Benign", "#27ae60"
    return f"DGA – {cls}", "#e67e22"


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600">{label}</span>'
    )


def _style_verdict(val: str) -> str:
    if val == "Benign":
        return "background-color:#d5f5e3;color:#1a5c35"
    if val == "Unknown / OOD":
        return "background-color:#fadbd8;color:#922b21"
    return "background-color:#fdebd0;color:#7d4608"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Open-Set DGA Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("Open-Set DGA Detection")
st.caption(
    "Character-level **BiLSTM** · 19 classes (benign + 18 DGA families) · "
    "Open-set: flags unknown DGA variants via Energy & MSP scoring"
)

tab_detect, tab_eda = st.tabs(["Domain Detection", "EDA Dashboard"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DOMAIN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_detect:
    # ── sidebar settings ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        ood_threshold = st.slider(
            "OOD threshold (MSP score)",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Domains whose MSP score exceeds this are flagged as unknown/OOD.",
        )
        st.divider()
        st.subheader("Model")
        st.markdown(
            "**Architecture:** BiLSTM (char-level)  \n"
            "**Hidden:** 64 · **Layers:** 1 · **Embed:** 32  \n"
            "**Binary acc:** 97.6 %  \n"
            "**Binary AUC:** 99.7 %  \n"
            "**Energy AUROC (unknown):** 86.1 %"
        )
        st.divider()
        st.subheader("Known DGA families")
        st.markdown(", ".join(sorted(_DGA_FAMILIES)))

    # ── load model ────────────────────────────────────────────────────────
    try:
        model, classes, energy_T = load_model()
    except FileNotFoundError:
        st.error("Model checkpoint not found at `models/bilstm_model.pt`.")
        st.stop()

    # ── input ─────────────────────────────────────────────────────────────
    st.subheader("Enter domain name(s)")
    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Quick examples**")
        example_groups = {
            "Benign": ["google.com", "microsoft.com", "github.com"],
            "DGA (known)": ["iecbmjqs.com", "qnhpfstlkr.net", "tkzxymqvwj.biz"],
            "Mixed": ["amazon.com", "iecbmjqs.com", "facebook.com"],
        }
        for label, examples in example_groups.items():
            if st.button(label, use_container_width=True):
                st.session_state["domain_input"] = "\n".join(examples)

    with col_input:
        domain_text = st.text_area(
            "One domain per line",
            value=st.session_state.get("domain_input", "google.com\niecbmjqs.com"),
            height=150,
            key="domain_input",
            placeholder="e.g.\ngoogle.com\niecbmjqs.com\nmycompany.io",
        )

    run_btn = st.button("Analyze", type="primary")

    if run_btn or domain_text:
        domains = [d.strip() for d in domain_text.strip().splitlines() if d.strip()]
        if not domains:
            st.warning("Please enter at least one domain name.")
        else:
            domains = domains[:50]
            results = predict(model, classes, energy_T, domains)

            st.divider()
            st.subheader(f"Results — {len(results)} domain(s)")

            table_rows = []
            for r in results:
                verdict, _ = _verdict(r, ood_threshold)
                table_rows.append({
                    "Domain": r["domain"],
                    "Prediction": r["predicted_class"],
                    "Confidence": f"{r['confidence']:.1%}",
                    "DGA probability": f"{r['dga_prob']:.1%}",
                    "MSP score": f"{r['msp_score']:.4f}",
                    "Energy score": f"{r['energy_score']:.4f}",
                    "Verdict": verdict,
                })

            df_res = pd.DataFrame(table_rows)
            st.dataframe(
                df_res.style.map(_style_verdict, subset=["Verdict"]),
                use_container_width=True,
                hide_index=True,
            )

            if len(results) == 1 or st.checkbox("Show probability charts", value=len(results) <= 5):
                for r in results:
                    verdict, color = _verdict(r, ood_threshold)
                    badge_html = _badge(verdict, color)
                    with st.expander(f"**{r['domain']}** — {badge_html}", expanded=(len(results) == 1)):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Predicted class", r["predicted_class"].upper())
                        c2.metric("Confidence", f"{r['confidence']:.1%}")
                        c3.metric("DGA probability", f"{r['dga_prob']:.1%}")
                        c4.metric("MSP / Energy", f"{r['msp_score']:.3f} / {r['energy_score']:.2f}")
                        prob_df = pd.DataFrame({
                            "Class": classes,
                            "Probability": r["probabilities"],
                        }).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Class"), height=260)

            if len(results) > 1:
                st.divider()
                st.subheader("Aggregate summary")
                n_benign = sum(1 for r in results if _verdict(r, ood_threshold)[0] == "Benign")
                n_ood = sum(1 for r in results if _verdict(r, ood_threshold)[0] == "Unknown / OOD")
                n_dga = len(results) - n_benign - n_ood
                a1, a2, a3 = st.columns(3)
                a1.metric("Benign", n_benign)
                a2.metric("Known DGA", n_dga)
                a3.metric("Unknown / OOD", n_ood)

                family_counts = {}
                for r in results:
                    v, _ = _verdict(r, ood_threshold)
                    if v not in ("Benign", "Unknown / OOD"):
                        fam = r["predicted_class"]
                        family_counts[fam] = family_counts.get(fam, 0) + 1
                if family_counts:
                    st.markdown("**Detected DGA families:**")
                    fam_df = (
                        pd.DataFrame(list(family_counts.items()), columns=["Family", "Count"])
                        .sort_values("Count", ascending=False)
                    )
                    st.bar_chart(fam_df.set_index("Family"), height=200)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA DASHBOARD
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

    # ── filters ───────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            selected_classes = st.multiselect(
                "Filter by class",
                options=all_classes,
                default=all_classes,
                help="Select which classes to display",
            )
        with f2:
            selected_feature = st.selectbox(
                "Primary feature (for distribution plots)",
                options=all_features,
                index=all_features.index("char_entropy") if "char_entropy" in all_features else 0,
            )

    if not selected_classes:
        st.warning("Select at least one class.")
        st.stop()

    eda_filtered = eda[eda["class_label"].isin(selected_classes)]

    # ── section 1: class distribution ────────────────────────────────────
    st.markdown("### Class distribution")
    class_counts = eda_filtered["class_label"].value_counts().reset_index()
    class_counts.columns = ["class_label", "count"]
    class_counts = class_counts.sort_values("count", ascending=False)
    st.bar_chart(class_counts.set_index("class_label"), height=300)

    # ── section 2: feature distribution by class ──────────────────────────
    st.markdown(f"### Distribution of `{selected_feature}` by class")
    pivot = (
        eda_filtered.groupby("class_label")[selected_feature]
        .describe()[["mean", "50%", "std"]]
        .rename(columns={"50%": "median"})
        .sort_values("mean", ascending=False)
    )
    st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)

    # Box-plot style using mean ± std bars
    st.markdown("**Mean ± 1 std per class**")
    mean_vals = eda_filtered.groupby("class_label")[selected_feature].mean().sort_values(ascending=False)
    st.bar_chart(mean_vals.rename("mean"), height=280)

    # ── section 3: feature comparison ─────────────────────────────────────
    st.markdown("### Feature comparison: benign vs DGA")
    compare_features = st.multiselect(
        "Select features to compare",
        options=all_features,
        default=["char_entropy", "length", "digit_ratio", "vowel_ratio",
                 "markov_log_likelihood", "compression_ratio"][:min(6, len(all_features))],
    )

    if compare_features:
        eda_filtered2 = eda_filtered.copy()
        eda_filtered2["group"] = eda_filtered2["class_label"].apply(
            lambda x: "benign" if x == "benign" else "DGA"
        )
        grp_mean = eda_filtered2.groupby("group")[compare_features].mean()
        st.dataframe(grp_mean.T.style.format("{:.4f}"), use_container_width=True)

        st.markdown("**Normalized mean per feature (benign vs DGA)**")
        norm = grp_mean.T
        norm_scaled = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
        st.bar_chart(norm_scaled, height=300)

    # ── section 4: raw sample table ───────────────────────────────────────
    st.markdown("### Raw data sample")
    n_show = st.slider("Rows to display", 10, 200, 50, step=10)
    display_cols = ["domain", "class_label"] + compare_features if compare_features else ["domain", "class_label"] + all_features[:8]
    st.dataframe(
        eda_filtered[display_cols].sample(min(n_show, len(eda_filtered)), random_state=0),
        use_container_width=True,
        hide_index=True,
    )

    # ── section 5: top features separating benign vs DGA ──────────────────
    st.markdown("### Top features separating benign vs DGA")
    st.caption("Ranked by absolute difference in class means (benign vs all DGA)")
    if len(selected_classes) > 1:
        eda_all = eda.copy()
        eda_all["group"] = eda_all["class_label"].apply(lambda x: "benign" if x == "benign" else "DGA")
        means = eda_all.groupby("group")[all_features].mean()
        if "benign" in means.index and "DGA" in means.index:
            diff = (means.loc["DGA"] - means.loc["benign"]).abs().sort_values(ascending=False)
            top10 = diff.head(10).reset_index()
            top10.columns = ["Feature", "|Mean difference|"]
            st.dataframe(top10.style.format({"| Mean difference|": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.bar_chart(diff.head(10).rename("|Mean diff| DGA vs Benign"), height=280)
