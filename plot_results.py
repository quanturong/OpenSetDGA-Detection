"""
plot_results.py â€” Trá»±c quan hÃ³a káº¿t quáº£ thá»±c nghiá»‡m Open-Set DGA Detection
Sinh 3 biá»ƒu Ä‘á»“ + 1 báº£ng tá»•ng há»£p vÃ o thÆ° má»¥c figures/
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# â”€â”€ style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GRAY   = "#6B7280"
GREEN  = "#16A34A"
RED    = "#DC2626"

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Figure 1 â€” Classification accuracy (MC-Acc / Binary AUC)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
models   = ["20-class\nLGBM", "Char-CNN", "BiLSTM"]
mc_acc   = [0.654,  0.926,  0.935]
bin_auc  = [0.946,  0.995,  0.996]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, mc_acc, w, label="MC-Accuracy", color=BLUE,   zorder=3)
bars2 = ax.bar(x + w/2, bin_auc, w, label="Binary AUC",  color=ORANGE, zorder=3)

# annotate
for bar, val in zip(bars1, mc_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

for bar, val in zip(bars2, bin_auc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=ORANGE, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Cháº¥t lÆ°á»£ng phÃ¢n loáº¡i (ID â€” test_known)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)

# highlight BiLSTM
ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, alpha=0.06, color=GREEN, zorder=0)
ax.text(x[-1], 1.04, "â˜… Best", ha="center", fontsize=10, color=GREEN, fontweight="bold")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_classification.png", dpi=150)
plt.close(fig)
print(f"[âœ“] {OUT_DIR}/fig1_classification.png")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Figure 2 â€” OOD unknown_family: chá»‰ best + 2 baseline Ä‘áº¡i diá»‡n + insight
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Giá»¯ láº¡i:
#   LGBM MSP   â†’ baseline ML cá»• Ä‘iá»ƒn
#   BiLSTM MSP â†’ baseline neural (Ä‘á»ƒ tháº¥y Energy >> MSP)
#   BiLSTM Energy â†’ BEST
uf_labels = ["LGBM\nMSP", "Char-CNN\nMSP", "BiLSTM\nMSP", "BiLSTM\nEnergy\n(best)"]
uf_auroc  = [0.550,       0.670,           0.661,           0.836               ]
uf_colors = [GRAY,        GRAY,            GRAY,            GREEN               ]

fig, ax = plt.subplots(figsize=(7, 4.8))
bars = ax.barh(uf_labels, uf_auroc, color=uf_colors, zorder=3, height=0.5)

for bar, val, color in zip(bars, uf_auroc, uf_colors):
    ax.text(val + 0.004, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=12,
            color=color, fontweight="bold")

ax.set_xlim(0.4, 0.93)
ax.set_xlabel("AUROC", fontsize=11)
ax.set_title("OOD â€” unknown_family (há» DGA chÆ°a tháº¥y)",
             fontsize=12, fontweight="bold", pad=10)
ax.axvline(0.5, color=RED, linestyle="--", alpha=0.4, linewidth=1.2, label="Random (0.5)")

bars[-1].set_edgecolor(GREEN)
bars[-1].set_linewidth(2)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_ood_unknown_family.png", dpi=150)
plt.close(fig)
print(f"[âœ“] {OUT_DIR}/fig2_ood_unknown_family.png")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Figure 3 â€” OOD unknown_ood: chá»‰ best + 2 baseline Ä‘áº¡i diá»‡n + insight
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Giá»¯ láº¡i:
#   LGBM MSP      â†’ baseline ML cá»• Ä‘iá»ƒn
#   BiLSTM Energy â†’ baseline neural tá»‘t nháº¥t trÆ°á»›c KNN (Ä‘á»ƒ tháº¥y KNN >> Energy á»Ÿ split nÃ y)
#   BiLSTM KNN k=5 â†’ BEST
oo_labels = ["LGBM\nMSP", "BiLSTM\nEnergy", "Char-CNN\nKNN k=5", "BiLSTM\nKNN k=5\n(best)"]
oo_auroc  = [0.529,       0.571,           0.680,              0.707              ]
oo_colors = [GRAY,        GRAY,            BLUE,               GREEN              ]

fig, ax = plt.subplots(figsize=(7, 4.8))
bars = ax.barh(oo_labels, oo_auroc, color=oo_colors, zorder=3, height=0.5)

for bar, val, color in zip(bars, oo_auroc, oo_colors):
    ax.text(val + 0.004, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=12,
            color=color, fontweight="bold")

ax.set_xlim(0.35, 0.78)
ax.set_xlabel("AUROC", fontsize=11)
ax.set_title("OOD â€” unknown_ood (domain legitimate)",
             fontsize=12, fontweight="bold", pad=10)
ax.axvline(0.5, color=RED, linestyle="--", alpha=0.4, linewidth=1.2, label="Random (0.5)")

bars[-1].set_edgecolor(GREEN)
bars[-1].set_linewidth(2)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_ood_unknown_ood.png", dpi=150)
plt.close(fig)
print(f"[âœ“] {OUT_DIR}/fig3_ood_unknown_ood.png")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Figure 4 â€” so sÃ¡nh Energy vs KNN trÃªn hai split (insight chÃ­nh)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
splits  = ["unknown_family\n(há» DGA má»›i)", "unknown_ood\n(domain legitimate)"]
energy  = [0.836, 0.571]
knn     = [0.647, 0.707]

x = np.arange(len(splits))
w = 0.3

fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - w/2, energy, w, label="BiLSTM + Energy", color=ORANGE, zorder=3)
b2 = ax.bar(x + w/2, knn,    w, label="BiLSTM + KNN k=5", color=BLUE,   zorder=3)

for bar, val in zip(b1, energy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=11, color=ORANGE, fontweight="bold")
for bar, val in zip(b2, knn):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=11, color=BLUE, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=11)
ax.set_ylim(0.4, 0.95)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("Energy vs KNN â€” BiLSTM",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.axhline(0.5, color=RED, linestyle="--", alpha=0.4, linewidth=1)


fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_energy_vs_knn.png", dpi=150)
plt.close(fig)
print(f"[âœ“] {OUT_DIR}/fig4_energy_vs_knn.png")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Figure 5 â€” PhÃ¢n phá»‘i OOD score (Confidence Score Distribution)
#            BiLSTM + Energy  (best scorer cho unknown_family)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
import pandas as pd

_bilstm_dir = "baseline_out/bilstm_20260414_222503"
_score_files = {
    "known (ID)":          f"{_bilstm_dir}/scores_energy_known.csv",
    "unknown family\n(DGA má»›i)": f"{_bilstm_dir}/scores_energy_unknown_family.csv",
    "unknown OOD\n(legitimate)": f"{_bilstm_dir}/scores_energy_unknown_ood.csv",
}

_loaded = {}
for _label, _path in _score_files.items():
    try:
        _df = pd.read_csv(_path)
        _loaded[_label] = _df["ood_score"].dropna().values
    except FileNotFoundError:
        print(f"  [skip] {_path} not found")

if len(_loaded) >= 2:
    _colors = {
        "known (ID)":          BLUE,
        "unknown family\n(DGA má»›i)": ORANGE,
        "unknown OOD\n(legitimate)": GRAY,
    }
    _SAMPLE = 10_000  # subsample for speed

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel A: histogram (overlaid) ---
    ax = axes[0]
    rng = np.random.default_rng(42)
    for _label, _scores in _loaded.items():
        _s = rng.choice(_scores, size=min(_SAMPLE, len(_scores)), replace=False)
        ax.hist(_s, bins=80, density=True, alpha=0.45,
                color=_colors.get(_label, GRAY), label=_label)
    ax.set_xlabel("OOD Score (Energy)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("PhÃ¢n phá»‘i OOD Score â€” BiLSTM Energy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # --- panel B: box plot ---
    ax2 = axes[1]
    _box_data  = list(_loaded.values())
    _box_labels = list(_loaded.keys())
    _box_colors = [_colors.get(l, GRAY) for l in _box_labels]

    bp = ax2.boxplot(_box_data, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=2),
                     flierprops=dict(marker=".", markersize=1, alpha=0.3),
                     widths=0.5)
    for patch, c in zip(bp["boxes"], _box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax2.set_xticks(range(1, len(_box_labels) + 1))
    ax2.set_xticklabels(_box_labels, fontsize=9)
    ax2.set_ylabel("OOD Score (Energy)", fontsize=11)
    ax2.set_title("Box Plot â€” phÃ¢n tÃ¡ch ID vs OOD", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Confidence Score Distribution: in-distribution vs OOD (BiLSTM + Energy)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[âœ“] {OUT_DIR}/fig5_score_distribution.png")
else:
    print("  [skip] fig5: khÃ´ng Ä‘á»§ score CSV, cháº¡y train_bilstm.py trÆ°á»›c")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Figure 6 â€” Benign overfit analysis: per-source OOD score breakdown
#            Chá»©ng minh tranco_tail â‰ˆ train benign â†’ model khÃ´ng phÃ¢n biá»‡t Ä‘Æ°á»£c
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_bilstm_dir = "baseline_out/bilstm_20260414_222503"
_dataset_dir = "dataset_out/run_20260222_193219"

try:
    _ood_meta  = pd.read_csv(f"{_dataset_dir}/unknown_ood/test_unknown_ood.csv")
    _ood_score = pd.read_csv(f"{_bilstm_dir}/scores_energy_unknown_ood.csv")
    # scores_energy_known.csv = ID (known) test split scores — use directly as reference
    _known_sc  = pd.read_csv(f"{_bilstm_dir}/scores_energy_known.csv")

    _ood_df = _ood_meta.merge(
        _ood_score.rename(columns={"ood_score": "energy"}), on="domain", how="inner"
    )
    # ID reference: all known test scores (benign + DGA — reflects what model sees as in-distribution)
    _id_scores = _known_sc["ood_score"].dropna().values

    _src_rename = {
        "phishing.army:phishing_army_blocklist.txt":       "phishing.army",
        "raw.githubusercontent.com:hagezi/dns-blocklists": "hagezi blocklist",
        "raw.githubusercontent.com:stamparm/blackbook":    "stamparm blackbook",
        "urlhaus.abuse.ch:text_online":                    "URLhaus",
        "openphish.com:feed.txt":                          "OpenPhish",
        "360netlab":                                       "360netlab",
        "crtsh":                                           "crt.sh",
        "tranco_tail":                                     "tranco_tail",
    }
    _ood_df["src_short"] = _ood_df["source"].map(_src_rename).fillna(_ood_df["source"])

    _src_stats = (
        _ood_df.groupby("src_short")["energy"]
        .median()
        .sort_values()
        .reset_index()
    )
    _ref_med = float(np.median(_id_scores))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- panel A: median energy by source ---
    ax = axes[0]
    _bar_colors = [
        ORANGE if src == "tranco_tail" else (BLUE if src == "crt.sh" else GRAY)
        for src in _src_stats["src_short"]
    ]

    bars = ax.barh(_src_stats["src_short"], _src_stats["energy"],
                   color=_bar_colors, zorder=3, height=0.6)
    ax.axvline(_ref_med, color=GREEN, linewidth=2, linestyle="--",
               label=f"known (ID) median ({_ref_med:.2f})")
    ax.set_xlabel("Median Energy OOD Score (higher = more OOD)", fontsize=11)
    ax.set_title("Median OOD Score per Source\n(BiLSTM + Energy)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, _src_stats["energy"]):
        ax.text(val - 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")

    _src_list = list(_src_stats["src_short"])
    if "tranco_tail" in _src_list:
        _tt_idx = _src_list.index("tranco_tail") + 1
        ax.annotate("~ known (ID)\n(hard to distinguish)",
                    xy=(_ref_med, _tt_idx - 1),
                    xytext=(_ref_med - 3.5, _tt_idx - 1 + 1.8),
                    fontsize=8, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))

    # --- panel B: histogram tranco_tail vs known vs 360netlab ---
    ax2 = axes[1]
    _rng = np.random.default_rng(42)

    _groups = {
        "known (ID — test_known)":   (_id_scores,  GREEN),
        "tranco_tail (OOD legit)":   (_ood_df[_ood_df["src_short"] == "tranco_tail"]["energy"].dropna().values, ORANGE),
        "360netlab (OOD malicious)": (_ood_df[_ood_df["src_short"] == "360netlab"]["energy"].dropna().values, RED),
    }
    for _lbl, (_sc, _c) in _groups.items():
        if len(_sc) == 0:
            continue
        _s = _rng.choice(_sc, size=min(5000, len(_sc)), replace=False)
        ax2.hist(_s, bins=60, density=True, alpha=0.50, color=_c, label=_lbl)

    ax2.set_xlabel("Energy OOD Score", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Score distribution: known vs tranco_tail vs 360netlab\n"
                  "(tranco_tail overlaps with known ID -> hard to detect)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Benign Overfit Analysis — Model cannot distinguish tranco_tail from known benign",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_benign_overfit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {OUT_DIR}/fig6_benign_overfit.png")

except FileNotFoundError as _e:
    print(f"  [skip] fig6: {_e}")

print("\nDone! All figures saved to figures/")
