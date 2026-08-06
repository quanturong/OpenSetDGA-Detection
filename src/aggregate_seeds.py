"""
aggregate_seeds.py – Compute mean ± std of AUROC and FPR@95 across multi-seed runs.

Usage (from project root):
  python src/aggregate_seeds.py
  python src/aggregate_seeds.py --seeds 0 1 2 3 4 --latex

Reads:
  baseline_out/lgbm_binary_s{S}/results.json
  baseline_out/lgbm_multi_s{S}/results.json
  baseline_out/cnn_s{S}/results.json
  baseline_out/bilstm_s{S}/results.json
  baseline_out/bilstm_oe_s{S}/results.json
  baseline_out/extra_ood_s{S}/results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ── metric extraction ─────────────────────────────────────────────────────────

def get_metrics(data: dict, key: str) -> tuple[float | None, float | None]:
    """Return (auroc, fpr95) from a results dict entry, or (None, None) if missing."""
    entry = data.get(key)
    if entry is None or not isinstance(entry, dict):
        return None, None
    return entry.get("auroc"), entry.get("fpr_at_tpr")


# ── spec: (display_name, dir_template, json_key) ─────────────────────────────
# dir_template uses {S} for seed, json_key is the key in results.json.
# split is appended automatically as _unknown_family / _unknown_ood.

SPLITS = ["unknown_family", "unknown_ood"]

SPEC = [
    # (label,              dir_template,        key_prefix,                            splits)
    # LGBM Binary
    ("LGBM-Bin / MSP",   "lgbm_binary_s{S}",  "ood_msp",                             SPLITS),
    ("LGBM-Bin / Energy","lgbm_binary_s{S}",  "ood_energy",                           SPLITS),
    # LGBM Multiclass
    ("LGBM-MC / MSP",    "lgbm_multi_s{S}",   "ood_msp",                             SPLITS),
    ("LGBM-MC / Energy", "lgbm_multi_s{S}",   "ood_energy",                           SPLITS),
    ("LGBM-MC / KNN",    "lgbm_multi_s{S}",   "ood_knn",                             SPLITS),
    # CNN (main file)
    ("CNN / MSP",        "cnn_s{S}",          "ood_msp",                             SPLITS),
    ("CNN / Mahal",      "cnn_s{S}",          "ood_mahalanobis",                     SPLITS),
    # CNN (extra_ood file)
    ("CNN / Energy",     "extra_ood_s{S}",    "cnn_energy",                          SPLITS),
    ("CNN / ODIN-e0",    "extra_ood_s{S}",    "cnn_odin_T1000_e0.0",                 SPLITS),
    ("CNN / KNN-k5",     "extra_ood_s{S}",    "cnn_knn_k5",                          SPLITS),
    # BiLSTM (main file)
    ("BiLSTM / MSP",     "bilstm_s{S}",       "ood_msp",                             SPLITS),
    ("BiLSTM / Energy",  "bilstm_s{S}",       "ood_energy",                          SPLITS),
    ("BiLSTM / Mahal",   "bilstm_s{S}",       "ood_mahalanobis",                     SPLITS),
    # BiLSTM (extra_ood file)
    ("BiLSTM / ODIN-e0", "extra_ood_s{S}",   "bilstm_odin_T1000_e0.0",              SPLITS),
    ("BiLSTM / KNN-k5",  "extra_ood_s{S}",   "bilstm_knn_k5",                       SPLITS),
    ("BiLSTM / ReAct-p90","extra_ood_s{S}",  "bilstm_react_p90",                    SPLITS),
    # BiLSTM-OE (main file; only unknown_family available)
    ("BiLSTM-OE / MSP",  "bilstm_oe_s{S}",   "ood_msp",                             ["unknown_family"]),
    ("BiLSTM-OE / Energy","bilstm_oe_s{S}",  "ood_energy",                          ["unknown_family"]),
    ("BiLSTM-OE / KNN",  "bilstm_oe_s{S}",   "ood_knn",                             ["unknown_family"]),
]


def collect(seeds: list[int], base: Path) -> dict:
    """Return nested dict: results[label][split] = {"auroc": [...], "fpr95": [...]}."""
    results: dict = {}

    for label, dir_tpl, key_prefix, splits in SPEC:
        results.setdefault(label, {})
        for split in splits:
            results[label].setdefault(split, {"auroc": [], "fpr95": []})

    loaded: dict[tuple, dict] = {}  # cache (dir_name) → json data

    for S in seeds:
        for label, dir_tpl, key_prefix, splits in SPEC:
            dir_name = dir_tpl.format(S=S)
            path = base / dir_name / "results.json"
            if dir_name not in loaded:
                if path.exists():
                    with open(path) as f:
                        loaded[dir_name] = json.load(f)
                else:
                    loaded[dir_name] = {}
            data = loaded[dir_name]

            for split in splits:
                key = f"{key_prefix}_{split}"
                auroc, fpr95 = get_metrics(data, key)
                if auroc is not None:
                    results[label][split]["auroc"].append(auroc)
                if fpr95 is not None:
                    results[label][split]["fpr95"].append(fpr95)

    return results


def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    m = np.mean(vals)
    s = np.std(vals, ddof=0)
    return f"{m:.4f} ± {s:.4f}"


def fmt_fpr(vals: list[float]) -> str:
    if not vals:
        return "—"
    m = np.mean(vals)
    s = np.std(vals, ddof=0)
    return f"{m:.4f} ± {s:.4f}"


def print_table(results: dict, seeds: list[int]):
    col_w = 22
    hdr = f"{'Method':<{col_w}}  {'AUROC (unk-fam)':<20}  {'AUROC (unk-ood)':<20}  {'FPR95 (unk-fam)':<20}  {'FPR95 (unk-ood)':<20}"
    sep = "-" * len(hdr)
    print(f"\nMulti-seed results  (seeds={seeds})")
    print(sep)
    print(hdr)
    print(sep)
    for label, by_split in results.items():
        fam = by_split.get("unknown_family", {})
        ood = by_split.get("unknown_ood", {})
        au_fam = fmt(fam.get("auroc", []))
        au_ood = fmt(ood.get("auroc", []))
        fp_fam = fmt_fpr(fam.get("fpr95", []))
        fp_ood = fmt_fpr(ood.get("fpr95", []))
        print(f"{label:<{col_w}}  {au_fam:<20}  {au_ood:<20}  {fp_fam:<20}  {fp_ood:<20}")
    print(sep)


def print_latex(results: dict, seeds: list[int]):
    """Print LaTeX table rows (copy into paper)."""
    print(f"\n% LaTeX rows  (seeds={seeds})\n")
    for label, by_split in results.items():
        fam = by_split.get("unknown_family", {})
        ood = by_split.get("unknown_ood", {})
        au_fam_v = fam.get("auroc", [])
        au_ood_v = ood.get("auroc", [])
        fp_fam_v = fam.get("fpr95", [])
        fp_ood_v = ood.get("fpr95", [])

        def ltx(vals):
            if not vals:
                return "--"
            m = np.mean(vals)
            s = np.std(vals, ddof=0)
            return f"${m:.4f}\\pm{s:.4f}$"

        print(f"{label} & {ltx(au_fam_v)} & {ltx(fp_fam_v)} & {ltx(au_ood_v)} & {ltx(fp_ood_v)} \\\\")


def save_json(results: dict, seeds: list[int], base: Path):
    out = {}
    for label, by_split in results.items():
        out[label] = {}
        for split, metrics in by_split.items():
            auroc_v = metrics.get("auroc", [])
            fpr95_v = metrics.get("fpr95", [])
            out[label][split] = {
                "auroc_mean": float(np.mean(auroc_v)) if auroc_v else None,
                "auroc_std":  float(np.std(auroc_v, ddof=0)) if auroc_v else None,
                "auroc_vals": auroc_v,
                "fpr95_mean": float(np.mean(fpr95_v)) if fpr95_v else None,
                "fpr95_std":  float(np.std(fpr95_v, ddof=0)) if fpr95_v else None,
                "fpr95_vals": fpr95_v,
            }
    out_path = base / "multi_seed_summary.json"
    with open(out_path, "w") as f:
        json.dump({"seeds": seeds, "results": out}, f, indent=2)
    print(f"\nSaved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 7, 99, 314])
    ap.add_argument("--base_dir", default="baseline_out")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--save", action="store_true", help="Save summary JSON to baseline_out/")
    args = ap.parse_args()

    base = Path(args.base_dir)
    results = collect(args.seeds, base)
    print_table(results, args.seeds)
    if args.latex:
        print_latex(results, args.seeds)
    if args.save:
        save_json(results, args.seeds, base)


if __name__ == "__main__":
    main()
