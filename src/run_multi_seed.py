"""
run_multi_seed.py – Run all training scripts across N random seeds for reproducibility.

Usage (from project root):
  python src/run_multi_seed.py
  python src/run_multi_seed.py --seeds 0 1 2 3 4 --device cpu
  python src/run_multi_seed.py --seeds 0 1 --skip_extra_ood

Each seed produces one set of output directories under baseline_out/:
  lgbm_binary_s{S}/   lgbm_multi_s{S}/   cnn_s{S}/
  bilstm_s{S}/        bilstm_oe_s{S}/    extra_ood_s{S}/

After all seeds complete, run:
  python src/aggregate_seeds.py
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], label: str) -> int:
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  -> {label}: {status}  [{elapsed/60:.1f} min]")
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99, 314])
    ap.add_argument("--run_dir", default="data/processed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_extra_ood", action="store_true",
                    help="Skip eval_extra_ood.py (ODIN/KNN/ReAct for CNN+BiLSTM)")
    ap.add_argument("--only", nargs="+",
                    choices=["lgbm_binary", "lgbm_multi", "cnn", "bilstm", "bilstm_oe", "extra_ood"],
                    help="Run only these models (default: all)")
    args = ap.parse_args()

    py = sys.executable
    seeds = args.seeds
    run_dir = args.run_dir
    device = args.device
    run_all = args.only is None
    only = set(args.only or [])

    total_t0 = time.time()
    failures = []
    base = Path("baseline_out")

    def should_skip(dir_name: str) -> bool:
        p = base / dir_name / "results.json"
        if p.exists():
            print(f"  [skip] {dir_name}/results.json already exists")
            return True
        return False

    for S in seeds:
        print(f"\n{'#'*70}")
        print(f"  SEED {S}")
        print(f"{'#'*70}")

        if run_all or "lgbm_binary" in only:
            if not should_skip(f"lgbm_binary_s{S}"):
                rc = run(
                    [py, "src/train_baseline.py",
                     "--run_dir", run_dir,
                     "--out_dir", f"baseline_out/lgbm_binary_s{S}",
                     "--seed", str(S)],
                    f"LGBM Binary  seed={S}",
                )
                if rc != 0:
                    failures.append(f"lgbm_binary_s{S}")

        if run_all or "lgbm_multi" in only:
            if not should_skip(f"lgbm_multi_s{S}"):
                rc = run(
                    [py, "src/train_multiclass.py",
                     "--run_dir", run_dir,
                     "--out_dir", f"baseline_out/lgbm_multi_s{S}",
                     "--seed", str(S)],
                    f"LGBM Multiclass  seed={S}",
                )
                if rc != 0:
                    failures.append(f"lgbm_multi_s{S}")

        if run_all or "cnn" in only:
            if not should_skip(f"cnn_s{S}"):
                rc = run(
                    [py, "src/train_neural.py",
                     "--run_dir", run_dir,
                     "--out_dir", f"baseline_out/cnn_s{S}",
                     "--device", device,
                     "--seed", str(S)],
                    f"CNN  seed={S}",
                )
                if rc != 0:
                    failures.append(f"cnn_s{S}")

        if run_all or "bilstm" in only:
            if not should_skip(f"bilstm_s{S}"):
                rc = run(
                    [py, "src/train_bilstm.py",
                     "--run_dir", run_dir,
                     "--out_dir", f"baseline_out/bilstm_s{S}",
                     "--device", device,
                     "--seed", str(S)],
                    f"BiLSTM  seed={S}",
                )
                if rc != 0:
                    failures.append(f"bilstm_s{S}")

        if run_all or "bilstm_oe" in only:
            if not should_skip(f"bilstm_oe_s{S}"):
                rc = run(
                    [py, "src/train_bilstm_oe.py",
                     "--run_dir", run_dir,
                     "--out_dir", f"baseline_out/bilstm_oe_s{S}",
                     "--device", device,
                     "--seed", str(S)],
                    f"BiLSTM-OE  seed={S}",
                )
                if rc != 0:
                    failures.append(f"bilstm_oe_s{S}")

        if (run_all or "extra_ood" in only) and not args.skip_extra_ood:
            cnn_dir = f"baseline_out/cnn_s{S}"
            bilstm_dir = f"baseline_out/bilstm_s{S}"
            if not should_skip(f"extra_ood_s{S}"):
                if Path(cnn_dir).exists() and Path(bilstm_dir).exists():
                    rc = run(
                        [py, "src/eval_extra_ood.py",
                         "--run_dir", run_dir,
                         "--cnn_dir", cnn_dir,
                         "--bilstm_dir", bilstm_dir,
                         "--out_dir", f"baseline_out/extra_ood_s{S}",
                         "--device", device],
                        f"Extra OOD (ODIN/KNN/ReAct)  seed={S}",
                    )
                    if rc != 0:
                        failures.append(f"extra_ood_s{S}")
                else:
                    print(f"  Skipping extra_ood_s{S}: CNN or BiLSTM dir missing")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  ALL SEEDS DONE  [{total_elapsed/60:.1f} min total]")
    if failures:
        print(f"  FAILURES: {', '.join(failures)}")
    else:
        print("  No failures.")
    print(f"{'='*70}")
    print("\nNext step:")
    print("  python src/aggregate_seeds.py")


if __name__ == "__main__":
    main()
