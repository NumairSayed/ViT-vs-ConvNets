"""
main.py — DLCV 2026 Assignment 2: Run all experiments
=======================================================
Usage:
    python main.py              # run all 7 experiments
    python main.py --exp 1 3 5  # run specific experiments

Results are saved under results/exp{N}_*/  and  saved_models/
"""

import argparse
import json
import time
from pathlib import Path

import torch

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
Path("saved_models").mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print(f"  DLCV 2026 — Assignment 2")
print(f"  Device : {DEVICE}" +
      (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))
print(f"{'='*60}\n")

#  Experiment registry ─
# Import lazily so individual modules can also be run standalone
def _get_experiments():
    from exp1_data_efficiency import run as run1
    from exp2_patch_size       import run as run2
    from exp3_cls_vs_mean      import run as run3
    from exp4_pos_encoding     import run as run4
    from exp5_attention        import run as run5
    from exp6_overlap          import run as run6
    from exp7_linear_probe     import run as run7
    return {
        1: ("Data Efficiency (ViT vs CNN)",           run1),
        2: ("Effect of Patch Size",                   run2),
        3: ("CLS Token vs Mean Pooling",              run3),
        4: ("Positional Encoding Ablation",           run4),
        5: ("Attention Maps & Entropy Analysis",      run5),
        6: ("Overlapping vs Non-overlapping Patches", run6),
        7: ("Layer-wise Linear Probing",              run7),
    }


#  Utilities ─

def save_summary(all_results: dict):
    """Flatten numeric leaf values into a single summary JSON."""
    import numpy as np

    def cvt(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    summary = {}
    for exp_key, res in all_results.items():
        if not isinstance(res, dict):
            continue
        for k, v in res.items():
            if isinstance(v, (int, float)):
                summary[f"{exp_key}/{k}"] = round(float(v), 5)
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                summary[f"{exp_key}/{k}"] = [round(float(x), 5) for x in v]

    path = RESULTS_DIR / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=cvt)
    print(f"\n  Summary saved → {path}")


#  Argument parsing 

def parse_args():
    parser = argparse.ArgumentParser(description="DLCV 2026 Assignment 2")
    parser.add_argument(
        "--exp", type=int, nargs="+", default=list(range(1, 8)),
        help="Experiment IDs to run (default: 1 2 3 4 5 6 7)")
    return parser.parse_args()


#  Main 

def main():
    args        = parse_args()
    experiments = _get_experiments()
    all_results = {}
    total_t0    = time.time()

    for eid in sorted(args.exp):
        if eid not in experiments:
            print(f"[WARN] Unknown experiment ID {eid} — skipping.")
            continue

        name, run_fn = experiments[eid]
        print(f"\n{'#'*60}")
        print(f"  Starting Experiment {eid}: {name}")
        print(f"{'#'*60}")

        t0  = time.time()
        res = run_fn()
        elapsed = time.time() - t0

        all_results[f"exp{eid}"] = res
        print(f"\n  ✓ Experiment {eid} completed in {elapsed/60:.1f} min")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  All done in {total_elapsed/60:.1f} min")
    print(f"{'='*60}")

    save_summary(all_results)


if __name__ == "__main__":
    main()