"""Multi-seed reproduction of the headline probe pipeline.

Reviewer objection (Limitation: single SAE seed underlies every number): the
committed AUROCs and Δ(SAE−Raw) come from one SAE training seed (42) and one
probe-fit seed (42). This script re-runs the *full* per-config pipeline — SAE
training AND probe fit — across several seeds, so we can report
training-induced variance, not just test-set bootstrap variance.

For each (config, seed) it:
  1. trains a fresh TopK-32 SAE with ``--seed S`` into a seed-specific dir,
  2. runs ``probing/probe.py --seed S`` against that SAE,
both via subprocess so the exact committed entrypoints are exercised. It then
aggregates the headline quantities (per-probe AUROC, Δ(Raw−Stats),
Δ(SAE−Raw)) across seeds as mean ± std with min/max.

Usage:
    python eval/multiseed.py --seeds 0 1 2 3 4
    python eval/multiseed.py --configs squad_l12 squad_l18 --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Repo root = parent of this file's directory (eval/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The three headline configs. Each retrains the SAE on the SAME activations the
# probe reads (self-consistent per layer/dataset), matching run_all.sh phase 2.
CONFIGS = {
    "hellaswag_l12": {
        "activations": "activations/hellaswag_activations.safetensors",
        "metadata": "activations/hellaswag_metadata.parquet",
    },
    "squad_l12": {
        "activations": "activations/squad_activations.safetensors",
        "metadata": "activations/squad_metadata.parquet",
    },
    "squad_l18": {
        "activations": "activations_late/squad_activations.safetensors",
        "metadata": "activations_late/squad_metadata.parquet",
    },
}

# Headline scalars pulled from each probe_results.json.
HEADLINE_KEYS = [
    "P1_AUROC", "P2_AUROC", "P3_AUROC", "P4_RawOnly_AUROC", "P5_SAEOnly_AUROC",
    "delta_raw", "delta_sae", "delta_sae_over_raw",
]


def run(cmd):
    """Run a subprocess from ROOT, streaming nothing; raise on failure."""
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def one_seed(py, cfg_name, cfg, seed, k, work_dir):
    """Train SAE + fit probes for one (config, seed). Returns the results dict."""
    ckpt_dir = os.path.join(work_dir, f"{cfg_name}_seed{seed}")
    results_json = os.path.join(work_dir, f"{cfg_name}_seed{seed}_probe.json")
    acts = cfg["activations"]
    meta = cfg["metadata"]

    print(f"  [seed {seed}] training SAE (k={k}) ...", flush=True)
    run([py, "sae/train_sae.py",
         "--activations", acts, "--metadata", meta,
         "--k", str(k), "--seed", str(seed),
         "--output_dir", ckpt_dir])

    sae_ckpt = os.path.join(ckpt_dir, f"sae_topk_{k}.pt")
    print(f"  [seed {seed}] fitting probes ...", flush=True)
    # scores_parquet is required by probe.py but unused here; send to work_dir.
    run([py, "probing/probe.py",
         "--activations", acts, "--metadata", meta,
         "--sae_ckpt", sae_ckpt, "--k", str(k), "--seed", str(seed),
         "--results_json", results_json,
         "--scores_parquet", os.path.join(work_dir, f"{cfg_name}_seed{seed}_scores.parquet")])

    with open(results_json) as f:
        return json.load(f)


def aggregate(per_seed):
    """per_seed: list of results dicts. Returns {key: {mean,std,min,max,n,values}}."""
    out = {}
    for key in HEADLINE_KEYS:
        vals = np.array([r[key] for r in per_seed], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
            "values": [float(v) for v in vals],
        }
    # chosen_C is a dict per seed; report the per-probe modal/listing.
    out["chosen_C_per_seed"] = [r.get("chosen_C", {}) for r in per_seed]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--python", default=sys.executable, help="Interpreter for subprocesses")
    ap.add_argument("--work_dir", default=None,
                    help="Where to write per-seed SAEs/results (default: a temp dir)")
    ap.add_argument("--out_json", default="eval/results/multiseed/multiseed_summary.json")
    args = ap.parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="multiseed_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"Work dir: {work_dir}")
    print(f"Configs: {args.configs}  Seeds: {args.seeds}  k={args.k}\n")

    summary = {"k": args.k, "seeds": args.seeds, "configs": {}}
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"=== {cfg_name} ===", flush=True)
        per_seed = [one_seed(args.python, cfg_name, cfg, s, args.k, work_dir)
                    for s in args.seeds]
        summary["configs"][cfg_name] = aggregate(per_seed)

    out_path = os.path.join(ROOT, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable console table.
    print("\n" + "=" * 72)
    print(f"MULTI-SEED SUMMARY  (n={len(args.seeds)} seeds: {args.seeds})")
    print("=" * 72)
    for cfg_name, agg in summary["configs"].items():
        print(f"\n{cfg_name}")
        for key in HEADLINE_KEYS:
            s = agg[key]
            print(f"  {key:24s} {s['mean']:+.4f} ± {s['std']:.4f}   "
                  f"[{s['min']:+.4f}, {s['max']:+.4f}]")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
