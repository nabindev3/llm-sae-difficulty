"""SAE design-space sweep (Limitation 2: 'is the null a TopK-32 artifact?').

Trains a grid of SAEs and probes each, recording whether Δ(SAE−Raw) stays
negative across:
  * expansion factor      4x / 8x / 16x          (all variants)
  * TopK sparsity k        16 / 32 / 64 / 128     (TopK only)
  * gated L1 coefficient   {grid}                 (gated only)
  * JumpReLU L0 coefficient{grid}                 (jumprelu only)

The transcoder variant is swept separately (eval/sae_sweep.py needs paired
MLP-in/MLP-out activations; see extract_mlp.py) and is not included here.

Each cell reuses the committed entrypoints (sae/train_sae.py for TopK,
sae/train_variants.py for gated/jumprelu, probing/probe.py for the probe) via
subprocess, so the sweep exercises exactly the same code as the headline run.
Resumable: a cell whose probe JSON already exists is skipped.

Usage:
    python eval/sae_sweep.py                       # full grid, both SQuAD layers
    python eval/sae_sweep.py --datasets squad_l12  # one layer
    python eval/sae_sweep.py --variants topk gated # subset of variants
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "squad_l12": {
        "activations": "activations/squad_activations.safetensors",
        "metadata": "activations/squad_metadata.parquet",
        "d_model": 1024,
    },
    "squad_l18": {
        "activations": "activations_late/squad_activations.safetensors",
        "metadata": "activations_late/squad_metadata.parquet",
        "d_model": 1024,
    },
    "hellaswag_l12": {
        "activations": "activations/hellaswag_activations.safetensors",
        "metadata": "activations/hellaswag_metadata.parquet",
        "d_model": 1024,
    },
}

EXPANSIONS = [4, 8, 16]
TOPK_K = [16, 32, 64, 128]
GATED_L1 = [3e-3, 1e-2, 3e-2, 1e-1]
JUMPRELU_L0 = [1e-3, 3e-3, 1e-2, 3e-2]

# Parse the trainer's final "... nMSE <x> ... L0 <y>" epoch line for diagnostics.
_NMSE = re.compile(r"nMSE[:\s]+([0-9.]+)")
_L0 = re.compile(r"L0[:\s]+([0-9.]+)")


def run(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError(f"failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def last_match(rx, text):
    m = rx.findall(text)
    return float(m[-1]) if m else None


def probe_cell(py, ds, cfg, sae_ckpt, k, results_json, work_dir, tag):
    run([py, "probing/probe.py",
         "--activations", cfg["activations"], "--metadata", cfg["metadata"],
         "--sae_ckpt", sae_ckpt, "--k", str(k),
         "--results_json", results_json,
         "--scores_parquet", os.path.join(work_dir, f"{tag}_scores.parquet")])
    with open(results_json) as f:
        r = json.load(f)
    return {
        "P4_RawOnly_AUROC": r["P4_RawOnly_AUROC"],
        "P5_SAEOnly_AUROC": r["P5_SAEOnly_AUROC"],
        "P3_InputStats_SAE_AUROC": r["P3_AUROC"],
        "delta_sae_over_raw": r["delta_sae_over_raw"],
        "delta_sae_over_raw_CI": [r["delta_sae_over_raw_CI_lower"], r["delta_sae_over_raw_CI_upper"]],
        "delta_sae": r["delta_sae"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["squad_l12", "squad_l18"],
                    choices=list(DATASETS))
    ap.add_argument("--variants", nargs="+", default=["topk", "gated", "jumprelu"],
                    choices=["topk", "gated", "jumprelu"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work_dir", default="eval/results/sweep/_work")
    ap.add_argument("--out_json", default="eval/results/sweep/sweep_summary.json")
    args = ap.parse_args()

    work = os.path.join(ROOT, args.work_dir)
    os.makedirs(work, exist_ok=True)
    out_path = os.path.join(ROOT, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Resume from any partial summary.
    summary = {"cells": []}
    if os.path.exists(out_path):
        summary = json.load(open(out_path))
    done = {(c["dataset"], c["tag"]) for c in summary["cells"]}

    def add_cell(cell):
        summary["cells"].append(cell)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        s = cell
        print(f"  -> {s['dataset']:13s} {s['tag']:22s} "
              f"P5={s['P5_SAEOnly_AUROC']:.3f} P4={s['P4_RawOnly_AUROC']:.3f} "
              f"Δ(SAE-Raw)={s['delta_sae_over_raw']:+.4f} "
              f"nMSE={s.get('nMSE')} L0={s.get('L0')}", flush=True)

    for ds in args.datasets:
        cfg = DATASETS[ds]
        dm = cfg["d_model"]
        print(f"\n=== {ds} ===", flush=True)

        for exp in EXPANSIONS:
            d_hidden = exp * dm

            if "topk" in args.variants:
                for k in TOPK_K:
                    tag = f"topk_x{exp}_k{k}"
                    if (ds, tag) in done:
                        continue
                    ckpt_dir = os.path.join(work, f"{ds}_{tag}")
                    log = run([args.python, "sae/train_sae.py",
                               "--activations", cfg["activations"], "--metadata", cfg["metadata"],
                               "--d_hidden", str(d_hidden), "--k", str(k),
                               "--output_dir", ckpt_dir])
                    sae_ckpt = os.path.join(ckpt_dir, f"sae_topk_{k}.pt")
                    rj = os.path.join(work, f"{ds}_{tag}_probe.json")
                    cell = {"dataset": ds, "variant": "topk", "tag": tag,
                            "expansion": exp, "k": k,
                            "nMSE": last_match(_NMSE, log), "L0": last_match(_L0, log)}
                    cell.update(probe_cell(args.python, ds, cfg, sae_ckpt, k, rj, work, f"{ds}_{tag}"))
                    add_cell(cell)

            if "gated" in args.variants:
                for l1 in GATED_L1:
                    tag = f"gated_x{exp}_l1{l1:g}"
                    if (ds, tag) in done:
                        continue
                    ckpt_dir = os.path.join(work, f"{ds}_{tag}")
                    log = run([args.python, "sae/train_variants.py", "--variant", "gated",
                               "--activations", cfg["activations"], "--metadata", cfg["metadata"],
                               "--d_hidden", str(d_hidden), "--l1_coeff", str(l1),
                               "--output_dir", ckpt_dir])
                    sae_ckpt = os.path.join(ckpt_dir, "sae_gated.pt")
                    rj = os.path.join(work, f"{ds}_{tag}_probe.json")
                    cell = {"dataset": ds, "variant": "gated", "tag": tag,
                            "expansion": exp, "l1_coeff": l1,
                            "nMSE": last_match(_NMSE, log), "L0": last_match(_L0, log)}
                    cell.update(probe_cell(args.python, ds, cfg, sae_ckpt, 32, rj, work, f"{ds}_{tag}"))
                    add_cell(cell)

            if "jumprelu" in args.variants:
                for l0 in JUMPRELU_L0:
                    tag = f"jumprelu_x{exp}_l0{l0:g}"
                    if (ds, tag) in done:
                        continue
                    ckpt_dir = os.path.join(work, f"{ds}_{tag}")
                    log = run([args.python, "sae/train_variants.py", "--variant", "jumprelu",
                               "--activations", cfg["activations"], "--metadata", cfg["metadata"],
                               "--d_hidden", str(d_hidden), "--l0_coeff", str(l0),
                               "--output_dir", ckpt_dir])
                    sae_ckpt = os.path.join(ckpt_dir, "sae_jumprelu.pt")
                    rj = os.path.join(work, f"{ds}_{tag}_probe.json")
                    cell = {"dataset": ds, "variant": "jumprelu", "tag": tag,
                            "expansion": exp, "l0_coeff": l0,
                            "nMSE": last_match(_NMSE, log), "L0": last_match(_L0, log)}
                    cell.update(probe_cell(args.python, ds, cfg, sae_ckpt, 32, rj, work, f"{ds}_{tag}"))
                    add_cell(cell)

    # Headline: does ANY cell make Δ(SAE−Raw) significantly positive?
    pos = [c for c in summary["cells"] if c["delta_sae_over_raw"] > 0]
    pos_sig = [c for c in pos if c["delta_sae_over_raw_CI"][0] > 0]
    summary["n_cells"] = len(summary["cells"])
    summary["n_positive_delta"] = len(pos)
    summary["n_positive_delta_sig"] = len(pos_sig)
    summary["max_delta_sae_over_raw"] = max(c["delta_sae_over_raw"] for c in summary["cells"])
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"SWEEP DONE: {summary['n_cells']} cells")
    print(f"  cells with Δ(SAE−Raw) > 0           : {summary['n_positive_delta']}")
    print(f"  cells with Δ(SAE−Raw) > 0 (CI>0)    : {summary['n_positive_delta_sig']}")
    print(f"  max Δ(SAE−Raw) over all cells       : {summary['max_delta_sae_over_raw']:+.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
