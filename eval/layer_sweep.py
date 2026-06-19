"""Per-layer SAE+probe sweep -> Delta(SAE-Raw) vs depth curve.

Consumes extract_layersweep.py's (N, n_layers, d_model) tensor: for each layer
slice the boundary activation, train a TopK-32 SAE, probe raw vs SAE, and record
Delta(SAE-Raw). Plots the curve so the null is shown across depth, not at two
isolated layers. Resumable per layer.

Usage:
    python eval/layer_sweep.py                          # all layers
    python eval/layer_sweep.py --layers 0 4 8 12 16 20  # subset
"""
import argparse
import json
import os
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERACT = "activations_layersweep/squad_layeract.safetensors"
META = "activations_layersweep/squad_metadata.parquet"


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:] + p.stderr[-3000:])
        raise RuntimeError(f"failed: {' '.join(cmd)}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="+", type=int, default=None, help="default: all layers")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work_dir", default="eval/results/layersweep/_work")
    ap.add_argument("--out_json", default="eval/results/layersweep/layersweep_summary.json")
    args = ap.parse_args()

    full = load_file(os.path.join(ROOT, LAYERACT))["encoder_embeddings"]  # (N, n_layers, d_model)
    n_layers = full.shape[1]
    layers = args.layers if args.layers is not None else list(range(n_layers))
    work = os.path.join(ROOT, args.work_dir)
    os.makedirs(work, exist_ok=True)
    out_path = os.path.join(ROOT, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = json.load(open(out_path)) if os.path.exists(out_path) else {"n_layers": n_layers, "cells": []}
    done = {c["layer"] for c in summary["cells"]}

    for L in layers:
        if L in done:
            continue
        print(f"=== layer {L}/{n_layers-1} ===", flush=True)
        sl = full[:, L:L + 1, :].contiguous()                 # (N, 1, d_model)
        acts_path = os.path.join(work, f"layer{L:02d}_acts.safetensors")
        save_file({"encoder_embeddings": sl}, acts_path)
        ckpt_dir = os.path.join(work, f"layer{L:02d}_sae")
        run([args.python, "sae/train_sae.py", "--activations", acts_path, "--metadata",
             os.path.join(ROOT, META), "--k", "32", "--output_dir", ckpt_dir])
        rj = os.path.join(work, f"layer{L:02d}_probe.json")
        run([args.python, "probing/probe.py", "--activations", acts_path,
             "--metadata", os.path.join(ROOT, META),
             "--sae_ckpt", os.path.join(ckpt_dir, "sae_topk_32.pt"),
             "--results_json", rj, "--scores_parquet", os.path.join(work, f"layer{L:02d}_scores.parquet")])
        r = json.load(open(rj))
        cell = {"layer": L, "P4_RawOnly_AUROC": r["P4_RawOnly_AUROC"],
                "P5_SAEOnly_AUROC": r["P5_SAEOnly_AUROC"],
                "delta_sae_over_raw": r["delta_sae_over_raw"],
                "delta_sae_over_raw_CI": [r["delta_sae_over_raw_CI_lower"], r["delta_sae_over_raw_CI_upper"]]}
        summary["cells"].append(cell)
        summary["cells"].sort(key=lambda c: c["layer"])
        json.dump(summary, open(out_path, "w"), indent=2)
        print(f"  L{L}: P4={cell['P4_RawOnly_AUROC']:.3f} P5={cell['P5_SAEOnly_AUROC']:.3f} "
              f"Δ(SAE-Raw)={cell['delta_sae_over_raw']:+.4f} CI={[round(x,3) for x in cell['delta_sae_over_raw_CI']]}", flush=True)

    sig_pos = [c for c in summary["cells"] if c["delta_sae_over_raw_CI"][0] > 0]
    summary["n_layers_swept"] = len(summary["cells"])
    summary["n_layers_sae_beats_raw_sig"] = len(sig_pos)
    json.dump(summary, open(out_path, "w"), indent=2)

    # Plot Delta(SAE-Raw) vs depth.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cells = sorted(summary["cells"], key=lambda c: c["layer"])
        xs = [c["layer"] for c in cells]
        ys = [c["delta_sae_over_raw"] for c in cells]
        lo = [c["delta_sae_over_raw"] - c["delta_sae_over_raw_CI"][0] for c in cells]
        hi = [c["delta_sae_over_raw_CI"][1] - c["delta_sae_over_raw"] for c in cells]
        plt.figure(figsize=(8, 4.5))
        plt.axhline(0, color="gray", lw=1, ls="--")
        plt.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3)
        plt.xlabel("layer index"); plt.ylabel("Δ(SAE − Raw) AUROC")
        plt.title("SAE vs Raw difficulty signal across depth (SQuAD, Pythia-410M)")
        plt.tight_layout()
        fig_path = out_path.replace(".json", ".png")
        plt.savefig(fig_path, dpi=130)
        print(f"Wrote plot {fig_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nLAYER SWEEP DONE: {len(summary['cells'])} layers, "
          f"{len(sig_pos)} where SAE significantly beats raw.\nWrote {out_path}")


if __name__ == "__main__":
    main()
