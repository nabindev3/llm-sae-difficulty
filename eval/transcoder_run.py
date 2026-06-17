"""Transcoder arm of the SAE design-space sweep (Step 2, variant: transcoder).

A transcoder decomposes the *MLP computation* (mlp-in -> mlp-out) rather than
autoencoding a single site. We ask the same question as the rest of the sweep:
do transcoder feature codes add difficulty signal over the raw residual stream?

For each (expansion, k):
  1. train a TopK transcoder on the paired MLP activations (train_variants.py),
  2. encode its codes from mlp-in,
  3. probe codes vs the residual-stream raw baseline (probe.py --sae_codes),
     i.e. Delta(SAE-Raw) is measured against the SAME resid raw used everywhere.

Run after extract_mlp.py. Light (small boundary tensors). Example:
    python eval/transcoder_run.py --expansions 4 8 --ks 16 32 64
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MLP_IN = "activations_mlp/squad_mlp_in.safetensors"
MLP_OUT = "activations_mlp/squad_mlp_out.safetensors"
MLP_META = "activations_mlp/squad_metadata.parquet"
# Residual-stream raw baseline (the same site used across the headline + sweep).
RESID_ACTS = "activations/squad_activations.safetensors"
RESID_META = "activations/squad_metadata.parquet"


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        raise RuntimeError(f"failed: {' '.join(cmd)}")
    return p.stdout


def encode_codes(py, ckpt, out_path):
    """Encode transcoder codes from mlp-in -> (N,1,d_hidden) safetensors."""
    snippet = (
        "import torch;from safetensors.torch import load_file,save_file;"
        "from sae.load_any import load_sae_any;"
        f"m,dh=load_sae_any({ckpt!r},32,'cpu');"
        f"x=load_file({MLP_IN!r})['encoder_embeddings'];N,S,d=x.shape;"
        "import torch as T;\n"
        "c=m.encode(x.reshape(-1,d).float()).detach();"
        f"save_file({{'encoder_embeddings':c.reshape(N,S,dh).to(torch.float16)}},{out_path!r});"
        "print('codes',tuple(c.shape))"
    )
    run([py, "-c", snippet])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expansions", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work_dir", default="eval/results/sweep/_work_transcoder")
    ap.add_argument("--out_json", default="eval/results/sweep/transcoder_summary.json")
    args = ap.parse_args()

    work = os.path.join(ROOT, args.work_dir)
    os.makedirs(work, exist_ok=True)
    d_model = 1024
    cells = []

    for exp in args.expansions:
        for k in args.ks:
            tag = f"transcoder_x{exp}_k{k}"
            ckpt_dir = os.path.join(work, tag)
            print(f"=== {tag} ===", flush=True)
            log = run([args.python, "sae/train_variants.py", "--variant", "transcoder",
                       "--activations", MLP_IN, "--target_activations", MLP_OUT,
                       "--metadata", MLP_META, "--d_hidden", str(exp * d_model),
                       "--k", str(k), "--output_dir", ckpt_dir])
            ckpt = os.path.join(ckpt_dir, "sae_transcoder.pt")
            codes = os.path.join(work, f"{tag}_codes.safetensors")
            encode_codes(args.python, ckpt, codes)

            rj = os.path.join(work, f"{tag}_probe.json")
            run([args.python, "probing/probe.py",
                 "--activations", RESID_ACTS, "--metadata", RESID_META,
                 "--sae_codes", codes, "--results_json", rj,
                 "--scores_parquet", os.path.join(work, f"{tag}_scores.parquet")])
            r = json.load(open(rj))
            cell = {"tag": tag, "expansion": exp, "k": k,
                    "P4_RawOnly_AUROC": r["P4_RawOnly_AUROC"],
                    "P5_SAEOnly_AUROC": r["P5_SAEOnly_AUROC"],
                    "delta_sae_over_raw": r["delta_sae_over_raw"],
                    "delta_sae_over_raw_CI": [r["delta_sae_over_raw_CI_lower"],
                                              r["delta_sae_over_raw_CI_upper"]]}
            cells.append(cell)
            print(f"  -> {tag}  P5={cell['P5_SAEOnly_AUROC']:.3f} "
                  f"P4={cell['P4_RawOnly_AUROC']:.3f} "
                  f"Δ(SAE-Raw)={cell['delta_sae_over_raw']:+.4f} "
                  f"CI={cell['delta_sae_over_raw_CI']}", flush=True)

    pos_sig = [c for c in cells if c["delta_sae_over_raw_CI"][0] > 0]
    summary = {"site": "L12 MLP (in->out) transcoder vs resid-raw",
               "n_cells": len(cells), "n_positive_delta_sig": len(pos_sig),
               "max_delta_sae_over_raw": max(c["delta_sae_over_raw"] for c in cells),
               "cells": cells}
    out = os.path.join(ROOT, args.out_json)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nTRANSCODER DONE: {len(cells)} cells, "
          f"{len(pos_sig)} significant-positive, "
          f"max Δ(SAE-Raw)={summary['max_delta_sae_over_raw']:+.4f}\nWrote {out}")


if __name__ == "__main__":
    main()
