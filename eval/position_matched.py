"""Position-matched fidelity-vs-coverage disentanglement (Limitation 4).

The committed 2x2 disentanglement compared a boundary SAE and an all-position SAE,
but at DIFFERENT boundary tokens (first_ending vs last_prompt) -- not strictly
position-controlled. This fixes that:

  1. Slice the boundary token (seq_len-1, the last PROMPT token) out of the
     all-position activations -> train a boundary SAE on exactly that position.
  2. Evaluate BOTH SAEs at the SAME position via causal_ablation --coverage 1
     --hellaswag_boundary last_prompt, plus the all-position SAE at full coverage.

Three runs give a clean decomposition (all at prompt_len-1 for the K=1 rows):
  A boundary-SAE  @ K=1     recon penalty   (low fidelity,  matched position)
  B allpos-SAE    @ K=1     recon penalty   (high fidelity, matched position)
  C allpos-SAE    @ K=all   recon penalty   (high fidelity, full coverage)
  fidelity effect  = B - A   (same coverage & position -> isolates SAE fidelity)
  coverage effect  = C - B   (same SAE             -> isolates coverage)

Usage:
    python eval/position_matched.py
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLPOS_ACTS = "activations_allpos/hellaswag_activations.safetensors"
ALLPOS_META = "activations_allpos/hellaswag_metadata.parquet"
ALLPOS_SAE = "sae/checkpoints_allpos_hellaswag/sae_topk_32.pt"
COVERAGE_ALL = 256   # >= any hellaswag prompt length -> patches the whole prompt


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:] + p.stderr[-3000:])
        raise RuntimeError(f"failed: {' '.join(cmd)}")
    return p.stdout


def slice_boundary(work):
    """Slice the last prompt token (seq_len-1) from the all-position tensor."""
    out = os.path.join(work, "boundary_acts.safetensors")
    if os.path.exists(out):
        return out
    acts = load_file(os.path.join(ROOT, ALLPOS_ACTS))["encoder_embeddings"]  # (N, max_seq, d)
    meta = pd.read_parquet(os.path.join(ROOT, ALLPOS_META))
    assert len(meta) == acts.shape[0], "metadata/activation mismatch"
    rows = []
    for i, sl in enumerate(meta["seq_len"].astype(int).values):
        idx = min(max(sl - 1, 0), acts.shape[1] - 1)
        rows.append(acts[i, idx, :].unsqueeze(0).unsqueeze(0))
    b = torch.cat(rows, dim=0)   # (N, 1, d)
    save_file({"encoder_embeddings": b}, out)
    print(f"sliced boundary activations {tuple(b.shape)} -> {out}")
    return out


def causal(py, sae_ckpt, acts, coverage, out_dir):
    rj = os.path.join(ROOT, out_dir, "hellaswag_causal_ablation.json")
    if not os.path.exists(rj):
        run([py, "eval/causal_ablation.py", "--dataset", "hellaswag", "--metric", "continuous",
             "--coverage", str(coverage), "--hellaswag_boundary", "last_prompt",
             "--sae_ckpt", sae_ckpt, "--activations", acts, "--metadata", ALLPOS_META,
             "--out_dir", out_dir])
    return json.load(open(rj))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--work_dir", default="eval/results/position_matched/_work")
    ap.add_argument("--out_json", default="eval/results/position_matched/summary.json")
    args = ap.parse_args()
    work = os.path.join(ROOT, args.work_dir)
    os.makedirs(work, exist_ok=True)

    # 1. boundary SAE trained on the matched position
    bnd_acts = slice_boundary(work)
    bnd_sae_dir = os.path.join(work, "boundary_sae")
    bnd_sae = os.path.join(bnd_sae_dir, "sae_topk_32.pt")
    if not os.path.exists(bnd_sae):
        print("training position-matched boundary SAE ...", flush=True)
        run([args.python, "sae/train_sae.py", "--activations", bnd_acts,
             "--metadata", ALLPOS_META, "--k", "32", "--output_dir", bnd_sae_dir])

    # 2. three position-controlled causal runs
    print("A: boundary-SAE @ K=1 ...", flush=True)
    A = causal(args.python, bnd_sae, bnd_acts, 1, "eval/results/position_matched/A_bnd_k1")
    print("B: allpos-SAE  @ K=1 ...", flush=True)
    B = causal(args.python, ALLPOS_SAE, ALLPOS_ACTS, 1, "eval/results/position_matched/B_allpos_k1")
    print("C: allpos-SAE  @ K=all ...", flush=True)
    C = causal(args.python, ALLPOS_SAE, ALLPOS_ACTS, COVERAGE_ALL, "eval/results/position_matched/C_allpos_all")

    def pen(r):
        return {"delta_recon": r["delta_recon_natural"],
                "ci": [r["delta_recon_natural_ci_lower"], r["delta_recon_natural_ci_upper"]]}

    out = {
        "A_boundarySAE_K1": pen(A), "B_allposSAE_K1": pen(B), "C_allposSAE_Kall": pen(C),
        "fidelity_effect_B_minus_A": B["delta_recon_natural"] - A["delta_recon_natural"],
        "coverage_effect_C_minus_B": C["delta_recon_natural"] - B["delta_recon_natural"],
        "note": "A,B both at position prompt_len-1 (matched) -> B-A isolates fidelity; "
                "B,C both allpos-SAE -> C-B isolates coverage.",
    }
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out_json)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, args.out_json), "w"), indent=2)
    print("\n=== POSITION-MATCHED DISENTANGLEMENT (HellaSwag, continuous nats) ===")
    print(f"  A boundary-SAE @K=1 : {out['A_boundarySAE_K1']['delta_recon']:+.4f} {out['A_boundarySAE_K1']['ci']}")
    print(f"  B allpos-SAE   @K=1 : {out['B_allposSAE_K1']['delta_recon']:+.4f} {out['B_allposSAE_K1']['ci']}")
    print(f"  C allpos-SAE   @Kall: {out['C_allposSAE_Kall']['delta_recon']:+.4f} {out['C_allposSAE_Kall']['ci']}")
    print(f"  fidelity effect (B-A, position-matched): {out['fidelity_effect_B_minus_A']:+.4f}")
    print(f"  coverage effect (C-B, SAE-matched)     : {out['coverage_effect_C_minus_B']:+.4f}")
    print(f"Wrote {os.path.join(ROOT, args.out_json)}")


if __name__ == "__main__":
    main()
