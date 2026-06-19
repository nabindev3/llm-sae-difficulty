"""Feature-steering: clamp top difficulty features, measure output shift.

Is the SAE's causal signal *actionable*? We take the most difficulty-aligned SAE
features (highest univariate AUROC vs the difficulty label), clamp each at the
Layer-12 boundary token -- to 0 (ablate) and to a high value (amplify) -- and
measure the shift in the model's cross-entropy on the gold answer vs the
unclamped SAE reconstruction. A large, reliable shift => steering these features
changes behavior (actionable); a shift indistinguishable from 0 => not.

SQuAD, Pythia-410M, boundary SAE (sae/checkpoints/sae_topk_32.pt). Subsamples the
test split for speed; reports mean ΔCE with a bootstrap CI per feature/direction.

Usage:
    python eval/steering.py --n_eval 400 --top_k 3 \
        --out_json eval/results/steering/squad.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae.sae_model import TopKSAE


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--activations", default="activations/squad_activations.safetensors")
    ap.add_argument("--metadata", default="activations/squad_metadata.parquet")
    ap.add_argument("--layer_idx", type=int, default=11)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()

    state = torch.load(args.sae_ckpt, map_location=device, weights_only=True)
    d_model, d_hidden = state["W_enc"].shape
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32).to(device)
    sae.load_state_dict(state); sae.eval()

    # Rank features by univariate difficulty AUROC; clamp value = 95th pct of active.
    raw = load_file(args.activations)["encoder_embeddings"]      # (N, 1, d)
    meta = pd.read_parquet(args.metadata)
    y = meta["difficulty"].values
    te = (meta["split"] == "test").values
    with torch.no_grad():
        codes, _, _ = sae(raw[:, 0, :].to(device).float())
    codes = codes.cpu().numpy()
    aurocs = []
    for f in range(d_hidden):
        c = codes[:, f]
        if np.ptp(c[te]) > 0 and len(np.unique(y[te])) > 1:
            a = roc_auc_score(y[te], c[te]); aurocs.append((max(a, 1 - a), f))
    aurocs.sort(reverse=True)
    top_feats = [f for _, f in aurocs[:args.top_k]]
    clamp_hi = {f: float(np.percentile(codes[codes[:, f] > 0, f], 95)) if (codes[:, f] > 0).any() else 1.0
                for f in top_feats}
    print("top difficulty features (feat, AUROC):", [(f, round(a, 3)) for a, f in aurocs[:args.top_k]])

    # Patch hook at the boundary token: reconstruct with feature f set to clamp_val.
    clamp_feat = {"f": None, "val": None, "plen": None}

    def hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        pos = clamp_feat["plen"] - 1
        if pos < 0 or pos >= hs.shape[1] or clamp_feat["f"] is None:
            return out
        x = hs[:, pos, :].float()
        with torch.no_grad():
            acts, recon, _ = sae(x)
            if clamp_feat["val"] is not None:                   # steer; else plain recon
                acts = acts.clone(); acts[:, clamp_feat["f"]] = clamp_feat["val"]
                recon = acts @ sae.W_dec + sae.b_dec
        patched = hs.clone()
        patched[:, pos, :] = recon.to(hs.dtype)
        return (patched,) + out[1:] if isinstance(out, tuple) else patched

    h = model.gpt_neox.layers[args.layer_idx].register_forward_hook(hook)

    test_rows = meta[te].head(args.n_eval).reset_index(drop=True)

    def gold_ce(prompt, gold):
        pids = tok.encode(prompt, add_special_tokens=True)
        if len(pids) > 200:
            pids = pids[-200:]
        plen = len(pids)
        tids = tok.encode(" " + gold.strip(), add_special_tokens=False)
        full = torch.tensor([pids + tids], device=device)
        clamp_feat["plen"] = plen
        with torch.no_grad():
            logits = model(full).logits
        return F.cross_entropy(logits[0, plen - 1:-1, :], full[0, plen:]).item()

    results = {}
    for f in top_feats:
        d_ablate, d_amp = [], []
        for _, r in test_rows.iterrows():
            prompt, gold = r["prompt"], r["gold_answer"]
            clamp_feat["f"] = f
            clamp_feat["val"] = None
            ce_recon = gold_ce(prompt, gold)            # plain SAE recon (baseline)
            clamp_feat["val"] = 0.0
            ce_ablate = gold_ce(prompt, gold)           # feature removed
            clamp_feat["val"] = clamp_hi[f]
            ce_amp = gold_ce(prompt, gold)              # feature amplified
            d_ablate.append(ce_ablate - ce_recon)
            d_amp.append(ce_amp - ce_recon)
        results[str(f)] = {
            "clamp_high_value": clamp_hi[f],
            "ablate_mean_dCE": float(np.mean(d_ablate)), "ablate_CI": boot_ci(d_ablate),
            "amplify_mean_dCE": float(np.mean(d_amp)), "amplify_CI": boot_ci(d_amp),
        }
    h.remove()

    out = {"n_eval": len(test_rows), "top_features": top_feats, "by_feature": results,
           "max_abs_shift_nats": max(max(abs(v["ablate_mean_dCE"]), abs(v["amplify_mean_dCE"]))
                                     for v in results.values())}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"\nsteering output shift on gold answer (ΔCE nats vs SAE recon), n={len(test_rows)}:")
    for f, v in results.items():
        print(f"  feat {f:>4s}: ablate {v['ablate_mean_dCE']:+.4f} {[round(x,3) for x in v['ablate_CI']]}  "
              f"amplify {v['amplify_mean_dCE']:+.4f} {[round(x,3) for x in v['amplify_CI']]}")
    print(f"max |shift| = {out['max_abs_shift_nats']:.4f} nats")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
