"""Direct-logit-attribution on the top causal features: why causal != predictive.

The novel tension: SAE features with measurable causal effect on the model's
output are nonetheless NOT predictive of difficulty. This quantifies it per
feature, for the top causal features (from the all-position causal ablation):

  * DLA  -- project the feature's decoder direction through the final LayerNorm
            gain and unembedding W_U -> which tokens it promotes, and a magnitude.
            Large DLA = the feature does real output work (causal).
  * predictive AUROC -- univariate ROC-AUC of that feature's boundary activation
            vs the difficulty label. ~0.5 = the feature does NOT track difficulty.

High DLA / causal effect with ~chance predictive AUROC is the dissociation:
the model uses these directions to compute, but their activation does not encode
the correctness signal a difficulty probe needs.

Usage:
    python eval/dla.py --dataset hellaswag \
        --sae_ckpt sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
        --activations activations_allpos/hellaswag_activations.safetensors \
        --metadata activations_allpos/hellaswag_metadata.parquet \
        --causal_json eval/results/allpos/hellaswag_causal_ablation.json \
        --out_json eval/results/dla/hellaswag.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae.sae_model import TopKSAE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--dataset", default="hellaswag")
    ap.add_argument("--sae_ckpt", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--causal_json", required=True)
    ap.add_argument("--top_tokens", type=int, default=8)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    W_U = model.embed_out.weight.detach()                       # (vocab, d_model)
    ln = model.gpt_neox.final_layer_norm
    ln_w = ln.weight.detach()                                   # (d_model,)

    state = torch.load(args.sae_ckpt, map_location=device, weights_only=True)
    d_model, d_hidden = state["W_enc"].shape
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32).to(device)
    sae.load_state_dict(state); sae.eval()

    top_feats = json.load(open(args.causal_json))["top_5_features"]
    causal_effects = json.load(open(args.causal_json)).get("feature_effects", {})

    # Per-sample boundary SAE codes (for predictive AUROC of each feature).
    raw = load_file(args.activations)["encoder_embeddings"]      # (N, max_seq, d)
    meta = pd.read_parquet(args.metadata)
    y = meta["difficulty"].values
    te = (meta["split"] == "test").values
    bnd = torch.stack([raw[i, int(s) - 1, :] for i, s in enumerate(meta["seq_len"].astype(int).values)]).to(device).float()
    with torch.no_grad():
        codes, _, _ = sae(bnd)                                  # (N, d_hidden)
    codes = codes.cpu().numpy()

    def dla_for(f):
        d = sae.W_dec[f].detach()                               # (d_model,) decoder direction
        # Fold final-LN gain and center (LayerNorm subtracts the mean), then unembed.
        d_eff = (d - d.mean()) * ln_w
        logits = (d_eff @ W_U.T).float().cpu().numpy()          # (vocab,)
        top = np.argsort(-logits)[:args.top_tokens]
        bot = np.argsort(logits)[:args.top_tokens]
        return {
            "dla_magnitude": float(logits.std()),
            "dla_max_logit": float(logits.max()),
            "promotes_tokens": [tok.decode([int(t)]).strip() for t in top],
            "suppresses_tokens": [tok.decode([int(t)]).strip() for t in bot],
        }

    rows = []
    for f in top_feats:
        col = codes[:, f]
        # Univariate predictive AUROC on test (guard against a constant column).
        if te.sum() > 0 and len(np.unique(y[te])) > 1 and np.ptp(col[te]) > 0:
            auroc = float(roc_auc_score(y[te], col[te]))
            auroc = max(auroc, 1 - auroc)                       # direction-free
        else:
            auroc = float("nan")
        d = dla_for(f)
        ce = causal_effects.get(str(f), {})
        rows.append({"feature": int(f), "predictive_AUROC": auroc,
                     "causal_delta_error": ce.get("delta_error"),
                     "causal_ci": [ce.get("ci_lower"), ce.get("ci_upper")], **d})

    out = {"dataset": args.dataset, "n_test": int(te.sum()), "features": rows,
           "mean_predictive_AUROC": float(np.nanmean([r["predictive_AUROC"] for r in rows]))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"{'feat':>5s} {'predAUROC':>9s} {'causalΔ':>9s} {'DLAmag':>7s}  promotes")
    for r in rows:
        ce = r["causal_delta_error"]
        print(f"{r['feature']:5d} {r['predictive_AUROC']:9.3f} "
              f"{(ce if ce is not None else float('nan')):+9.4f} {r['dla_magnitude']:7.3f}  {r['promotes_tokens'][:5]}")
    print(f"\nmean predictive AUROC of top causal features: {out['mean_predictive_AUROC']:.3f} "
          f"(>=~0.5 + nonzero causal Δ => causal != predictive)")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
