"""Train a non-TopK SAE variant (gated / jumprelu / transcoder).

Mirrors sae/train_sae.py's data handling (train-split filter + padding mask)
but drives training through each variant's ``compute_loss`` and writes a
self-describing checkpoint ``{variant, config, state_dict}`` that
sae.load_any.load_sae_any can reconstruct without guessing the architecture.

Examples:
    python sae/train_variants.py --variant gated --l1_coeff 1e-3 \
        --activations activations/squad_activations.safetensors \
        --metadata activations/squad_metadata.parquet \
        --d_hidden 4096 --output_dir sae/checkpoints_sweep/squad_l12_gated_x4

    python sae/train_variants.py --variant transcoder --k 32 \
        --activations activations_mlp/squad_mlp_in.safetensors \
        --target_activations activations_mlp/squad_mlp_out.safetensors \
        --metadata activations_mlp/squad_metadata.parquet \
        --output_dir sae/checkpoints_sweep/squad_l12_transcoder_x4
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sae.sae_variants import build_variant


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokens(activations_path, metadata_path, split_filter="train"):
    """Load activations, filter to a split, drop padding -> (n_tokens, d)."""
    acts = load_file(activations_path)["encoder_embeddings"]
    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_parquet(metadata_path)
        assert len(meta) == acts.shape[0], "metadata/activation row mismatch"
        keep = (meta["split"] == split_filter).values
        acts = acts[keep]
        meta = meta[keep].reset_index(drop=True)
        valid = [acts[i, :int(r["seq_len"]), :] for i, r in meta.iterrows()]
        acts = torch.cat(valid, dim=0)
    elif acts.dim() == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    return acts.to(torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=["gated", "jumprelu", "transcoder"])
    ap.add_argument("--activations", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--target_activations", default=None,
                    help="Target site tensor for the transcoder (e.g. MLP output)")
    ap.add_argument("--d_hidden", type=int, default=None, help="default 4x d_model")
    ap.add_argument("--k", type=int, default=32, help="transcoder TopK")
    ap.add_argument("--l1_coeff", type=float, default=1e-3, help="gated sparsity coeff")
    ap.add_argument("--l0_coeff", type=float, default=1e-2, help="jumprelu L0 coeff")
    ap.add_argument("--bandwidth", type=float, default=1e-3, help="jumprelu STE bandwidth")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--split_filter", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(args.seed)

    print(f"Loading inputs from {args.activations} ...")
    X = load_tokens(args.activations, args.metadata, args.split_filter)
    d_model = X.shape[-1]
    d_hidden = args.d_hidden or 4 * d_model
    print(f"Tokens: {tuple(X.shape)}  d_model={d_model}  d_hidden={d_hidden}")
    in_var = X.var(dim=0).mean().item()

    target = None
    if args.variant == "transcoder":
        if not args.target_activations:
            raise SystemExit("--target_activations is required for the transcoder")
        target = load_tokens(args.target_activations, args.metadata, args.split_filter)
        assert target.shape[0] == X.shape[0], "input/target token count mismatch"
        tgt_var = target.var(dim=0).mean().item()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    kw = dict(d_model=d_model, d_hidden=d_hidden)
    if args.variant == "gated":
        kw["l1_coeff"] = args.l1_coeff
    elif args.variant == "jumprelu":
        kw.update(l0_coeff=args.l0_coeff, bandwidth=args.bandwidth)
    elif args.variant == "transcoder":
        kw.update(k=args.k, d_out=target.shape[-1])
    model = build_variant(args.variant, **kw).to(device)

    # Initialize decoder bias to the (target) mean; transcoder also centers input.
    if args.variant == "transcoder":
        model.b_in.data = X.mean(dim=0).to(device)
        model.b_dec.data = target.mean(dim=0).to(device)
        norm_var = tgt_var
    else:
        model.b_dec.data = X.mean(dim=0).to(device)
        norm_var = in_var

    tensors = (X,) if target is None else (X, target)
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, float(s) / max(1, args.warmup_steps)))

    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        last = {}
        for batch in pbar:
            xb = batch[0].to(device)
            tb = batch[1].to(device) if target is not None else None
            opt.zero_grad()
            loss, met = model.compute_loss(xb, target=tb)
            loss.backward()
            opt.step()
            sched.step()
            model.normalize_decoder()
            tot += met["l_recon"]
            last = met
            pbar.set_postfix(nMSE=f"{met['l_recon']/(norm_var+1e-8):.3f}", L0=f"{met['l0']:.1f}")
        print(f"Epoch {epoch+1} | nMSE {tot/len(loader)/(norm_var+1e-8):.3f} | L0 {last.get('l0',0):.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"sae_{args.variant}.pt")
    torch.save({"variant": args.variant, "config": model.config(),
                "state_dict": model.state_dict()}, save_path)
    print(f"Saved {args.variant} checkpoint to {save_path}")


if __name__ == "__main__":
    main()
