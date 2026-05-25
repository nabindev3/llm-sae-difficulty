import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_file
from sae_model import TopKSAE
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _resample_dead(model, optimizer, activations, dead_mask, device):
    """Hard-reset dead features. For each dead feature:
      - draw a random input token, center it against b_dec, normalize;
      - copy that direction into the encoder column and decoder row;
      - zero its bias and Adam moments so it re-trains from scratch.
    """
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return
    n = activations.shape[0]
    idx = torch.randint(0, n, (n_dead,))
    samples = activations[idx].to(device).float()
    samples = samples - model.b_dec
    samples = F.normalize(samples, dim=-1)

    model.W_enc.data[:, dead_mask] = samples.T
    model.b_enc.data[dead_mask] = 0.0
    model.W_dec.data[dead_mask] = samples

    for p in (model.W_enc, model.b_enc, model.W_dec):
        st = optimizer.state.get(p, None)
        if not st or "exp_avg" not in st:
            continue
        if p is model.W_enc:
            st["exp_avg"][:, dead_mask] = 0
            st["exp_avg_sq"][:, dead_mask] = 0
        elif p is model.b_enc:
            st["exp_avg"][dead_mask] = 0
            st["exp_avg_sq"][dead_mask] = 0
        elif p is model.W_dec:
            st["exp_avg"][dead_mask] = 0
            st["exp_avg_sq"][dead_mask] = 0


def train_sae(activations_path, metadata_path=None, split_filter="train", d_model=None, d_hidden=None, k=32, aux_k=512, batch_size=2048, lr=5e-4, warmup_steps=100, epochs=10, dead_after_steps=50, resample_every=0, output_dir="sae/checkpoints"):
    print(f"Loading activations from {activations_path}...")
    tensors = load_file(activations_path)
    activations = tensors["encoder_embeddings"] # Shape: (num_prompts, max_seq_len, d_model)
    print(f"Loaded activations with shape: {activations.shape}")

    detected_d_model = int(activations.shape[-1])
    if d_model is None:
        d_model = detected_d_model
        print(f"Auto-detected d_model = {d_model}")
    elif d_model != detected_d_model:
        raise ValueError(f"--d_model={d_model} but activations have last dim {detected_d_model}")
    
    if d_hidden is None:
        d_hidden = 4 * d_model  # 4x expansion for LLM (4096 hidden size for 1024 d_model)
        print(f"Auto-set d_hidden = {d_hidden}")

    # Load metadata to filter to TRAIN split and mask out zero-padded tokens
    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_parquet(metadata_path)
        assert len(meta) == activations.shape[0], "Metadata and activations row count mismatch"
        
        # 1. Filter prompts by split (train only)
        keep_prompts = (meta["split"] == split_filter).values
        print(f"Filtering to split='{split_filter}': {int(keep_prompts.sum())}/{len(meta)} prompts")
        activations = activations[keep_prompts]
        meta_filtered = meta[keep_prompts].reset_index(drop=True)
        
        # 2. Extract only valid sequence tokens (excluding pad tokens) using actual seq_len
        print("Masking out padding tokens for SAE training...")
        valid_acts = []
        for idx, row in meta_filtered.iterrows():
            seq_len = int(row["seq_len"])
            valid_acts.append(activations[idx, :seq_len, :])
        activations = torch.cat(valid_acts, dim=0) # Shape: (total_valid_tokens, d_model)
    else:
        print("WARNING: Metadata not found or not provided! Reshaping full activation tensor directly (training includes padding!).")
        if activations.dim() == 3:
            activations = activations.reshape(-1, activations.shape[-1])

    print(f"Total tokens for SAE training: {tuple(activations.shape)}")
    activations = activations.to(torch.float32)
    activation_mean = activations.mean(dim=0)
    activation_variance = activations.var(dim=0).mean().item()
    print(f"Token activation variance: {activation_variance:.4f}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using training device: {device}")
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=k, aux_k=aux_k).to(device)
    # Initialize decoder bias to activation mean
    model.b_dec.data = activation_mean.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Warmup scheduler
    def warmup_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    os.makedirs(output_dir, exist_ok=True)
    
    steps_since_fired = torch.zeros(d_hidden, device=device)
    
    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_mse = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            dead_mask = (steps_since_fired > dead_after_steps)

            optimizer.zero_grad()
            acts, reconstructed, aux_loss = model(batch, dead_mask=dead_mask)
            
            mse_loss = F.mse_loss(reconstructed, batch)
            loss = mse_loss
            if isinstance(aux_loss, torch.Tensor):
                 loss = loss + aux_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            model.normalize_decoder()
            
            # Update last fired status
            fired = (acts > 0).sum(dim=0) > 0
            steps_since_fired = torch.where(
                fired,
                torch.zeros_like(steps_since_fired),
                steps_since_fired + 1,
            )

            total_mse += mse_loss.item()
            dead_fraction = (steps_since_fired > dead_after_steps).float().mean().item()
            mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()
            norm_mse = mse_loss.item() / (activation_variance + 1e-8)

            pbar.set_postfix({
                "nMSE": f"{norm_mse:.3f}",
                "L0": f"{mean_l0:.1f}",
                "dead": f"{dead_fraction:.1%}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            global_step += 1

            # Hard reset of dead features if requested
            if resample_every and global_step % resample_every == 0 and dead_mask.any():
                _resample_dead(model, optimizer, activations, dead_mask, device)
                steps_since_fired[dead_mask] = 0
            
        avg_mse = total_mse / len(dataloader)
        avg_norm_mse = avg_mse / (activation_variance + 1e-8)
        print(f"Epoch {epoch+1} | MSE: {avg_mse:.4f} | nMSE: {avg_norm_mse:.3f} | L0: {mean_l0:.1f} | Dead: {dead_fraction:.1%}")
        
    save_path = os.path.join(output_dir, f"sae_topk_{k}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved SAE checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TopK Sparse Autoencoder")
    parser.add_argument("--activations", type=str, default="activations/hellaswag_activations.safetensors", help="Path to cached activations")
    parser.add_argument("--metadata", type=str, default="activations/hellaswag_metadata.parquet", help="Metadata parquet path")
    parser.add_argument("--d_model", type=int, default=None, help="Model hidden size")
    parser.add_argument("--d_hidden", type=int, default=None, help="SAE hidden size (defaults to 4x d_model)")
    parser.add_argument("--k", type=int, default=32, help="TopK active features")
    parser.add_argument("--aux_k", type=int, default=512, help="Auxiliary K for dead feature revival")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--split_filter", type=str, default="train", help="Which split to train the SAE on")
    parser.add_argument("--dead_after_steps", type=int, default=50, help="Steps before feature is dead")
    parser.add_argument("--resample_every", type=int, default=0, help="Hard-reset dead features every N steps (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="sae/checkpoints", help="Output directory")

    args = parser.parse_args()
    set_seed(args.seed)

    train_sae(
        activations_path=args.activations,
        metadata_path=args.metadata,
        split_filter=args.split_filter,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        k=args.k,
        aux_k=args.aux_k,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        epochs=args.epochs,
        dead_after_steps=args.dead_after_steps,
        resample_every=args.resample_every,
        output_dir=args.output_dir,
    )
