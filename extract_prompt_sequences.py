"""Capture the FULL prompt-portion activations for each sample at a chosen layer.

Companion to extract_activations.py, which only saves a single boundary token per
sample. This script re-runs the prompt-only forward pass (so we capture the residual
stream at every position the prompt occupies, not just the boundary) and saves the
padded full-prompt tensor for SAE training and all-position causal ablation.

Output:
  activations_allpos/{dataset}_activations.safetensors   shape (N, max_seq_len, d_model)
  activations_allpos/{dataset}_metadata.parquet         mirrors the existing metadata,
                                                          but `seq_len` reflects the
                                                          actual prompt length per row.

We re-use window_id, difficulty, perplexity, and split assignments from the existing
metadata (loaded from --base_metadata) to guarantee parity with the boundary-only
pipeline — no re-running of the leakage purge or threshold logic.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hellaswag", "squad"], required=True)
    ap.add_argument("--base_metadata", required=True,
                    help="Existing metadata.parquet from extract_activations.py — used to "
                         "pin window_id/split/difficulty so all-position activations align "
                         "row-for-row with the boundary-only ones.")
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--layer_idx", type=int, default=11,
                    help="0-indexed layer to hook. Default 11 = Layer 12 (mid-residual).")
    ap.add_argument("--output_dir", default="activations_allpos")
    ap.add_argument("--max_seq_len", type=int, default=None,
                    help="If None: 128 for hellaswag, 200 for squad (matches extract_activations.py limits).")
    args = ap.parse_args()

    if args.max_seq_len is None:
        args.max_seq_len = 128 if args.dataset == "hellaswag" else 200

    os.makedirs(args.output_dir, exist_ok=True)
    base_meta = pd.read_parquet(args.base_metadata)
    print(f"Loaded base metadata: {len(base_meta)} rows (window_ids {base_meta['window_id'].min()}..{base_meta['window_id'].max()})")

    if args.dataset == "hellaswag":
        ds = load_dataset("hellaswag", split="validation")
    else:
        ds = load_dataset("squad", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device in ["cpu", "mps"] else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    captured = []
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().cpu().to(torch.float16))
    handle = model.gpt_neox.layers[args.layer_idx].register_forward_hook(hook)

    d_model = model.config.hidden_size
    N = len(base_meta)
    all_acts = torch.zeros(N, args.max_seq_len, d_model, dtype=torch.float16)
    actual_seq_lens = np.zeros(N, dtype=np.int32)
    print(f"Allocating output tensor: ({N}, {args.max_seq_len}, {d_model}) fp16 = "
          f"{N * args.max_seq_len * d_model * 2 / 1e9:.2f} GB")

    for row_i, row in enumerate(tqdm(base_meta.itertuples(), total=N)):
        wid = int(row.window_id)
        sample = ds[wid]

        if args.dataset == "hellaswag":
            ctx_a = sample["ctx_a"]
            ctx_b = sample["ctx_b"]
            prompt = ctx_a + (" " + ctx_b if ctx_b else "")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) > args.max_seq_len:
                prompt_ids = prompt_ids[-args.max_seq_len:]
        else:
            context = sample["context"]
            question = sample["question"]
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) > args.max_seq_len:
                prompt_ids = prompt_ids[-args.max_seq_len:]

        seq_len = len(prompt_ids)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        captured.clear()
        with torch.no_grad():
            _ = model(input_tensor)
        assert len(captured) == 1, "Hook failed to capture activations"
        raw = captured[0][0]  # (seq_len, d_model)
        all_acts[row_i, :seq_len, :] = raw
        actual_seq_lens[row_i] = seq_len

    handle.remove()

    # Write metadata that mirrors base but with the all-position seq_len
    out_meta = base_meta.copy()
    out_meta["seq_len"] = actual_seq_lens

    safetensors_path = os.path.join(args.output_dir, f"{args.dataset}_activations.safetensors")
    save_file({"encoder_embeddings": all_acts}, safetensors_path)
    print(f"Saved activations: {safetensors_path} (shape {tuple(all_acts.shape)})")

    parquet_path = os.path.join(args.output_dir, f"{args.dataset}_metadata.parquet")
    out_meta.to_parquet(parquet_path)
    print(f"Saved metadata:    {parquet_path}")
    print(f"seq_len stats: min={actual_seq_lens.min()} max={actual_seq_lens.max()} mean={actual_seq_lens.mean():.1f}")


if __name__ == "__main__":
    main()
