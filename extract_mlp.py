"""Extract paired MLP-in / MLP-out activations for the transcoder variant.

A transcoder learns a sparse dictionary that maps one activation site to
another (here: the input and output of a layer's MLP sublayer) rather than
autoencoding a single site. The committed extractor (extract_activations.py)
only saves the residual stream, so this script replays the SAME SQuAD prompts
in the SAME order and the SAME boundary-token convention (prompt_len - 1),
hooking ``gpt_neox.layers[L].mlp`` to capture both its input and output.

Because the sample order and truncation match extract_activations.py exactly,
row i here corresponds to ``window_id == i`` in the committed
``<dataset>_metadata.parquet`` — so that metadata (difficulty/split/seq_len)
is reused directly for training the transcoder. We assert the row count
matches to guard against drift.

Usage:
    python extract_mlp.py --dataset squad --layer_idx 11 --max_samples 5000 \
        --ref_metadata activations/squad_metadata.parquet \
        --output_dir activations_mlp
"""
import argparse
import os

import pandas as pd
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="squad", choices=["squad"])
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--layer_idx", type=int, default=11)
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--ref_metadata", default="activations/squad_metadata.parquet",
                    help="Committed metadata whose row order/labels we reuse")
    ap.add_argument("--output_dir", default="activations_mlp")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("squad", split="validation")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"Device {device}; hooking gpt_neox.layers[{args.layer_idx}].mlp")

    captured = {}
    def hook(module, inp, out):
        captured["in"] = (inp[0] if isinstance(inp, tuple) else inp).detach().cpu().to(torch.float16)
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach().cpu().to(torch.float16)
    handle = model.gpt_neox.layers[args.layer_idx].mlp.register_forward_hook(hook)

    mlp_in, mlp_out = [], []
    for idx in tqdm(range(len(dataset))):
        s = dataset[idx]
        prompt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(prompt_ids) > 200:                       # match extract_activations.py
            prompt_ids = prompt_ids[-200:]
        prompt_len = len(prompt_ids)
        target_ids = tokenizer.encode(" " + s["answers"]["text"][0].strip(), add_special_tokens=False)
        full = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)

        captured.clear()
        with torch.no_grad():
            model(full)
        # mlp activations have shape (1, seq, d_model); boundary token = prompt_len-1.
        mlp_in.append(captured["in"][0, prompt_len - 1, :].clone().unsqueeze(0).unsqueeze(0))
        mlp_out.append(captured["out"][0, prompt_len - 1, :].clone().unsqueeze(0).unsqueeze(0))

    handle.remove()
    in_t = torch.cat(mlp_in, dim=0)     # (N, 1, d_model)
    out_t = torch.cat(mlp_out, dim=0)
    print(f"mlp_in {tuple(in_t.shape)}  mlp_out {tuple(out_t.shape)}")

    # Reuse the committed metadata (same order); guard against drift.
    ref = pd.read_parquet(args.ref_metadata)
    assert len(ref) == in_t.shape[0], (
        f"row mismatch: ref metadata has {len(ref)} rows, extracted {in_t.shape[0]}. "
        f"Use the same --max_samples as the committed extraction.")
    meta_path = os.path.join(args.output_dir, f"{args.dataset}_metadata.parquet")
    ref.to_parquet(meta_path)

    save_file({"encoder_embeddings": in_t}, os.path.join(args.output_dir, f"{args.dataset}_mlp_in.safetensors"))
    save_file({"encoder_embeddings": out_t}, os.path.join(args.output_dir, f"{args.dataset}_mlp_out.safetensors"))
    print(f"Saved MLP-in/out + reused metadata to {args.output_dir}/")


if __name__ == "__main__":
    main()
