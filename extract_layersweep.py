"""Capture boundary activations at EVERY layer in one pass (for the depth sweep).

Limitation: the SAE-vs-raw null is reported at only two layers (L12, L18) -- two
points isn't a curve. This hooks all decoder layers simultaneously and stores the
boundary-token residual at each, so a per-layer SAE+probe sweep can plot
Delta(SAE-Raw) vs depth from a single extraction.

It also records, at the boundary token, signals for the *stronger baseline*
(Limitation: harden the cheap baseline):
  * max_softmax   -- max next-token probability (model confidence)
  * logit_entropy -- entropy of the next-token distribution
These let probing/features.py build a harder P1 without re-running the model.

Continuous difficulty (gold-target perplexity quantile), same leakage controls
as extract_activations.py. Pythia/GPTNeoX by default.

Output:
  <out>/squad_layeract.safetensors   encoder_embeddings (N, n_layers, d_model)
  <out>/squad_metadata.parquet       difficulty/split/.../max_softmax/logit_entropy

Usage:
    python extract_layersweep.py --model EleutherAI/pythia-410m --dataset squad \
        --max_samples 5000 --output_dir activations_layersweep
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

LAYER_PATHS = ["gpt_neox.layers", "transformer.h", "model.layers"]


def get_layers(model):
    for path in LAYER_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise ValueError("could not locate decoder layers")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--dataset", default="squad", choices=["squad"])
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--output_dir", default="activations_layersweep")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    layers = get_layers(model)
    n_layers = len(layers)
    print(f"Device {device}; {args.model}: {n_layers} layers")

    captured = {}
    for i, layer in enumerate(layers):
        layer.register_forward_hook(
            lambda _m, _i, out, i=i: captured.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach()))

    ds = load_dataset("rajpurkar/squad", split="validation").select(range(args.max_samples))
    train_cutoff = int(len(ds) * 0.7)

    acts, metadata = [], []
    for idx in tqdm(range(len(ds))):
        s = ds[idx]
        prompt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
        target = " " + s["answers"]["text"][0].strip()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(prompt_ids) > 200:
            prompt_ids = prompt_ids[-200:]
        prompt_len = len(prompt_ids)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        full = torch.tensor([prompt_ids + target_ids], device=device)

        captured.clear()
        with torch.no_grad():
            logits = model(full).logits
        # Boundary residual at every layer -> (n_layers, d_model)
        per_layer = torch.stack([captured[i][0, prompt_len - 1, :].float() for i in range(n_layers)], dim=0)

        # Stronger-baseline signals at the boundary (next-token distribution).
        bdist = F.softmax(logits[0, prompt_len - 1, :].float(), dim=-1)
        max_softmax = float(bdist.max())
        logit_entropy = float(-(bdist * (bdist + 1e-12).log()).sum())

        # Continuous difficulty (gold-target perplexity) + prompt ppl.
        tgt_ce = F.cross_entropy(logits[0, prompt_len - 1:-1, :].float(), full[0, prompt_len:]).item()
        p_ce = F.cross_entropy(logits[0, :prompt_len - 1, :].float(), full[0, 1:prompt_len]).item()

        acts.append(per_layer.cpu().to(torch.float16).unsqueeze(0))   # (1, n_layers, d_model)
        metadata.append({
            "window_id": idx, "dataset": "squad", "prompt": prompt,
            "gold_answer": s["answers"]["text"][0],
            "perplexity": float(np.exp(tgt_ce)), "prompt_perplexity": float(np.exp(p_ce)),
            "max_softmax": max_softmax, "logit_entropy": logit_entropy,
            "activity_label": s.get("title", "all"), "seq_len": 1,
            "split": "train" if idx < train_cutoff else "test",
        })

    df = pd.DataFrame(metadata)
    # Leakage controls (mirror extract_activations.py)
    df.loc[df["prompt_perplexity"] <= 1.5, "split"] = "purge"
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf = vec.fit_transform(df["prompt"])
    tr, te = df[df.split == "train"].index.values, df[df.split == "test"].index.values
    if len(tr) and len(te):
        sim = cosine_similarity(tfidf[te], tfidf[tr])
        df.loc[te[np.max(sim, axis=1) >= 0.7], "split"] = "purge"
    train_rows = (df.split == "train").values
    mean_tr, std_tr = df.loc[train_rows, "perplexity"].mean(), df.loc[train_rows, "perplexity"].std()
    df["perplexity_norm"] = (df["perplexity"] - mean_tr) / (std_tr + 1e-8)
    thr = df.loc[train_rows, "perplexity_norm"].quantile(0.75)
    df["difficulty"] = (df["perplexity_norm"] >= thr).astype(int)
    df["correct"] = 1 - df["difficulty"]
    print(df["split"].value_counts().to_string())

    emb = torch.cat(acts, dim=0)   # (N, n_layers, d_model)
    print(f"layer activations {tuple(emb.shape)}")
    save_file({"encoder_embeddings": emb}, os.path.join(args.output_dir, f"{args.dataset}_layeract.safetensors"))
    df.to_parquet(os.path.join(args.output_dir, f"{args.dataset}_metadata.parquet"))
    print(f"Saved to {args.output_dir}/  (n_layers={n_layers})")


if __name__ == "__main__":
    main()
