"""Architecture- and dataset-general activation extractor.

Generalizes extract_activations.py (which is hardwired to Pythia/GPTNeoX +
hellaswag/squad) to support new backbones and benchmarks for Limitations 1/6/7:

  * Backbones (any HF causal LM) -- layers located by walking known paths
    (gpt_neox.layers / transformer.h / model.layers), so Pythia, GPT-2,
    Qwen2.5, Llama all work without per-model code.
  * Datasets:
      - mc  (binary correctness): hellaswag, arc_easy  -> difficulty = 1-correct
        where correct = argmax over per-choice mean log-prob == gold.
      - gen (continuous):         squad, triviaqa      -> difficulty = top-25%
        of train-normalized gold-answer perplexity (the model's own).

Stores the BOUNDARY-token residual activation (last prompt token), seq_len=1,
shape (N,1,d_model) -- uniform with the SQuAD pipeline, so the committed
sae/train_sae.py + probing/probe.py consume the output unchanged. Applies the
same leakage controls (prompt-ppl<=1.5 contamination purge + TF-IDF dedup).

Usage:
    python extract_general.py --model EleutherAI/pythia-410m --dataset arc_easy \
        --layer_idx 11 --max_samples 5000 --output_dir activations_arc_l12
    python extract_general.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset squad \
        --layer_idx 12 --max_samples 5000 --output_dir activations_qwen_squad_l12
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

MC_DATASETS = {"hellaswag", "arc_easy"}
GEN_DATASETS = {"squad", "triviaqa"}
LAYER_PATHS = ["gpt_neox.layers", "transformer.h", "model.layers", "model.decoder.layers"]


def get_layers(model):
    """Return the decoder-layer ModuleList for any supported architecture."""
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
            return obj, path
    raise ValueError(f"could not locate decoder layers; tried {LAYER_PATHS}")


def load_mc_items(dataset, max_samples):
    """Return list of {prompt, choices:[str], gold_idx, category} for MC tasks."""
    items = []
    if dataset == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation").select(range(max_samples))
        for s in ds:
            ctx = s["ctx_a"] + ((" " + s["ctx_b"]) if s["ctx_b"] else "")
            items.append({"prompt": ctx, "choices": list(s["endings"]),
                          "gold_idx": int(s["label"]), "category": s.get("activity_label", "all")})
    elif dataset == "arc_easy":
        # ARC-Easy splits are small (val=570); combine for a usable N (~5197),
        # then re-split 70/30 internally like the rest of the pipeline.
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train+validation+test")
        ds = ds.select(range(min(max_samples, len(ds))))
        for s in ds:
            labels = s["choices"]["label"]
            texts = s["choices"]["text"]
            key = s["answerKey"]
            if key not in labels:      # rare malformed rows
                continue
            items.append({"prompt": f"Question: {s['question']}\nAnswer:",
                          "choices": [" " + t.strip() for t in texts],
                          "gold_idx": labels.index(key), "category": "arc"})
    return items


def load_gen_items(dataset, max_samples):
    """Return list of {prompt, target, category} for generative tasks."""
    items = []
    if dataset == "squad":
        ds = load_dataset("rajpurkar/squad", split="validation").select(range(max_samples))
        for s in ds:
            items.append({"prompt": f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:",
                          "target": " " + s["answers"]["text"][0].strip(),
                          "category": s.get("title", "all")})
    elif dataset == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        ds = ds.select(range(min(max_samples, len(ds))))
        for s in ds:
            ans = s["answer"]["value"].strip()
            if not ans:
                continue
            items.append({"prompt": f"Question: {s['question']}\nAnswer:",
                          "target": " " + ans, "category": "trivia"})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--dataset", required=True, choices=sorted(MC_DATASETS | GEN_DATASETS))
    ap.add_argument("--layer_idx", type=int, default=11)
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    task = "mc" if args.dataset in MC_DATASETS else "gen"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16
    print(f"Device {device}; loading {args.model} (task={task}, dataset={args.dataset})")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    layers, path = get_layers(model)
    if args.layer_idx >= len(layers):
        raise SystemExit(f"--layer_idx {args.layer_idx} >= n_layers {len(layers)} for {args.model}")
    print(f"Hooking {path}[{args.layer_idx}] of {len(layers)} layers")
    captured = []

    def hook(_m, _i, out):
        captured.append((out[0] if isinstance(out, tuple) else out).detach())
    handle = layers[args.layer_idx].register_forward_hook(hook)

    items = (load_mc_items if task == "mc" else load_gen_items)(args.dataset, args.max_samples)
    n = len(items)
    train_cutoff = int(n * 0.7)
    print(f"{n} items loaded")

    embeddings, metadata = [], []
    for idx in tqdm(range(n)):
        it = items[idx]
        prompt_ids = tokenizer.encode(it["prompt"], add_special_tokens=True)
        if len(prompt_ids) > 200:
            prompt_ids = prompt_ids[-200:]
        prompt_len = len(prompt_ids)

        # Forward over the prompt: boundary-token activation + prompt perplexity.
        captured.clear()
        with torch.no_grad():
            p_logits = model(torch.tensor([prompt_ids], device=device)).logits
        boundary = captured[0][0, prompt_len - 1, :].float().cpu()
        p_ce = F.cross_entropy(p_logits[0, :-1, :].float(),
                               torch.tensor(prompt_ids[1:], device=device)).item()
        prompt_ppl = float(np.exp(p_ce))

        row = {"window_id": idx, "dataset": args.dataset, "prompt": it["prompt"],
               "activity_label": it["category"], "seq_len": 1, "prompt_perplexity": prompt_ppl,
               "split": "train" if idx < train_cutoff else "test"}

        if task == "mc":
            scores = []
            for ch in it["choices"]:
                ch_ids = tokenizer.encode(ch, add_special_tokens=False)
                if not ch_ids:
                    scores.append(-1e9)
                    continue
                full = torch.tensor([prompt_ids + ch_ids], device=device)
                with torch.no_grad():
                    logits = model(full).logits
                lp = F.log_softmax(logits[0, prompt_len - 1:-1, :].float(), dim=-1)
                tok_lp = lp[torch.arange(len(ch_ids)), torch.tensor(ch_ids)]
                scores.append(tok_lp.mean().item())
            pred = int(np.argmax(scores))
            row["correct"] = int(pred == it["gold_idx"])
            row["difficulty"] = 1 - row["correct"]
            row["perplexity"] = prompt_ppl            # no gold target; use prompt ppl
        else:  # gen: gold-answer perplexity from one extra forward
            tgt_ids = tokenizer.encode(it["target"], add_special_tokens=False)
            full_ids = prompt_ids + tgt_ids
            full = torch.tensor([full_ids], device=device)
            with torch.no_grad():
                logits = model(full).logits
            t_ce = F.cross_entropy(logits[0, prompt_len - 1:-1, :].float(),
                                   torch.tensor(tgt_ids, device=device)).item()
            row["perplexity"] = float(np.exp(t_ce))   # gold-target ppl (continuous difficulty)

        embeddings.append(boundary.reshape(1, 1, -1))
        metadata.append(row)

    handle.remove()
    df = pd.DataFrame(metadata)

    # --- Leakage controls (mirror extract_activations.py) ----------------------
    contam = df["prompt_perplexity"] <= 1.5
    df.loc[contam, "split"] = "purge"
    print(f"Purged {int(contam.sum())} contaminated prompts (prompt_ppl <= 1.5).")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf = vec.fit_transform(df["prompt"])
    tr = df[df["split"] == "train"].index.values
    te = df[df["split"] == "test"].index.values
    if len(tr) and len(te):
        sim = cosine_similarity(tfidf[te], tfidf[tr])
        purged = te[np.max(sim, axis=1) >= 0.7]
        df.loc[purged, "split"] = "purge"
        print(f"Purged {len(purged)} test prompts with TF-IDF cosine >= 0.7 to train.")

    if task == "gen":
        train_rows = (df["split"] == "train").values
        mean_tr, std_tr = df.loc[train_rows, "perplexity"].mean(), df.loc[train_rows, "perplexity"].std()
        df["perplexity_norm"] = (df["perplexity"] - mean_tr) / (std_tr + 1e-8)
        thr = df.loc[train_rows, "perplexity_norm"].quantile(0.75)
        df["difficulty"] = (df["perplexity_norm"] >= thr).astype(int)
        df["correct"] = 1 - df["difficulty"]
        print(f"Difficulty threshold (top 25% train ppl): {thr:.3f}")

    print("Split / difficulty:")
    print(df["split"].value_counts().to_string())
    if "difficulty" in df:
        hard = df.loc[df.split == "test", "difficulty"].mean() if (df.split == "test").any() else float("nan")
        print(f"test hard-rate: {hard:.3f}")

    emb = torch.cat(embeddings, dim=0)
    print(f"activations {tuple(emb.shape)}")
    save_file({"encoder_embeddings": emb}, os.path.join(args.output_dir, f"{args.dataset}_activations.safetensors"))
    df.to_parquet(os.path.join(args.output_dir, f"{args.dataset}_metadata.parquet"))
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
