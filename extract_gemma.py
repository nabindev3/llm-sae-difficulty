"""Gemma-2-2B + Gemma Scope extraction for the frontier-SAE test (Limitation 5).

Reviewer objection: the null (SAE features add no incremental difficulty signal
over raw activations) might just reflect the quality of a small model's
self-trained SAE. This script repeats the SQuAD difficulty-probe setup on a
stronger model (Gemma-2-2B) using an *off-the-shelf, frontier-quality* SAE
(Gemma Scope) rather than one we trained.

Mirrors extract_activations.py's SQuAD path exactly so the comparison is fair:
  - same prompt template and boundary-token convention (prompt_len - 1),
  - difficulty = top-25% of train-normalized gold-target perplexity (Gemma's own),
  - same leakage controls (prompt-ppl<=1.5 contamination purge + TF-IDF dedup).

Emits, all aligned and in the format probing/probe.py already consumes:
  gemma_activations.safetensors  raw resid_post at the SAE's layer, (N,1,d_model)
  gemma_metadata.parquet         difficulty/split/seq_len/prompt/activity_label/ppl
  gemma_sae_codes.safetensors    Gemma Scope SAE codes, (N,1,d_hidden)

Then, in the MAIN pipeline venv:
  python probing/probe.py --activations <out>/gemma_activations.safetensors \
      --metadata <out>/gemma_metadata.parquet \
      --sae_codes <out>/gemma_sae_codes.safetensors \
      --results_json probing/results/gemma_probe_results.json \
      --scores_parquet <out>/gemma_scores.parquet

Runs in the dedicated ~/.venvs/gemma-scope venv (sae_lens + transformer_lens).
Requires HF auth + accepted google/gemma-2-2b license.

Usage:
    python extract_gemma.py --layer 12 --width 16k --max_samples 5000 \
        --output_dir activations_gemma
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
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--sae_release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="Force a device. 'cpu' uses pageable RAM (no wired-MPS jetsam "
                         "thrash) at the cost of speed -- the only local option on a 16GB Mac.")
    ap.add_argument("--chunk_size", type=int, default=500,
                    help="Samples per on-disk chunk. Extraction is resumable at "
                         "chunk granularity, so a jetsam kill costs <= one chunk.")
    ap.add_argument("--output_dir", default="activations_gemma")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    chunk_dir = os.path.join(args.output_dir, "_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    dataset = load_dataset("rajpurkar/squad", split="validation").select(range(args.max_samples))
    n = len(dataset)
    train_cutoff = int(n * 0.7)
    starts = list(range(0, n, args.chunk_size))

    def chunk_path(st):
        return os.path.join(chunk_dir, f"chunk_{st:06d}.pt")

    pending = [st for st in starts if not os.path.exists(chunk_path(st))]
    print(f"{n} samples, {len(starts)} chunks of {args.chunk_size}; {len(pending)} pending.")

    # Only pay the model/SAE memory cost if there is extraction left to do. Once
    # all chunks exist (possibly across several jetsam-killed restarts), assembly
    # below needs no model -- so the final pass is cheap and always completes.
    if pending:
        from transformer_lens import HookedTransformer
        from sae_lens import SAE

        if args.device != "auto":
            device = args.device
        else:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        # fp16 on MPS: best kernel coverage (bf16 falls back to CPU for several
        # Gemma-2 ops -> ~25s/iter thrash). bf16 elsewhere: half the ~10GB fp32
        # footprint (fits in pageable RAM on CPU) and matches Gemma Scope's training.
        dtype = torch.float16 if device == "mps" else torch.bfloat16
        print(f"Device {device}; loading {args.model} ...")
        # from_pretrained_no_processing (not from_pretrained): (1) skips the
        # LN-folding/centering weight-processing pass that spikes RAM and SIGKILLs
        # on a ~13GB Colab instance; (2) leaves the residual stream in the model's
        # RAW basis -- which is what Gemma Scope SAEs were trained on. Default
        # processing rewrites resid_post and would feed the SAE out-of-basis acts.
        model = HookedTransformer.from_pretrained_no_processing(args.model, dtype=dtype).to(device)
        model.eval()

        hook_name = f"blocks.{args.layer}.hook_resid_post"
        sae_id = f"layer_{args.layer}/width_{args.width}/canonical"
        print(f"Loading Gemma Scope SAE: {args.sae_release} / {sae_id}")
        loaded = SAE.from_pretrained(args.sae_release, sae_id, device=device)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded   # older sae_lens returns a tuple
        sae.eval()

        for st in pending:
            en = min(st + args.chunk_size, n)
            raw_c, code_c, meta_c = [], [], []
            for idx in tqdm(range(st, en), desc=f"chunk {st}-{en}"):
                s = dataset[idx]
                prompt = f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer:"
                target = " " + s["answers"]["text"][0].strip()

                prompt_ids = model.to_tokens(prompt)[0].tolist()         # includes BOS
                target_ids = model.to_tokens(target, prepend_bos=False)[0].tolist()
                if len(prompt_ids) > 200:
                    prompt_ids = prompt_ids[-200:]
                prompt_len = len(prompt_ids)
                full = torch.tensor([prompt_ids + target_ids], device=device)

                with torch.no_grad():
                    logits, cache = model.run_with_cache(full, names_filter=hook_name)
                    resid = cache[hook_name][0]                          # (seq, d_model)
                    boundary = resid[prompt_len - 1, :].float()          # (d_model,)
                    code = sae.encode(boundary.to(sae.W_enc.dtype)).float()  # (d_hidden,)

                    # Gold-target perplexity (continuous self-difficulty), Gemma's own.
                    shift_logits = logits[0, prompt_len - 1:-1, :]
                    shift_labels = full[0, prompt_len:]
                    tgt_ce = F.cross_entropy(shift_logits.float(), shift_labels).item()

                    # Prompt perplexity (Pile-contamination check) from the SAME forward:
                    # next-token CE over the prompt span only -- no second forward pass.
                    p_ce = F.cross_entropy(logits[0, :prompt_len - 1, :].float(),
                                           full[0, 1:prompt_len]).item()
                    del cache

                raw_c.append(boundary.cpu().to(torch.float16).reshape(1, 1, -1))
                code_c.append(code.cpu().to(torch.float16).reshape(1, 1, -1))
                meta_c.append({
                    "window_id": idx, "dataset": "squad", "prompt": prompt,
                    "gold_answer": s["answers"]["text"][0],
                    "perplexity": float(np.exp(tgt_ce)),
                    "prompt_perplexity": float(np.exp(p_ce)),
                    "activity_label": s.get("title", "Unknown"),
                    "seq_len": 1,
                    "split": "train" if idx < train_cutoff else "test",
                })
            torch.save({"raw": torch.cat(raw_c), "codes": torch.cat(code_c), "meta": meta_c},
                       chunk_path(st))
            print(f"saved chunk {st}-{en}", flush=True)

    # --- Assemble all chunks (no model needed) ---------------------------------
    raw_parts, code_parts, metadata = [], [], []
    for st in starts:
        d = torch.load(chunk_path(st), weights_only=False)
        raw_parts.append(d["raw"])
        code_parts.append(d["codes"])
        metadata.extend(d["meta"])
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

    train_rows = (df["split"] == "train").values
    mean_tr, std_tr = df.loc[train_rows, "perplexity"].mean(), df.loc[train_rows, "perplexity"].std()
    df["perplexity_norm"] = (df["perplexity"] - mean_tr) / (std_tr + 1e-8)
    thr = df.loc[train_rows, "perplexity_norm"].quantile(0.75)
    df["difficulty"] = (df["perplexity_norm"] >= thr).astype(int)
    df["correct"] = 1 - df["difficulty"]
    print(f"Difficulty threshold (top 25% train ppl): {thr:.3f}")
    print(df["split"].value_counts().to_string())

    raw_t = torch.cat(raw_parts, dim=0)
    code_t = torch.cat(code_parts, dim=0)
    print(f"raw {tuple(raw_t.shape)}  codes {tuple(code_t.shape)}")
    save_file({"encoder_embeddings": raw_t}, os.path.join(args.output_dir, "gemma_activations.safetensors"))
    save_file({"encoder_embeddings": code_t}, os.path.join(args.output_dir, "gemma_sae_codes.safetensors"))
    df.to_parquet(os.path.join(args.output_dir, "gemma_metadata.parquet"))
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
