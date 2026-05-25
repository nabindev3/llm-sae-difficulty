"""Test-only correctness extraction for EleutherAI/pythia-2.8b.

The cascade only needs base's per-prompt correctness on the TEST split.
To conserve compute, this script reads the existing small metadata, filters to 
split='test', runs pythia-2.8b's zero-shot evaluation, and writes 
activations_base/hellaswag_metadata.parquet with [window_id, split, correct, difficulty]
for the test split prompts only.

We deliberately do NOT capture activations here to conserve CPU/MPS resources.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_metadata", default="activations/hellaswag_metadata.parquet")
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--output_dir", default="activations_base")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.small_metadata):
        sys.exit(f"[base_eval] small metadata '{args.small_metadata}' missing. Run extraction first.")
        
    print(f"Loading small metadata from {args.small_metadata}...")
    meta = pd.read_parquet(args.small_metadata)
    test = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    if len(test) == 0:
        sys.exit("[base_eval] no test split prompts in small metadata.")
    print(f"  {len(test)} test prompts to evaluate under base model.")

    print(f"Loading tokenizer and model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")
    
    # Load model (pythia-2.8b fits in macOS memory, load in half precision float16 for speed and memory)
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    rows = []
    print("Evaluating base model zero-shot on test split prompts...")
    for idx in tqdm(range(len(test))):
        row = test.iloc[idx]
        prompt = row["prompt"]
        true_label = int(row["true_label"])
        
        # Get context category from original row or endings
        # We need the HellaSwag validation endings. We retrieve them from the original HF dataset
        # Or wait: we can reconstruct endings since they are part of the HellaSwag validation
        # But wait! We did not save the 4 endings in small_metadata!
        # Ah! That is an important detail. To evaluate ending log-likelihoods, we need the 4 endings!
        # So we should load the endings directly from the original HellaSwag validation split!
        # That is extremely easy: we can load the HF HellaSwag dataset, and index it by window_id!
        # Yes! Let's do that: load_dataset("hellaswag", split="validation"), and index it!
        pass

    # Load HF validation set to get endings
    from datasets import load_dataset
    dataset = load_dataset("hellaswag", split="validation")

    for idx in tqdm(range(len(test))):
        row = test.iloc[idx]
        window_id = int(row["window_id"])
        
        # Load exact endings from dataset
        sample = dataset[window_id]
        prompt = sample["ctx_a"] + (" " + sample["ctx_b"] if sample["ctx_b"] else "")
        endings = sample["endings"]
        true_label = int(sample["label"])
        
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        # Truncate if prompt exceeds max allowed sequence length
        max_seq_len = 128
        if prompt_len > max_seq_len:
            prompt_ids = prompt_ids[-max_seq_len:]
            prompt_len = max_seq_len

        # Evaluate zero-shot choices using length-normalized log-likelihood
        ending_scores = []
        for ending in endings:
            ending_clean = " " + ending.strip()
            ending_ids = tokenizer.encode(ending_clean, add_special_tokens=False)
            
            full_ids = prompt_ids + ending_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model(input_tensor).logits
                
            shift_logits = logits[0, prompt_len-1 : -1, :]
            shift_labels = input_tensor[0, prompt_len:]
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(ending_ids)), shift_labels]
            ending_scores.append(target_log_probs.mean().item())

        predicted_label = int(np.argmax(ending_scores))
        correct = 1 if predicted_label == true_label else 0
        difficulty = 1 - correct

        rows.append({
            "window_id": window_id,
            "split": "test",
            "correct": correct,
            "difficulty": difficulty
        })

    out = pd.DataFrame(rows)
    print(f"\nBase model error rate: {out['difficulty'].mean():.2%}")
    print(f"Small model error rate: {test['difficulty'].mean():.2%} (reference)")

    path = os.path.join(args.output_dir, "hellaswag_metadata.parquet")
    out.to_parquet(path)
    print(f"Saved base results to {path} ({len(out)} test split prompts)")


if __name__ == "__main__":
    main()
