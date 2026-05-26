"""Test-only correctness / perplexity extraction for EleutherAI/pythia-2.8b.

The cascade only needs base's per-prompt metrics on the TEST split.
This script reads the existing small metadata, filters to split='test', runs
pythia-2.8b's zero-shot evaluation (HellaSwag choice scoring or SQuAD generation perplexity),
and writes activations_base/[dataset]_metadata.parquet for the test split prompts.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="hellaswag", choices=["hellaswag", "squad"])
    ap.add_argument("--small_metadata", type=str, default=None)
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--output_dir", default="activations_base")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Set default metadata path based on dataset if not specified
    if args.small_metadata is None:
        args.small_metadata = f"activations/{args.dataset}_metadata.parquet"

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
    
    # Load model (pythia-2.8b fit in macOS float16)
    model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    # Load dataset to retrieve candidates or gold answers
    print(f"Loading HF validation dataset for {args.dataset} to retrieve prompt completions...")
    if args.dataset == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
    else:
        dataset = load_dataset("squad", split="validation")

    rows = []
    print("Evaluating base model zero-shot on test split prompts...")
    for idx in tqdm(range(len(test))):
        row = test.iloc[idx]
        window_id = int(row["window_id"])
        
        sample = dataset[window_id]
        
        if args.dataset == "hellaswag":
            prompt = sample["ctx_a"] + (" " + sample["ctx_b"] if sample["ctx_b"] else "")
            endings = sample["endings"]
            true_label = int(sample["label"])
            
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(prompt_ids)

            max_seq_len = 128
            if prompt_len > max_seq_len:
                prompt_ids = prompt_ids[-max_seq_len:]
                prompt_len = max_seq_len

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
            
        else:  # squad
            context = sample["context"]
            question = sample["question"]
            gold_answer = sample["answers"]["text"][0]
            
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            target = " " + gold_answer.strip()
            
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            target_len = len(target_ids)
            
            max_prompt_len = 200
            if prompt_len > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]
                prompt_len = max_prompt_len
                
            full_ids = prompt_ids + target_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model(input_tensor).logits
                
            shift_logits = logits[0, prompt_len-1 : -1, :]
            shift_labels = input_tensor[0, prompt_len:]
            
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean").item()
            target_perplexity = np.exp(loss)
            
            rows.append({
                "window_id": window_id,
                "split": "test",
                "perplexity": target_perplexity
            })

    out = pd.DataFrame(rows)
    
    if args.dataset == "squad":
        # SQuAD uses perplexity. We normalize and define binary difficulty based on train split stats
        # To keep it completely consistent, we load the small metadata train split stats
        train_rows = (meta["split"] == "train").values
        mean_tr = meta.loc[train_rows, "perplexity"].mean()
        std_tr = meta.loc[train_rows, "perplexity"].std()
        
        # SQuAD base normalized perplexity
        out["perplexity_norm"] = (out["perplexity"] - mean_tr) / (std_tr + 1e-8)
        
        # SQuAD base difficulty (using small model's train threshold!)
        threshold_ppl = meta.loc[train_rows, "perplexity_norm"].quantile(0.75)
        out["difficulty"] = (out["perplexity_norm"] >= threshold_ppl).astype(int)
        out["correct"] = 1 - out["difficulty"]
        
        print(f"\nBase model error rate (squad): {out['difficulty'].mean():.2%}")
        print(f"Small model error rate (squad): {test['difficulty'].mean():.2%} (reference)")
    else:
        print(f"\nBase model error rate (hellaswag): {out['difficulty'].mean():.2%}")
        print(f"Small model error rate (hellaswag): {test['difficulty'].mean():.2%} (reference)")

    path = os.path.join(args.output_dir, f"{args.dataset}_metadata.parquet")
    out.to_parquet(path)
    print(f"Saved base results to {path} ({len(out)} test split prompts)")


if __name__ == "__main__":
    main()
