import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors.torch import save_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Extract activations from Pythia-410M on HellaSwag")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m", help="Hugging Face model ID")
    parser.add_argument("--layer_idx", type=int, default=11, help="Layer index to hook (0-indexed, default 11 = layer 12)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum validation samples to process (None for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="activations", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Step 1: Loading HellaSwag dataset ===")
    dataset = load_dataset("hellaswag", split="validation")
    total_samples = len(dataset)
    print(f"Loaded HellaSwag validation split containing {total_samples} samples.")

    if args.max_samples is not None:
        num_to_process = min(args.max_samples, total_samples)
        indices = list(range(num_to_process))
        dataset = dataset.select(indices)
        print(f"Sub-selected the first {len(dataset)} samples for processing.")
    else:
        num_to_process = total_samples

    # Define 70% Train and 30% Test splits
    train_cutoff = int(len(dataset) * 0.7)
    print(f"Initial split boundaries: {train_cutoff} train, {len(dataset) - train_cutoff} test")

    print("\n=== Step 2: Loading model and tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")
    
    # Load model
    model_dtype = torch.float32 if device in ["cpu", "mps"] else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    # Hook setup
    captured_acts = []
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured_acts.append(hidden_states.detach().cpu().to(torch.float16))

    # Hook the output of the requested layer residual stream
    print(f"Hooking gpt_neox.layers[{args.layer_idx}]...")
    handle = model.gpt_neox.layers[args.layer_idx].register_forward_hook(hook_fn)

    metadata = []
    all_embeddings = []
    
    print("\n=== Step 3: Running zero-shot evaluation and activation extraction ===")
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        ctx_a = sample["ctx_a"]
        ctx_b = sample["ctx_b"]
        prompt = ctx_a + (" " + ctx_b if ctx_b else "")
        endings = sample["endings"]
        true_label = int(sample["label"])
        category = sample.get("activity_label", "Unknown")

        # Encode prompt alone to compute perplexity and boundary location
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        # Truncate if prompt exceeds standard limit to prevent array indexing errors
        max_prompt_len = 124  # leave space for candidate tokens
        if prompt_len > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
            prompt_len = max_prompt_len

        # Compute prompt perplexity under Pythia-410M to detect Pile pretraining contamination
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        captured_acts.clear()
        with torch.no_grad():
            outputs = model(prompt_tensor)
            prompt_logits = outputs.logits
        
        shift_prompt_logits = prompt_logits[0, :-1, :]
        shift_prompt_labels = prompt_tensor[0, 1:]
        loss = F.cross_entropy(shift_prompt_logits, shift_prompt_labels, reduction="mean").item()
        prompt_ppl = np.exp(loss)

        # For multiple choice probing, we evaluate the prompt + each_candidate sequence
        # and capture the residual activation at the FIRST token of the candidate completion.
        # Index of first token of ending is exactly prompt_len (0-indexed).
        question_acts = []
        ending_scores = []
        
        for ending in endings:
            ending_clean = " " + ending.strip()
            ending_ids = tokenizer.encode(ending_clean, add_special_tokens=False)
            
            full_ids = prompt_ids + ending_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            captured_acts.clear()
            with torch.no_grad():
                logits = model(input_tensor).logits
                
            # Retrieve Layer 12 residual activation captured by the hook
            assert len(captured_acts) == 1, "Hook failed to capture activations"
            # raw_act shape: (seq_len, d_model)
            raw_act = captured_acts[0][0]
            
            # The first token position of the candidate is exactly index prompt_len
            first_candidate_token_act = raw_act[prompt_len, :].clone()
            question_acts.append(first_candidate_token_act.unsqueeze(0))
            
            # Compute likelihoods of the ending tokens
            shift_logits = logits[0, prompt_len-1 : -1, :]
            shift_labels = input_tensor[0, prompt_len:]
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(ending_ids)), shift_labels]
            ending_scores.append(target_log_probs.mean().item())

        # Stack the 4 candidate activations along the sequence dimension: shape (4, d_model)
        stacked_question_act = torch.cat(question_acts, dim=0) # Shape: (4, d_model)
        all_embeddings.append(stacked_question_act.unsqueeze(0))

        predicted_label = int(np.argmax(ending_scores))
        correct = 1 if predicted_label == true_label else 0
        difficulty = 1 - correct  # 1 if incorrect, 0 if correct

        metadata.append({
            "window_id": idx,
            "dataset": "hellaswag",
            "prompt": prompt,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": correct,
            "difficulty": difficulty,
            "activity_label": category,
            "seq_len": 4,  # sequence length is exactly 4 candidates now
            "perplexity": prompt_ppl,
            "split": "train" if idx < train_cutoff else "test"
        })

    handle.remove()
    print("Hook successfully removed.")

    df_meta = pd.DataFrame(metadata)

    print("\n=== Step 4: Leakage Control (Purging overlaps & contamination) ===")
    
    # 1. Pile contamination purge (perplexity <= 1.5)
    contamination_mask = df_meta["perplexity"] <= 1.5
    num_contaminated = contamination_mask.sum()
    df_meta.loc[contamination_mask, "split"] = "purge"
    print(f"Purged {num_contaminated} contaminated prompts (perplexity <= 1.5).")

    # 2. Prompt-cluster TF-IDF deduplication
    print("Computing TF-IDF matrices for deduplication...")
    valid_prompts_mask = df_meta["split"] != "purge"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df_meta["prompt"])

    train_indices = df_meta[(df_meta["split"] == "train") & valid_prompts_mask].index.values
    test_indices = df_meta[(df_meta["split"] == "test") & valid_prompts_mask].index.values

    if len(train_indices) > 0 and len(test_indices) > 0:
        print("Calculating similarity matrices...")
        train_tfidf = tfidf_matrix[train_indices]
        test_tfidf = tfidf_matrix[test_indices]
        
        sim = cosine_similarity(test_tfidf, train_tfidf)
        overlaps = np.max(sim, axis=1) >= 0.7
        purged_test_indices = test_indices[overlaps]
        
        df_meta.loc[purged_test_indices, "split"] = "purge"
        print(f"Purged {len(purged_test_indices)} test prompts sharing TF-IDF cosine similarity >= 0.7 with train prompts.")
    else:
        print("Skipping deduplication: train or test split is empty.")

    print("\nSplit statistics:")
    print(df_meta["split"].value_counts().to_string())
    print(f"Total HellaSwag Zero-Shot Accuracy: {df_meta[df_meta['split'] != 'purge']['correct'].mean():.2%}")

    # Save outputs
    print("\n=== Step 5: Saving activations and metadata ===")
    final_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Final embeddings shape: {final_tensor.shape}")

    safetensors_path = os.path.join(args.output_dir, "hellaswag_activations.safetensors")
    save_file({"encoder_embeddings": final_tensor}, safetensors_path)
    print(f"Saved activations to {safetensors_path}")

    parquet_path = os.path.join(args.output_dir, "hellaswag_metadata.parquet")
    df_meta.to_parquet(parquet_path, engine="pyarrow")
    print(f"Saved metadata to {parquet_path}")


if __name__ == "__main__":
    main()
