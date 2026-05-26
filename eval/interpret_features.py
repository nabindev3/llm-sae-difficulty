"""Extracts the top-activating prompts for the top-5 difficulty-predictive SAE features on SQuAD.
Allows qualitative analysis of what semantic, syntactic, or contextual properties trigger
internal LLM difficulty representations.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

# Resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sae')))
from sae.sae_model import TopKSAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="activations/squad_metadata.parquet")
    ap.add_argument("--activations", default="activations/squad_activations.safetensors")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--ablation_results", default="eval/results/squad/squad_causal_ablation.json")
    ap.add_argument("--top_k", type=int, default=5, help="Number of top prompts to display per feature")
    ap.add_argument("--output_json", default="eval/results/squad/feature_interpretations.json")
    args = ap.parse_args()

    for p in (args.metadata, args.activations, args.sae_ckpt):
        if not os.path.exists(p):
            sys.exit(f"Error: missing {p}. Run SQuAD pipeline first.")

    print("Loading SQuAD activations and metadata...")
    df_meta = pd.read_parquet(args.metadata)
    tensors = load_file(args.activations)
    raw_acts = tensors["encoder_embeddings"] # Shape: (N, max_seq, d_model)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"Loading SAE checkpoint from {args.sae_ckpt}...")
    state = torch.load(args.sae_ckpt, map_location=device)
    d_model, d_hidden = state["W_enc"].shape
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32).to(device)
    sae.load_state_dict(state)
    sae.eval()

    # Identify top-5 difficulty-predictive features from causal ablation JSON
    if os.path.exists(args.ablation_results):
        print(f"Loading top features from {args.ablation_results}...")
        with open(args.ablation_results, "r") as f:
            abl_data = json.load(f)
        top_features = [int(f) for f in abl_data.get("feature_effects", {}).keys()]
    else:
        # Fallback: identify top features by fitting L1 Logistic directly on all train data
        print("Ablation results JSON not found. Fitting quick L1 logistic regression to identify features...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        # Batch extract SAE activations
        N, max_seq, d_model_raw = raw_acts.shape
        raw_acts_2d = raw_acts.reshape(-1, d_model_raw).to(device).to(torch.float32)
        sae_acts_list = []
        with torch.no_grad():
            sae_batch_size = 8192
            for i in range(0, raw_acts_2d.shape[0], sae_batch_size):
                batch_slice = raw_acts_2d[i:i+sae_batch_size]
                acts_2d, _, _ = sae(batch_slice)
                sae_acts_list.append(acts_2d.cpu())
        sae_acts = torch.cat(sae_acts_list, dim=0).reshape(N, max_seq, d_hidden).numpy()
        sae_agg = sae_acts[:, 0, :] # sequence length is 1 for SQuAD

        train_mask = (df_meta["split"] == "train").values
        y_tr = df_meta.loc[train_mask, "difficulty"].values
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(sae_agg[train_mask])
        
        clf = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000, C=0.1)
        clf.fit(X_tr_s, y_tr)
        coefs = clf.coef_[0]
        top_features = np.argsort(-np.abs(coefs))[:5].tolist()

    print(f"Targeting top difficulty features: {top_features}")

    # Extract SAE activations across all prompts
    N, max_seq, d_model_raw = raw_acts.shape
    raw_acts_2d = raw_acts.reshape(-1, d_model_raw).to(device).to(torch.float32)
    sae_acts_list = []
    
    print("Running SAE forward passes over SQuAD dataset...")
    with torch.no_grad():
        sae_batch_size = 8192
        for i in range(0, raw_acts_2d.shape[0], sae_batch_size):
            batch_slice = raw_acts_2d[i:i+sae_batch_size]
            acts_2d, _, _ = sae(batch_slice)
            sae_acts_list.append(acts_2d.cpu())
            
    sae_acts = torch.cat(sae_acts_list, dim=0).reshape(N, max_seq, d_hidden).numpy()
    sae_agg = sae_acts[:, 0, :] # Shape: (N, d_hidden)

    interpretations = {}
    
    print("\n====================================================================")
    print("           Top-Activating SQuAD Prompts for SAE Difficulty Features")
    print("====================================================================")
    
    for feat in top_features:
        feat_acts = sae_agg[:, feat]
        
        # Find indices of top activating prompts
        top_indices = np.argsort(-feat_acts)[:args.top_k]
        
        print(f"\n--- Feature {feat} (Difficulty Probe Coefficient Importance) ---")
        interpretations[feat] = []
        
        for rank, idx in enumerate(top_indices):
            act_val = float(feat_acts[idx])
            if act_val <= 1e-6:
                continue # Skip unactivated prompts
                
            row = df_meta.iloc[idx]
            prompt = row["prompt"]
            gold_answer = row["gold_answer"]
            ppl = float(row["perplexity"])
            difficulty = int(row["difficulty"])
            
            # Format and display
            print(f"Rank {rank+1} | Activation: {act_val:.4f} | Perplexity: {ppl:.1f} | Difficulty: {difficulty}")
            # Truncate prompt context for clean printing but print question clearly
            lines = prompt.split("\n")
            context_line = lines[0] if len(lines) > 0 else ""
            question_line = lines[1] if len(lines) > 1 else prompt
            if len(context_line) > 100:
                context_line = context_line[:100] + "..."
            
            print(f"  Context : {context_line}")
            print(f"  Question: {question_line}")
            print(f"  Answer  : {gold_answer}")
            print("-" * 50)
            
            interpretations[feat].append({
                "rank": rank + 1,
                "prompt_index": int(idx),
                "activation": act_val,
                "perplexity": ppl,
                "difficulty": difficulty,
                "prompt_text": prompt,
                "gold_answer": gold_answer
            })
            
    # Write JSON output
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(interpretations, f, indent=2)
    print(f"\nSuccessfully saved interpretations to {args.output_json}")


if __name__ == "__main__":
    main()
