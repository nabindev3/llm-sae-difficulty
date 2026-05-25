"""Hook-based causal ablation of Top-5 difficulty-predictive SAE features.

Mishra-style residual stream patching on EleutherAI/pythia-410m at Layer 12.
For each test split prompt, we measure the zero-shot accuracy under three conditions:
  - natural (no hook)
  - recon (hidden states replaced by SAE reconstruction)
  - ablate(feat=j) (hidden states replaced by SAE reconstruction with feature j zeroed out)

We report the change in error rate (ablate - recon) and run paired bootstrap to compute
95% confidence intervals.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sae')))
from sae.sae_model import TopKSAE
from probing.features import aggregate_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="activations/hellaswag_activations.safetensors")
    ap.add_argument("--metadata", default="activations/hellaswag_metadata.parquet")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for p in (args.activations, args.metadata, args.sae_ckpt):
        if not os.path.exists(p):
            sys.exit(f"[ablation] missing input file: {p}. Run extraction & SAE training first.")

    print("Loading metadata, activations, SAE...")
    meta = pd.read_parquet(args.metadata)
    raw_acts = load_file(args.activations)["encoder_embeddings"]
    state = torch.load(args.sae_ckpt, map_location="cpu")
    d_model, d_hidden = state["W_enc"].shape
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32).to(device)
    sae.load_state_dict(state)
    sae.eval()

    y = meta["difficulty"].values.astype(int)
    tr = (meta["split"] == "train").values
    te = (meta["split"] == "test").values
    y_tr, y_te = y[tr], y[te]

    # Step 1: Identify top-5 difficulty-predictive SAE features using L1 Logistic on train split
    print("Fitting a quick L1 logistic regression to identify top-5 SAE features...")
    # Reshape and pass raw acts to SAE to getaggregated train split features
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
    sae_agg = aggregate_sequence(sae_acts, meta)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(sae_agg[tr])
    
    # Fit L1 logistic regression with balanced class weights
    clf = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000, C=0.1)
    clf.fit(X_tr_s, y_tr)
    
    # Logistic coefficients correspond to mean, max, last of each feature
    # We want the top-5 feature indices that have the highest absolute coefficient
    coefs = clf.coef_[0]
    # Reshape to (3, d_hidden) and sum absolute coefficients across the 3 poolings to rank features
    feature_importance = np.abs(coefs.reshape(3, d_hidden)).sum(axis=0)
    top_5_features = np.argsort(-feature_importance)[:5].tolist()
    print(f"Top-5 features identified: {top_5_features}")

    # Step 2: Load model and register the patch hook
    print(f"Loading {args.model} and registering patch hook...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_dtype = torch.float32 if device in ["cpu", "mps"] else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype).to(device)
    model.eval()

    # Hook control states
    active_feat_idx = None
    ablate_active = False
    recon_active = False

    def patch_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        x = hidden_states.to(torch.float32)
        orig_shape = x.shape
        x_2d = x.reshape(-1, d_model)
        
        with torch.no_grad():
            acts, x_reconstruct, _ = sae(x_2d)
            
        if ablate_active and active_feat_idx is not None:
            # Zero out the selected feature
            acts[:, active_feat_idx] = 0.0
            x_reconstruct = acts @ sae.W_dec + sae.b_dec
            
        if recon_active or ablate_active:
            reconstructed_states = x_reconstruct.reshape(orig_shape).to(hidden_states.dtype)
            if isinstance(output, tuple):
                return (reconstructed_states,) + output[1:]
            else:
                return reconstructed_states
        return output

    # Hook Layer 12
    handle = model.gpt_neox.layers[11].register_forward_hook(patch_hook)

    # Step 3: Run zero-shot evaluation on test prompts for the three conditions
    # Load HellaSwag validation dataset to get prompt endings
    from datasets import load_dataset
    dataset = load_dataset("hellaswag", split="validation")
    test_meta = meta[te].copy().reset_index(drop=True)

    print(f"Evaluating {len(test_meta)} test prompts under ablated conditions...")
    results_rows = []

    for idx in tqdm(range(len(test_meta))):
        row = test_meta.iloc[idx]
        window_id = int(row["window_id"])
        sample = dataset[window_id]
        
        prompt = sample["ctx_a"] + (" " + sample["ctx_b"] if sample["ctx_b"] else "")
        endings = sample["endings"]
        true_label = int(sample["label"])
        
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)
        
        max_seq_len = 128
        if prompt_len > max_seq_len:
            prompt_ids = prompt_ids[-max_seq_len:]
            prompt_len = max_seq_len

        def run_eval():
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
            return 0 if predicted_label == true_label else 1 # returns 1 for incorrect (difficulty)

        # 1. Natural condition
        recon_active, ablate_active = False, False
        err_natural = run_eval()

        # 2. Reconstruction condition
        recon_active, ablate_active = True, False
        err_recon = run_eval()

        # 3. Ablated conditions for each of the top-5 features
        recon_active, ablate_active = False, True
        err_ablations = {}
        for feat in top_5_features:
            active_feat_idx = feat
            err_ablations[feat] = run_eval()

        row_res = {
            "window_id": window_id,
            "err_natural": err_natural,
            "err_recon": err_recon,
            **{f"err_ablate_{feat}": err_ablations[feat] for feat in top_5_features}
        }
        results_rows.append(row_res)

    handle.remove()
    print("Ablation hook removed.")

    df_res = pd.DataFrame(results_rows)
    df_res.to_parquet("eval/results/causal_ablation.parquet")
    print("Saved eval/results/causal_ablation.parquet")

    # Step 4: Perform Paired Bootstrap (B=2000) to compute deltas and CIs
    print("Running paired bootstrap on causal ablation outcomes...")
    n_test = len(df_res)
    rng = np.random.default_rng(args.seed)
    
    natural_errors = df_res["err_natural"].values
    recon_errors = df_res["err_recon"].values
    
    delta_recon_natural = float((recon_errors - natural_errors).mean())
    
    boot_drn = []
    boot_df = {feat: [] for feat in top_5_features}
    idx_all = np.arange(n_test)
    
    for _ in range(2000):
        idx = rng.choice(idx_all, size=n_test, replace=True)
        boot_drn.append((recon_errors[idx] - natural_errors[idx]).mean())
        for feat in top_5_features:
            feat_errs = df_res[f"err_ablate_{feat}"].values
            boot_df[feat].append((feat_errs[idx] - recon_errors[idx]).mean())

    def _ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    recon_nat_ci = _ci(boot_drn)
    
    summary = {
        "top_5_features": top_5_features,
        "natural_error_mean": float(natural_errors.mean()),
        "recon_error_mean": float(recon_errors.mean()),
        "delta_recon_natural": delta_recon_natural,
        "delta_recon_natural_ci_lower": recon_nat_ci[0],
        "delta_recon_natural_ci_upper": recon_nat_ci[1],
        "feature_effects": {}
    }

    print("\n--- Causal Ablation Results (Mean Delta Error Rate vs Recon) ---")
    print(f"Recon - Natural reconstruction penalty: {delta_recon_natural:+.3f}  95% CI [{recon_nat_ci[0]:+.3f}, {recon_nat_ci[1]:+.3f}]")
    
    for feat in top_5_features:
        feat_errs = df_res[f"err_ablate_{feat}"].values
        delta_feat = float((feat_errs - recon_errors).mean())
        feat_ci = _ci(boot_df[feat])
        
        summary["feature_effects"][feat] = {
            "delta_error": delta_feat,
            "ci_lower": feat_ci[0],
            "ci_upper": feat_ci[1]
        }
        print(f"Feature {feat:4d} ablation: {delta_feat:+.3f}  95% CI [{feat_ci[0]:+.3f}, {feat_ci[1]:+.3f}]")

    with open(os.path.join(args.out_dir, "causal_ablation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {os.path.join(args.out_dir, 'causal_ablation.json')}")


if __name__ == "__main__":
    main()
