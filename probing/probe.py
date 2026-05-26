import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Include the parent and sibling directory in sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sae')))
from sae.sae_model import TopKSAE
from probing.features import compute_prompt_stats, aggregate_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="activations/hellaswag_metadata.parquet")
    parser.add_argument("--activations", type=str, default="activations/hellaswag_activations.safetensors")
    parser.add_argument("--sae_ckpt", type=str, default="sae/checkpoints/sae_topk_32.pt")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--results_json", type=str, default="probing/results/probe_results.json")
    parser.add_argument("--scores_parquet", type=str, default="activations/probe_scores.parquet")
    args = parser.parse_args()

    print("Loading HellaSwag metadata and activations...")
    df_meta = pd.read_parquet(args.metadata)
    tensors = load_file(args.activations)
    raw_acts = tensors["encoder_embeddings"] # (batch, max_seq_len, d_model)
    
    print("Computing prompt statistics...")
    input_stats = compute_prompt_stats(df_meta)
    
    print("Loading SAE...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    if not os.path.exists(args.sae_ckpt):
        sys.exit(f"[probe] SAE checkpoint '{args.sae_ckpt}' not found. Train the SAE first; refusing to probe with random weights.")
        
    state = torch.load(args.sae_ckpt, map_location=device)
    if "W_enc" not in state:
        sys.exit(f"[probe] '{args.sae_ckpt}' is not a TopKSAE checkpoint.")
        
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    print(f"Auto-detected SAE dimensions from checkpoint: d_model={d_model_ckpt}, d_hidden={d_hidden_ckpt}")
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=args.k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    
    print("Aggregating activations (padding-aware sequence pooling)...")
    raw_agg = aggregate_sequence(raw_acts.numpy(), df_meta)
    
    # Process padded prompt activations in batch through the SAE
    N, max_seq, d_model_raw = raw_acts.shape
    raw_acts_2d = raw_acts.reshape(-1, d_model_raw).to(device).to(torch.float32)
    sae_acts_list = []
    
    with torch.no_grad():
        # Batch size for forward pass through the SAE
        sae_batch_size = 8192
        for i in range(0, raw_acts_2d.shape[0], sae_batch_size):
            batch_slice = raw_acts_2d[i:i+sae_batch_size]
            acts_2d, _, _ = sae(batch_slice)
            sae_acts_list.append(acts_2d.cpu())
            
    sae_acts = torch.cat(sae_acts_list, dim=0).reshape(N, max_seq, d_hidden_ckpt).numpy()
    
    print("Aggregating SAE codes (padding-aware sequence pooling)...")
    sae_agg = aggregate_sequence(sae_acts, df_meta)
    
    # Target difficulty: 1 if incorrect, 0 if correct
    y = df_meta["difficulty"].values
    
    train_mask = (df_meta["split"] == "train").values
    test_mask = (df_meta["split"] == "test").values
    
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        sys.exit("Not enough train/test split data. Run full extraction with larger max_samples first.")
        
    print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}, Hard (incorrect) rate: {y[test_mask].mean():.1%}")
    
    y_train, y_test = y[train_mask], y[test_mask]
    
    probes = {
        "P1_InputStats":     input_stats,
        "P2_InputStats_Raw": np.concatenate([input_stats, raw_agg], axis=1),
        "P3_InputStats_SAE": np.concatenate([input_stats, sae_agg], axis=1),
        "P4_RawOnly":        raw_agg,
        "P5_SAEOnly":        sae_agg,
    }
    
    results = {}
    preds = {}

    # Inner CV settings
    n_splits = max(2, min(5, int(np.bincount(y_train).min()) - 1, train_mask.sum() // 10))
    C_grid = {"C": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]}

    # Stratified CV: We combine category (activity_label) and label (correctness)
    # into a composite key to perform stratified-by-topic cross-validation
    strat_key = df_meta.loc[train_mask, "activity_label"].astype(str) + "_" + y_train.astype(str)
    counts = strat_key.value_counts()
    
    # If any topic-label group is too small, fallback to standard stratified CV by y
    if (counts < n_splits).any():
        print(f"  CV: Using stratified-by-label CV (some topic-label groups have count < {n_splits})")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_target = y_train
    else:
        print(f"  CV: Using stratified-by-topic CV across {n_splits} folds")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_target = strat_key.values

    for name, X in probes.items():
        print(f"Training probe: {name} (features: {X.shape[1]})")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_mask])
        X_test_s = scaler.transform(X[test_mask])

        base = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000)
        gs = GridSearchCV(base, C_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        
        from joblib import parallel_backend
        with parallel_backend("threading"):
            gs.fit(X_train_s, cv_target if cv_target is not y_train else y_train)
        
        preds[name] = gs.predict_proba(X_test_s)[:, 1]
        point = roc_auc_score(y_test, preds[name]) if len(np.unique(y_test)) > 1 else 0.0
        results[name] = {"AUROC": point, "best_C": gs.best_params_["C"]}
        print(f"  {name} point AUROC = {point:.3f}  (C={gs.best_params_['C']})")

    # PAIRED bootstrap to compute 95% CIs and deltas
    print("Running paired bootstrap (B=2000)...")
    rng = np.random.default_rng(42)
    names = list(probes.keys())
    boot = {n: [] for n in names}
    pairs = [
        ("P2_InputStats_Raw", "P1_InputStats"),
        ("P3_InputStats_SAE", "P1_InputStats"),
        ("P3_InputStats_SAE", "P2_InputStats_Raw")
    ]
    boot_delta = {f"{a}-{b}": [] for a, b in pairs}
    idx_all = np.arange(len(y_test))
    
    if len(np.unique(y_test)) > 1:
        for _ in range(2000):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if len(np.unique(y_test[idx])) < 2:
                continue
            per = {n: roc_auc_score(y_test[idx], preds[n][idx]) for n in names}
            for n in names:
                boot[n].append(per[n])
            for a, b in pairs:
                boot_delta[f"{a}-{b}"].append(per[a] - per[b])

    def _ci(arr):
        if not arr:
            return (np.nan, np.nan)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    for n in names:
        lo, hi = _ci(boot[n])
        results[n]["95%_CI_lower"] = lo
        results[n]["95%_CI_upper"] = hi
        print(f"  {n} AUROC 95% CI: [{lo:.3f}, {hi:.3f}]")

    delta_raw = results["P2_InputStats_Raw"]["AUROC"] - results["P1_InputStats"]["AUROC"]
    delta_sae = results["P3_InputStats_SAE"]["AUROC"] - results["P1_InputStats"]["AUROC"]
    delta_sae_over_raw = results["P3_InputStats_SAE"]["AUROC"] - results["P2_InputStats_Raw"]["AUROC"]
    d_raw_ci = _ci(boot_delta["P2_InputStats_Raw-P1_InputStats"])
    d_sae_ci = _ci(boot_delta["P3_InputStats_SAE-P1_InputStats"])
    d_sor_ci = _ci(boot_delta["P3_InputStats_SAE-P2_InputStats_Raw"])
    
    print("\n--- Incremental Predictive Power (ΔAUROC, paired bootstrap) ---")
    print(f"Δ Raw - Stats : {delta_raw:+.3f}  95% CI [{d_raw_ci[0]:+.3f}, {d_raw_ci[1]:+.3f}]")
    print(f"Δ SAE - Stats : {delta_sae:+.3f}  95% CI [{d_sae_ci[0]:+.3f}, {d_sae_ci[1]:+.3f}]")
    print(f"Δ SAE - Raw   : {delta_sae_over_raw:+.3f}  95% CI [{d_sor_ci[0]:+.3f}, {d_sor_ci[1]:+.3f}]")
        
    df_test = df_meta[test_mask].copy()
    for name, p in preds.items():
        df_test[f"pred_{name}"] = p
        
    os.makedirs(os.path.dirname(os.path.abspath(args.scores_parquet)), exist_ok=True)
    df_test.to_parquet(args.scores_parquet)
    print(f"\nSaved {args.scores_parquet}")

    # Save probe results JSON
    import json
    os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
    
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    hard_fraction = float(y[test_mask].mean()) if n_test > 0 else 0.0
    
    final_results = {
        "n_total": len(df_meta),
        "n_train": n_train,
        "n_test": n_test,
        "hard_fraction": hard_fraction,
        "P1_AUROC": results.get("P1_InputStats", {}).get("AUROC", 0.0),
        "P1_CI_lower": results.get("P1_InputStats", {}).get("95%_CI_lower", 0.0),
        "P1_CI_upper": results.get("P1_InputStats", {}).get("95%_CI_upper", 0.0),
        "P2_AUROC": results.get("P2_InputStats_Raw", {}).get("AUROC", 0.0),
        "P2_CI_lower": results.get("P2_InputStats_Raw", {}).get("95%_CI_lower", 0.0),
        "P2_CI_upper": results.get("P2_InputStats_Raw", {}).get("95%_CI_upper", 0.0),
        "P3_AUROC": results.get("P3_InputStats_SAE", {}).get("AUROC", 0.0),
        "P3_CI_lower": results.get("P3_InputStats_SAE", {}).get("95%_CI_lower", 0.0),
        "P3_CI_upper": results.get("P3_InputStats_SAE", {}).get("95%_CI_upper", 0.0),
        "delta_raw": float(delta_raw),
        "delta_raw_CI_lower": d_raw_ci[0],
        "delta_raw_CI_upper": d_raw_ci[1],
        "delta_sae": float(delta_sae),
        "delta_sae_CI_lower": d_sae_ci[0],
        "delta_sae_CI_upper": d_sae_ci[1],
        "delta_sae_over_raw": float(delta_sae_over_raw),
        "delta_sae_over_raw_CI_lower": d_sor_ci[0],
        "delta_sae_over_raw_CI_upper": d_sor_ci[1],
        # Diagnostic probes
        "P4_RawOnly_AUROC": results.get("P4_RawOnly", {}).get("AUROC", 0.0),
        "P4_RawOnly_CI_lower": results.get("P4_RawOnly", {}).get("95%_CI_lower", 0.0),
        "P4_RawOnly_CI_upper": results.get("P4_RawOnly", {}).get("95%_CI_upper", 0.0),
        "P5_SAEOnly_AUROC": results.get("P5_SAEOnly", {}).get("AUROC", 0.0),
        "P5_SAEOnly_CI_lower": results.get("P5_SAEOnly", {}).get("95%_CI_lower", 0.0),
        "P5_SAEOnly_CI_upper": results.get("P5_SAEOnly", {}).get("95%_CI_upper", 0.0),
        "chosen_C": {k: v.get("best_C") for k, v in results.items() if "best_C" in v},
    }
    
    with open(args.results_json, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved {args.results_json}")


if __name__ == "__main__":
    main()
