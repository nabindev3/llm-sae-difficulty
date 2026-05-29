"""Per-pooling ablation for aggregate_sequence.

Question: when probing difficulty from sequence activations, which of the three
pooling operations carries the signal — mean, max, or last token? The current
default concatenates all three (3x feature dimension). If one pooling dominates,
the feature space can be simplified for the methods section.

We fit P_raw (raw activations) and P_sae (SAE features) probes on each pooling
separately using the all-position activations and dataset-matched all-position
SAE checkpoints, and report test-set AUROC + paired-bootstrap CI.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import parallel_backend

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sae.sae_model import TopKSAE
from probing.features import compute_prompt_stats


POOLS = ["mean", "max", "last"]


def pool_sequence(seq_tensor: np.ndarray, seq_lens: np.ndarray, kind: str) -> np.ndarray:
    """Pool (N, max_seq_len, d) along the seq axis, respecting per-row seq_len."""
    if seq_tensor.ndim != 3:
        raise ValueError(f"Expected (N, max_seq, d), got {seq_tensor.shape}")
    N, max_seq, d = seq_tensor.shape
    out = np.zeros((N, d), dtype=seq_tensor.dtype)
    for i in range(N):
        L = min(int(seq_lens[i]), max_seq)
        if L <= 0:
            continue
        v = seq_tensor[i, :L, :]
        if kind == "mean":
            out[i] = v.mean(axis=0)
        elif kind == "max":
            out[i] = v.max(axis=0)
        elif kind == "last":
            out[i] = v[-1, :]
        else:
            raise ValueError(kind)
    return out


def encode_sae(raw_acts: torch.Tensor, sae: TopKSAE, device: str, batch_size: int = 8192):
    """Encode (N, max_seq, d_model) → (N, max_seq, d_hidden) via TopK SAE."""
    N, max_seq, d_model = raw_acts.shape
    flat = raw_acts.reshape(-1, d_model).to(device).to(torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], batch_size):
            acts, _, _ = sae(flat[i:i + batch_size])
            out.append(acts.cpu())
    return torch.cat(out, dim=0).reshape(N, max_seq, -1).numpy()


def fit_probe(X: np.ndarray, y: np.ndarray, train_mask, test_mask, n_splits=5, seed=42, B=2000):
    """Fit L1 logistic with inner CV, return point AUROC + bootstrap CI."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros((len(y_train), 1)), y_train))

    base = LogisticRegression(penalty="l1", solver="liblinear",
                              class_weight="balanced", max_iter=2000)
    grid = {"C": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]}
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=splits, n_jobs=-1)
    with parallel_backend("threading"):
        gs.fit(X_train, y_train)

    p_test = gs.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.0

    rng = np.random.default_rng(seed)
    boot = []
    idx_all = np.arange(len(y_test))
    for _ in range(B):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y_test[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_test[idx], p_test[idx]))
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)

    return {
        "AUROC": auc,
        "CI_lower": lo,
        "CI_upper": hi,
        "chosen_C": float(gs.best_params_["C"]),
        "n_features": int(X.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--sae_ckpt", required=True)
    ap.add_argument("--label", required=True, help="Short label for the JSON output (e.g. 'hellaswag_l12_allpos').")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== Per-pooling ablation — {args.label} ===")
    meta = pd.read_parquet(args.metadata)
    raw_acts = load_file(args.activations)["encoder_embeddings"]
    seq_lens = meta["seq_len"].to_numpy(dtype=np.int32)
    y = meta["difficulty"].values.astype(int)
    train_mask = (meta["split"] == "train").values
    test_mask = (meta["split"] == "test").values
    input_stats = compute_prompt_stats(meta)
    print(f"  N={len(meta)} train={train_mask.sum()} test={test_mask.sum()}  "
          f"max_seq={raw_acts.shape[1]} d_model={raw_acts.shape[2]}")

    # Load SAE and encode the full sequence
    state = torch.load(args.sae_ckpt, map_location="cpu")
    d_model, d_hidden = state["W_enc"].shape
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32).to(device)
    sae.load_state_dict(state); sae.eval()
    print(f"  Encoding {raw_acts.numel():,} tokens through SAE on {device}...")
    sae_acts_np = encode_sae(raw_acts, sae, device)
    raw_acts_np = raw_acts.numpy()

    results = {"label": args.label, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
               "by_pooling": {}}

    for pool in POOLS:
        print(f"\n  -- pool = {pool} --")
        raw_pooled = pool_sequence(raw_acts_np, seq_lens, pool)
        sae_pooled = pool_sequence(sae_acts_np, seq_lens, pool)

        raw_X = np.concatenate([input_stats, raw_pooled], axis=1)  # P2-like
        sae_X = np.concatenate([input_stats, sae_pooled], axis=1)  # P3-like
        raw_only_X = raw_pooled                                     # P4-like
        sae_only_X = sae_pooled                                     # P5-like

        per_pool = {}
        for probe_name, X in [
            ("P2_Stats+Raw", raw_X),
            ("P3_Stats+SAE", sae_X),
            ("P4_RawOnly",   raw_only_X),
            ("P5_SAEOnly",   sae_only_X),
        ]:
            r = fit_probe(X, y, train_mask, test_mask, seed=args.seed)
            per_pool[probe_name] = r
            print(f"    {probe_name:14s} n_feat={r['n_features']:5d}  C={r['chosen_C']:<7.4g}  "
                  f"AUROC={r['AUROC']:.4f}  [{r['CI_lower']:.4f}, {r['CI_upper']:.4f}]")
        results["by_pooling"][pool] = per_pool

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
