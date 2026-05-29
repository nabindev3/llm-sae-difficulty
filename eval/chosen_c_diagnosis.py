"""Diagnose why chosen_C varies wildly across L1-regularized probes (0.01–1.0).

Hypothesis: feature-count-aware scaling. With L1 logistic regression's
liblinear solver, the optimal C scales inversely with the regularization
budget needed to control overfitting on n_features. Higher-dimensional probes
(P3, P5 with ~4k features) should select smaller C; the low-dimensional
P1 baseline (8 features) should select larger C.

This script re-fits all five probe families with the full C grid, stores the
inner-CV ROC-AUC mean for each grid point, and produces a per-probe
regularization-path plot showing where the optimum lies.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sae.sae_model import TopKSAE
from probing.features import compute_prompt_stats, aggregate_sequence


C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]


def diagnose(label, meta_path, acts_path, sae_path, out_json, out_png):
    meta = pd.read_parquet(meta_path)
    raw_acts = load_file(acts_path)["encoder_embeddings"]
    state = torch.load(sae_path, map_location="cpu")
    d_model, d_hidden = state["W_enc"].shape
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32).to(device)
    sae.load_state_dict(state); sae.eval()

    # Pool raw + SAE features (mean+max+last for sequences with max_seq > 1)
    raw_agg = aggregate_sequence(raw_acts.numpy(), meta)
    N, max_seq, _ = raw_acts.shape
    flat = raw_acts.reshape(-1, d_model).to(device).to(torch.float32)
    sae_codes = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], 8192):
            a, _, _ = sae(flat[i:i+8192]); sae_codes.append(a.cpu())
    sae_codes = torch.cat(sae_codes, 0).reshape(N, max_seq, d_hidden).numpy()
    sae_agg = aggregate_sequence(sae_codes, meta)
    input_stats = compute_prompt_stats(meta)
    y = meta["difficulty"].values.astype(int)
    tr = (meta["split"] == "train").values

    probes = {
        "P1_InputStats":     input_stats,
        "P2_InputStats_Raw": np.concatenate([input_stats, raw_agg], axis=1),
        "P3_InputStats_SAE": np.concatenate([input_stats, sae_agg], axis=1),
        "P4_RawOnly":        raw_agg,
        "P5_SAEOnly":        sae_agg,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros((tr.sum(), 1)), y[tr]))

    print(f"=== {label}: chosen_C diagnosis ===")
    summary = {"label": label, "n_train": int(tr.sum()), "C_grid": C_GRID, "probes": {}}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for name, X in probes.items():
        n_feat = X.shape[1]
        Xs = StandardScaler().fit_transform(X[tr])
        base = LogisticRegression(penalty="l1", solver="liblinear",
                                  class_weight="balanced", max_iter=2000)
        gs = GridSearchCV(base, {"C": C_GRID}, scoring="roc_auc", cv=splits, n_jobs=-1)
        with parallel_backend("threading"):
            gs.fit(Xs, y[tr])

        mean_aucs = gs.cv_results_["mean_test_score"].tolist()
        std_aucs = gs.cv_results_["std_test_score"].tolist()
        chosen = float(gs.best_params_["C"])
        print(f"  {name:18s} n_feat={n_feat:5d}  chosen_C={chosen:<7.4g}  "
              f"best_cv_auc={max(mean_aucs):.4f}")
        for c, m, s in zip(C_GRID, mean_aucs, std_aucs):
            mark = " ← chosen" if c == chosen else ""
            print(f"      C={c:<7.4g}  mean_cv_auc={m:.4f} ± {s:.4f}{mark}")

        summary["probes"][name] = {
            "n_features": n_feat,
            "chosen_C": chosen,
            "mean_cv_auc_per_C": dict(zip([f"{c:g}" for c in C_GRID], mean_aucs)),
            "std_cv_auc_per_C":  dict(zip([f"{c:g}" for c in C_GRID], std_aucs)),
        }

        ax.errorbar(C_GRID, mean_aucs, yerr=std_aucs, marker="o", markersize=4,
                    linewidth=1.4, capsize=2, label=f"{name} (d={n_feat})")
        ax.axvline(chosen, color=ax.lines[-1].get_color(), linestyle=":", alpha=0.4)

    ax.set_xscale("log"); ax.set_xlabel("C (L1 inverse-regularization strength, log scale)")
    ax.set_ylabel("Inner-CV ROC AUC (5-fold, train split)")
    ax.set_title(f"L1 regularization path — {label}")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {out_png}\n  Saved {out_json}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True,
                    help="Each case: 'LABEL:meta_parquet:acts_safetensors:sae_ckpt'.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for spec in args.cases:
        label, m, a, s = spec.split(":")
        diagnose(label, m, a, s,
                 out_json=os.path.join(args.out_dir, f"{label}.json"),
                 out_png=os.path.join(args.out_dir, f"{label}.png"))
