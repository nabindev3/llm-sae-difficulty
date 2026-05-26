"""Post-hoc recalibration of LLM difficulty probe probabilities.

Uses Platt (sigmoid) and Isotonic regression, fit on Out-Of-Fold (OOF)
predictions over the HellaSwag training split to guarantee zero test leakage.

Plots recalibrated reliability diagrams to eval/results/reliability_recalibrated.png.
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
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sae')))
from sae.sae_model import TopKSAE
from probing.features import compute_prompt_stats, aggregate_sequence


def compute_ece_brier(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def reliability_pts(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out_x, out_y = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out_x.append(float(p[m].mean()))
        out_y.append(float(y[m].mean()))
    return out_x, out_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="activations/hellaswag_activations.safetensors")
    ap.add_argument("--metadata", default="activations/hellaswag_metadata.parquet")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--probe_scores", default=None,
                    help="If provided, write pred_<probe>_platt and pred_<probe>_isotonic "
                         "columns back into this parquet so cascade.py can route on "
                         "calibrated probabilities.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading metadata, activations, SAE...")
    meta = pd.read_parquet(args.metadata)
    raw_acts = load_file(args.activations)["encoder_embeddings"]
    state = torch.load(args.sae_ckpt, map_location="cpu")
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=32)
    sae.load_state_dict(state)
    sae.eval()

    tr = (meta["split"] == "train").values
    te = (meta["split"] == "test").values
    y = meta["difficulty"].values.astype(int)
    y_tr, y_te = y[tr], y[te]
    
    print(f"  n_train={tr.sum()}  n_test={te.sum()}  "
          f"hard_frac_train={y_tr.mean():.3f}  hard_frac_test={y_te.mean():.3f}")

    print("Computing prompt statistics...")
    input_stats = compute_prompt_stats(meta)

    # Focus recalibration on P1 and P3 (the primary baselines)
    probes = {"P1_InputStats": input_stats}

    print("Computing SAE-aggregated features (mean+max+last) for P3...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    sae = sae.to(device)
    N, max_seq, d_model_raw = raw_acts.shape
    raw_acts_2d = raw_acts.reshape(-1, d_model_raw).to(device).to(torch.float32)
    sae_acts_list = []
    
    with torch.no_grad():
        sae_batch_size = 8192
        for i in range(0, raw_acts_2d.shape[0], sae_batch_size):
            batch_slice = raw_acts_2d[i:i+sae_batch_size]
            acts_2d, _, _ = sae(batch_slice)
            sae_acts_list.append(acts_2d.cpu())
            
    sae_acts = torch.cat(sae_acts_list, dim=0).reshape(N, max_seq, d_hidden_ckpt).numpy()
    sae_agg = aggregate_sequence(sae_acts, meta)
    probes["P3_InputStats_SAE"] = np.concatenate([input_stats, sae_agg], axis=1)

    # 5-fold KFold OOF on train set
    n_tr = int(tr.sum())
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    results = {}
    calibrated_test_preds = {}   # name -> dict(platt: array, isotonic: array) on test set
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)

    for ax, (name, X) in zip(axes, probes.items()):
        print(f"\n--- {name} (features: {X.shape[1]}) ---")
        scaler = StandardScaler()
        X_tr_full = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        C_pick = 1.0 if name == "P1_InputStats" else 0.1

        # 5-fold OOF predictions on train split
        p_cal = np.zeros(n_tr, dtype=float)
        for fold_tr_idx, fold_te_idx in kf.split(X_tr_full):
            clf = LogisticRegression(penalty="l1", solver="liblinear",
                                     class_weight="balanced", max_iter=2000,
                                     C=C_pick)
            clf.fit(X_tr_full[fold_tr_idx], y_tr[fold_tr_idx])
            p_cal[fold_te_idx] = clf.predict_proba(X_tr_full[fold_te_idx])[:, 1]
            
        y_cal = y_tr

        # Final fit on full train to predict on test split
        base = LogisticRegression(penalty="l1", solver="liblinear",
                                  class_weight="balanced", max_iter=2000,
                                  C=C_pick)
        base.fit(X_tr_full, y_tr)
        p_te_raw = base.predict_proba(X_te)[:, 1]

        # Isotonic recalibration
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        p_te_iso = iso.transform(p_te_raw)

        # Platt recalibration (logistic regression on OOF scores)
        platt = LogisticRegression(C=1e6)
        platt.fit(p_cal.reshape(-1, 1), y_cal)
        p_te_pl = platt.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]

        # Compute ECE / Brier / AUROC
        ece_b, brier_b = compute_ece_brier(y_te, p_te_raw)
        ece_iso, brier_iso = compute_ece_brier(y_te, p_te_iso)
        ece_pl, brier_pl = compute_ece_brier(y_te, p_te_pl)
        
        unique_y_te = len(np.unique(y_te)) > 1
        auroc_b = float(roc_auc_score(y_te, p_te_raw)) if unique_y_te else None
        auroc_iso = float(roc_auc_score(y_te, p_te_iso)) if unique_y_te else None
        auroc_pl = float(roc_auc_score(y_te, p_te_pl)) if unique_y_te else None
        
        results[name] = {
            "raw":      {"ece": ece_b,   "brier": brier_b,   "auroc": auroc_b},
            "platt":    {"ece": ece_pl,  "brier": brier_pl,  "auroc": auroc_pl},
            "isotonic": {"ece": ece_iso, "brier": brier_iso, "auroc": auroc_iso},
        }
        calibrated_test_preds[name] = {"platt": p_te_pl, "isotonic": p_te_iso}
        
        print(f"  raw      ECE {ece_b:.3f}  Brier {brier_b:.3f}  AUROC {auroc_b:.3f}")
        print(f"  Platt    ECE {ece_pl:.3f}  Brier {brier_pl:.3f}  AUROC {auroc_pl:.3f}   (monotone, AUROC preserved)")
        print(f"  isotonic ECE {ece_iso:.3f}  Brier {brier_iso:.3f}  AUROC {auroc_iso:.3f}   (lower ECE)")

        bx, by = reliability_pts(y_te, p_te_raw)
        px, py = reliability_pts(y_te, p_te_pl)
        ix, iy = reliability_pts(y_te, p_te_iso)
        
        ax.plot([0, 1], [0, 1], ":", color="k", lw=1, label="perfect")
        ax.plot(bx, by, "o-", color="#e45756", lw=1.6,
                label=f"raw  (ECE {ece_b:.3f}, AUROC {auroc_b:.3f})")
        ax.plot(px, py, "s-", color="#54a24b", lw=1.6,
                label=f"Platt  (ECE {ece_pl:.3f}, AUROC {auroc_pl:.3f})")
        ax.plot(ix, iy, "^-", color="#4c78a8", lw=1.6,
                label=f"isotonic (ECE {ece_iso:.3f}, AUROC {auroc_iso:.3f})")
        ax.set_title(name)
        ax.set_xlabel("predicted P(hard)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
        
    axes[0].set_ylabel("actual hard frequency")
    fig.suptitle("Probe recalibration fit on K-fold OOF train predictions", fontsize=11)
    fig.tight_layout()
    
    out_png = os.path.join(args.out_dir, "reliability_recalibrated.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")
    
    with open(os.path.join(args.out_dir, "recalibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'recalibration_results.json')}")

    # Write calibrated probabilities back to probe_scores.parquet so cascade.py
    # can route on a true probability scale (τ = P(hard) > 0.x).
    if args.probe_scores and os.path.exists(args.probe_scores):
        df_scores = pd.read_parquet(args.probe_scores)
        test_window_ids = meta.loc[te, "window_id"].values
        for name, preds_d in calibrated_test_preds.items():
            for variant, arr in preds_d.items():
                col = f"pred_{name}_{variant}"
                mapping = dict(zip(test_window_ids, arr))
                df_scores[col] = df_scores["window_id"].map(mapping)
        df_scores.to_parquet(args.probe_scores)
        print(f"Wrote calibrated columns to {args.probe_scores}")
    elif args.probe_scores:
        print(f"[recalibrate] --probe_scores given but file not found: {args.probe_scores}; skipping write-back")


if __name__ == "__main__":
    main()
