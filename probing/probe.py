import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from joblib import parallel_backend
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

from sae.sae_model import TopKSAE
from probing.features import compute_prompt_stats, aggregate_sequence


# Canonical rung names. Kept as module constants so the ladder, the bootstrap
# pairs, and the JSON keys can never drift out of lockstep.
P1, P2, P3, P4, P5 = (
    "P1_InputStats",
    "P2_InputStats_Raw",
    "P3_InputStats_SAE",
    "P4_RawOnly",
    "P5_SAEOnly",
)

# The three headline comparisons, as (numerator, reference) rung pairs.
HEADLINE_PAIRS = [(P2, P1), (P3, P1), (P3, P2)]

C_GRID = {"C": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="activations/hellaswag_metadata.parquet")
    parser.add_argument("--activations", type=str, default="activations/hellaswag_activations.safetensors")
    parser.add_argument("--sae_ckpt", type=str, default="sae/checkpoints/sae_topk_32.pt")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--results_json", type=str, default="probing/results/probe_results.json")
    parser.add_argument("--scores_parquet", type=str, default="activations/probe_scores.parquet")
    return parser.parse_args()


def select_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def load_dataset(metadata_path, activations_path):
    """Load prompt metadata and the raw encoder activations tensor."""
    df_meta = pd.read_parquet(metadata_path)
    tensors = load_file(activations_path)
    raw_acts = tensors["encoder_embeddings"]  # (batch, max_seq_len, d_model)
    return df_meta, raw_acts


def load_sae(sae_ckpt, k, device):
    """Load a TopKSAE from a checkpoint, auto-detecting its dimensions.

    Exits (rather than silently probing with random weights) if the checkpoint
    is missing or is not a TopKSAE. Returns the SAE and its hidden width.
    """
    if not os.path.exists(sae_ckpt):
        sys.exit(f"[probe] SAE checkpoint '{sae_ckpt}' not found. Train the SAE first; refusing to probe with random weights.")

    state = torch.load(sae_ckpt, map_location=device, weights_only=True)
    if "W_enc" not in state:
        sys.exit(f"[probe] '{sae_ckpt}' is not a TopKSAE checkpoint.")

    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    print(f"Auto-detected SAE dimensions from checkpoint: d_model={d_model_ckpt}, d_hidden={d_hidden_ckpt}")
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    return sae, d_hidden_ckpt


def encode_sae_codes(sae, raw_acts, device, d_hidden, batch_size=8192):
    """Run the padded prompt activations through the SAE -> dense code tensor.

    Returns a (N, max_seq, d_hidden) numpy array of SAE codes, matching the
    shape of the raw activations so it can be pooled the same way.
    """
    N, max_seq, d_model_raw = raw_acts.shape
    raw_acts_2d = raw_acts.reshape(-1, d_model_raw).to(device).to(torch.float32)
    sae_acts_list = []
    with torch.no_grad():
        for i in range(0, raw_acts_2d.shape[0], batch_size):
            batch_slice = raw_acts_2d[i:i + batch_size]
            acts_2d, _, _ = sae(batch_slice)
            sae_acts_list.append(acts_2d.cpu())
    return torch.cat(sae_acts_list, dim=0).reshape(N, max_seq, d_hidden).numpy()


def build_ladder(input_stats, raw_agg, sae_agg):
    """Assemble the five feature matrices from the three building blocks."""
    return {
        P1: input_stats,
        P2: np.concatenate([input_stats, raw_agg], axis=1),
        P3: np.concatenate([input_stats, sae_agg], axis=1),
        P4: raw_agg,
        P5: sae_agg,
    }


def make_cv_splits(df_meta, y_train, train_mask):
    """Build inner-CV folds over the TRAIN rows.

    Stratifies by a composite (topic, label) key so folds are balanced by topic
    as well as correctness; falls back to plain stratified-by-label CV when any
    topic-label group is too small to appear in every fold.
    """
    n_splits = max(2, min(5, int(np.bincount(y_train).min()) - 1, train_mask.sum() // 10))

    # Composite stratification key: activity_label (topic) x correctness label.
    strat_key = df_meta.loc[train_mask, "activity_label"].astype(str) + "_" + y_train.astype(str)
    counts = strat_key.value_counts()

    if (counts < n_splits).any():
        print(f"  CV: Using stratified-by-label CV (some topic-label groups have count < {n_splits})")
        cv_target = y_train
    else:
        print(f"  CV: Using stratified-by-topic CV across {n_splits} folds")
        cv_target = strat_key.values

    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(cv_splitter.split(np.zeros((len(y_train), 1)), cv_target))


def fit_ladder(probes, y_train, y_test, train_mask, test_mask, splits):
    """Fit every rung with C chosen by inner CV.

    Returns ``results`` ({rung: {"AUROC", "best_C"}}) and ``preds`` ({rung:
    test-set P(hard) predictions}) for the downstream bootstrap.
    """
    results = {}
    preds = {}
    for name, X in probes.items():
        print(f"Training probe: {name} (features: {X.shape[1]})")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_mask])
        X_test_s = scaler.transform(X[test_mask])

        base = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000)
        gs = GridSearchCV(base, C_GRID, scoring="roc_auc", cv=splits, n_jobs=-1)

        # threading backend avoids fork deadlocks on Apple Silicon.
        with parallel_backend("threading"):
            gs.fit(X_train_s, y_train)

        preds[name] = gs.predict_proba(X_test_s)[:, 1]
        point = roc_auc_score(y_test, preds[name]) if len(np.unique(y_test)) > 1 else 0.0
        results[name] = {"AUROC": point, "best_C": gs.best_params_["C"]}
        print(f"  {name} point AUROC = {point:.3f}  (C={gs.best_params_['C']})")
    return results, preds


def paired_bootstrap(preds, y_test, names, pairs, n_boot=2000, seed=42):
    """Paired bootstrap of per-rung AUROC CIs and per-pair delta CIs.

    Resampling the SAME test indices for every rung (paired) keeps the delta
    CIs honest about the correlation between rungs. Returns ``auroc_ci``
    ({rung: (lo, hi)}) and ``delta_ci`` ({f"{a}-{b}": (lo, hi)}).
    """
    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    boot_delta = {f"{a}-{b}": [] for a, b in pairs}
    idx_all = np.arange(len(y_test))

    if len(np.unique(y_test)) > 1:
        for _ in range(n_boot):
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

    auroc_ci = {n: _ci(boot[n]) for n in names}
    delta_ci = {key: _ci(vals) for key, vals in boot_delta.items()}
    return auroc_ci, delta_ci


def assemble_results(df_meta, train_mask, test_mask, y, results, deltas):
    """Flatten the per-rung results + headline deltas into the JSON payload.

    ``deltas`` maps each headline name ("raw", "sae", "sae_over_raw") to a
    ``(point, (ci_lo, ci_hi))`` tuple.
    """
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    hard_fraction = float(y[test_mask].mean()) if n_test > 0 else 0.0

    (delta_raw, d_raw_ci) = deltas["raw"]
    (delta_sae, d_sae_ci) = deltas["sae"]
    (delta_sae_over_raw, d_sor_ci) = deltas["sae_over_raw"]

    return {
        "n_total": len(df_meta),
        "n_train": n_train,
        "n_test": n_test,
        "hard_fraction": hard_fraction,
        "P1_AUROC": results.get(P1, {}).get("AUROC", 0.0),
        "P1_CI_lower": results.get(P1, {}).get("95%_CI_lower", 0.0),
        "P1_CI_upper": results.get(P1, {}).get("95%_CI_upper", 0.0),
        "P2_AUROC": results.get(P2, {}).get("AUROC", 0.0),
        "P2_CI_lower": results.get(P2, {}).get("95%_CI_lower", 0.0),
        "P2_CI_upper": results.get(P2, {}).get("95%_CI_upper", 0.0),
        "P3_AUROC": results.get(P3, {}).get("AUROC", 0.0),
        "P3_CI_lower": results.get(P3, {}).get("95%_CI_lower", 0.0),
        "P3_CI_upper": results.get(P3, {}).get("95%_CI_upper", 0.0),
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
        "P4_RawOnly_AUROC": results.get(P4, {}).get("AUROC", 0.0),
        "P4_RawOnly_CI_lower": results.get(P4, {}).get("95%_CI_lower", 0.0),
        "P4_RawOnly_CI_upper": results.get(P4, {}).get("95%_CI_upper", 0.0),
        "P5_SAEOnly_AUROC": results.get(P5, {}).get("AUROC", 0.0),
        "P5_SAEOnly_CI_lower": results.get(P5, {}).get("95%_CI_lower", 0.0),
        "P5_SAEOnly_CI_upper": results.get(P5, {}).get("95%_CI_upper", 0.0),
        "chosen_C": {k: v.get("best_C") for k, v in results.items() if "best_C" in v},
    }


def save_scores(df_meta, test_mask, preds, scores_parquet):
    """Persist per-test-row P(hard) predictions for each rung."""
    df_test = df_meta[test_mask].copy()
    for name, p in preds.items():
        df_test[f"pred_{name}"] = p
    os.makedirs(os.path.dirname(os.path.abspath(scores_parquet)), exist_ok=True)
    df_test.to_parquet(scores_parquet)
    print(f"\nSaved {scores_parquet}")


def save_results_json(final_results, results_json):
    os.makedirs(os.path.dirname(os.path.abspath(results_json)), exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Saved {results_json}")


def main():
    args = parse_args()
    device = select_device()

    print("Loading HellaSwag metadata and activations...")
    df_meta, raw_acts = load_dataset(args.metadata, args.activations)

    print("Computing prompt statistics...")
    input_stats = compute_prompt_stats(df_meta)

    print("Loading SAE...")
    sae, d_hidden = load_sae(args.sae_ckpt, args.k, device)

    print("Aggregating activations (padding-aware sequence pooling)...")
    raw_agg = aggregate_sequence(raw_acts.numpy(), df_meta)

    sae_acts = encode_sae_codes(sae, raw_acts, device, d_hidden)
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

    probes = build_ladder(input_stats, raw_agg, sae_agg)
    splits = make_cv_splits(df_meta, y_train, train_mask)
    results, preds = fit_ladder(probes, y_train, y_test, train_mask, test_mask, splits)

    # PAIRED bootstrap to compute 95% CIs and deltas
    print("Running paired bootstrap (B=2000)...")
    names = list(probes.keys())
    auroc_ci, delta_ci = paired_bootstrap(preds, y_test, names, HEADLINE_PAIRS)

    for n in names:
        lo, hi = auroc_ci[n]
        results[n]["95%_CI_lower"] = lo
        results[n]["95%_CI_upper"] = hi
        print(f"  {n} AUROC 95% CI: [{lo:.3f}, {hi:.3f}]")

    delta_raw = results[P2]["AUROC"] - results[P1]["AUROC"]
    delta_sae = results[P3]["AUROC"] - results[P1]["AUROC"]
    delta_sae_over_raw = results[P3]["AUROC"] - results[P2]["AUROC"]
    d_raw_ci = delta_ci[f"{P2}-{P1}"]
    d_sae_ci = delta_ci[f"{P3}-{P1}"]
    d_sor_ci = delta_ci[f"{P3}-{P2}"]

    print("\n--- Incremental Predictive Power (ΔAUROC, paired bootstrap) ---")
    print(f"Δ Raw - Stats : {delta_raw:+.3f}  95% CI [{d_raw_ci[0]:+.3f}, {d_raw_ci[1]:+.3f}]")
    print(f"Δ SAE - Stats : {delta_sae:+.3f}  95% CI [{d_sae_ci[0]:+.3f}, {d_sae_ci[1]:+.3f}]")
    print(f"Δ SAE - Raw   : {delta_sae_over_raw:+.3f}  95% CI [{d_sor_ci[0]:+.3f}, {d_sor_ci[1]:+.3f}]")

    save_scores(df_meta, test_mask, preds, args.scores_parquet)

    deltas = {
        "raw": (delta_raw, d_raw_ci),
        "sae": (delta_sae, d_sae_ci),
        "sae_over_raw": (delta_sae_over_raw, d_sor_ci),
    }
    final_results = assemble_results(df_meta, train_mask, test_mask, y, results, deltas)
    save_results_json(final_results, args.results_json)


if __name__ == "__main__":
    main()
