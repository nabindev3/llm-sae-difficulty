"""Stronger cheap baseline: max-softmax, logit-entropy, embedding-distance-to-train.

"The harder the baseline, the more the null means." The committed P1 is 8 lexical
stats. This augments it with three difficulty-predictive signals that don't need
the SAE:
  * max_softmax    -- model confidence (max next-token prob) at the boundary
  * logit_entropy  -- entropy of the next-token distribution
  * emb_dist_train -- cosine distance from the prompt's boundary activation to its
                      nearest TRAIN activation (novelty / distribution shift)

It runs the exact committed ladder (probe.build_ladder/fit_ladder/paired_bootstrap)
twice -- weak P1 (8 stats) vs strong P1 (11) -- and reports whether raw/SAE still
add over the stronger baseline, and crucially whether the SAE-vs-Raw null holds.

max_softmax/logit_entropy are read from metadata (produced by extract_layersweep.py);
emb_dist_train is computed here from the activations.

Usage:
    python eval/strong_baseline.py \
        --activations eval/results/layersweep/_work/layer12_acts.safetensors \
        --metadata activations_layersweep/squad_metadata.parquet \
        --sae_ckpt eval/results/layersweep/_work/layer12_sae/sae_topk_32.pt \
        --out_json eval/results/strong_baseline/squad_l12.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probing.features import compute_prompt_stats, aggregate_sequence
from probing.probe import (build_ladder, make_cv_splits, fit_ladder, paired_bootstrap,
                           load_sae, encode_sae_codes, select_device, HEADLINE_PAIRS,
                           P1, P2, P3, P4, P5)


def emb_distance_to_train(raw_agg, train_mask):
    """Cosine distance (1 - max sim) from each row to the nearest TRAIN row.

    Leave-one-out for train rows: a train point's nearest train neighbour is
    itself (distance 0), which would make the feature constant on train and blow
    up under StandardScaler. We mask the self-match so train rows use their
    nearest OTHER train point; test rows use the full train set.
    """
    Xs = StandardScaler().fit_transform(raw_agg)
    train_X = Xs[train_mask]
    pos_in_train = np.full(len(Xs), -1, dtype=int)
    pos_in_train[np.where(train_mask)[0]] = np.arange(train_X.shape[0])
    out = np.empty(len(Xs), dtype=np.float64)
    step = 512
    for i in range(0, len(Xs), step):
        sims = cosine_similarity(Xs[i:i + step], train_X)   # (chunk, n_train)
        for r in range(sims.shape[0]):
            p = pos_in_train[i + r]
            if p >= 0:                                       # train row -> exclude self
                sims[r, p] = -np.inf
        out[i:i + step] = 1.0 - sims.max(axis=1)
    return out.reshape(-1, 1)


def run_ladder(input_stats, raw_agg, sae_agg, df_meta, y, train_mask, test_mask, seed=42):
    probes = build_ladder(input_stats, raw_agg, sae_agg)
    splits = make_cv_splits(df_meta, y[train_mask], train_mask, seed=seed)
    results, preds = fit_ladder(probes, y[train_mask], y[test_mask], train_mask, test_mask, splits)
    names = list(probes.keys())
    _, delta_ci = paired_bootstrap(preds, y[test_mask], names, HEADLINE_PAIRS, seed=seed)
    return results, delta_ci


def summarize(tag, results, delta_ci):
    return {
        "tag": tag,
        "P1_AUROC": results[P1]["AUROC"], "P2_AUROC": results[P2]["AUROC"],
        "P3_AUROC": results[P3]["AUROC"], "P4_RawOnly_AUROC": results[P4]["AUROC"],
        "P5_SAEOnly_AUROC": results[P5]["AUROC"],
        "delta_raw_over_stats": results[P2]["AUROC"] - results[P1]["AUROC"],
        "delta_raw_over_stats_CI": list(delta_ci[f"{P2}-{P1}"]),
        "delta_sae_over_raw": results[P3]["AUROC"] - results[P2]["AUROC"],
        "delta_sae_over_raw_CI": list(delta_ci[f"{P3}-{P2}"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--sae_ckpt", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    device = select_device()

    df = pd.read_parquet(args.metadata)
    raw_acts = load_file(args.activations)["encoder_embeddings"]
    y = df["difficulty"].values
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    base_stats = compute_prompt_stats(df)                  # (N, 8)
    raw_agg = aggregate_sequence(raw_acts.numpy(), df)
    sae, d_hidden = load_sae(args.sae_ckpt, args.k, device)
    sae_agg = aggregate_sequence(encode_sae_codes(sae, raw_acts, device, d_hidden), df)

    # Strong extra features: confidence + entropy (from metadata) + emb distance.
    for col in ("max_softmax", "logit_entropy"):
        if col not in df.columns:
            sys.exit(f"[strong_baseline] metadata missing '{col}'; re-extract with extract_layersweep.py")
    extra = np.column_stack([
        df["max_softmax"].to_numpy(float),
        df["logit_entropy"].to_numpy(float),
        emb_distance_to_train(raw_agg, train_mask).ravel(),
    ])
    strong_stats = np.concatenate([base_stats, extra], axis=1)

    print("Running WEAK baseline (8 stats)...")
    w_res, w_dci = run_ladder(base_stats, raw_agg, sae_agg, df, y, train_mask, test_mask)
    print("Running STRONG baseline (8 + max_softmax + entropy + emb_dist)...")
    s_res, s_dci = run_ladder(strong_stats, raw_agg, sae_agg, df, y, train_mask, test_mask)

    out = {"weak": summarize("weak_P1_8stats", w_res, w_dci),
           "strong": summarize("strong_P1_11feats", s_res, s_dci),
           "n_extra_features": 3}
    # The strong baseline only matters if it actually raised P1.
    out["P1_gain_from_strong"] = out["strong"]["P1_AUROC"] - out["weak"]["P1_AUROC"]
    out["null_holds_strong"] = bool(out["strong"]["delta_sae_over_raw_CI"][0] <= 0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    for k in ("weak", "strong"):
        c = out[k]
        print(f"\n[{k}] P1={c['P1_AUROC']:.3f} P2(stats+raw)={c['P2_AUROC']:.3f} "
              f"P4raw={c['P4_RawOnly_AUROC']:.3f} P5sae={c['P5_SAEOnly_AUROC']:.3f}")
        print(f"   Δ(Raw-Stats)={c['delta_raw_over_stats']:+.4f} {[round(x,3) for x in c['delta_raw_over_stats_CI']]}"
              f"   Δ(SAE-Raw)={c['delta_sae_over_raw']:+.4f} {[round(x,3) for x in c['delta_sae_over_raw_CI']]}")
    print(f"\nP1 gain from strengthening: {out['P1_gain_from_strong']:+.4f}; "
          f"SAE-vs-Raw null holds under strong baseline: {out['null_holds_strong']}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
