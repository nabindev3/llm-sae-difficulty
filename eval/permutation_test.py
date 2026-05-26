"""One-sided label-permutation test for Δ(SAE − Raw) AUROC deltas.

Loads probe scores and labels, computes the observed Δ(AUROC_P3 − AUROC_P2),
then shuffles the labels B times and recomputes the delta under the null
hypothesis that both probe outputs are independent of the true difficulty
label. Reports the one-sided p-value (in the observed direction), the
two-sided p-value, and the empirical null distribution.

This complements the existing paired-bootstrap CI by giving a clean p-value
for "is the observed delta unlikely under the null?".
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_scores", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--score_a", required=True,
                    help="Numerator probe column (e.g. pred_P3_InputStats_SAE).")
    ap.add_argument("--score_b", required=True,
                    help="Reference probe column (e.g. pred_P2_InputStats_Raw).")
    ap.add_argument("--label_col", default="difficulty")
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=10000, help="Permutation count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    scores = pd.read_parquet(args.probe_scores)
    meta = pd.read_parquet(args.metadata)

    # Some probe_scores parquets already only contain the test split; some include all.
    if "split" in scores.columns:
        scores = scores[scores["split"] == args.split].reset_index(drop=True)
    if args.label_col not in scores.columns:
        # Merge label from metadata on window_id
        meta_test = meta[meta["split"] == args.split][["window_id", args.label_col]]
        scores = scores.merge(meta_test, on="window_id")

    for c in [args.score_a, args.score_b]:
        if c not in scores.columns:
            raise SystemExit(f"Column {c!r} not in probe_scores. Available: "
                             f"{[c for c in scores.columns if c.startswith('pred_')]}")

    y = scores[args.label_col].values.astype(int)
    pa = scores[args.score_a].values.astype(float)
    pb = scores[args.score_b].values.astype(float)

    if len(np.unique(y)) < 2:
        raise SystemExit("Need both classes in the test set for AUROC.")

    auc_a = roc_auc_score(y, pa)
    auc_b = roc_auc_score(y, pb)
    obs_delta = auc_a - auc_b

    rng = np.random.default_rng(args.seed)
    perm_deltas = np.empty(args.B, dtype=np.float64)
    for i in range(args.B):
        y_perm = rng.permutation(y)
        try:
            a = roc_auc_score(y_perm, pa)
            b = roc_auc_score(y_perm, pb)
        except ValueError:
            perm_deltas[i] = 0.0
            continue
        perm_deltas[i] = a - b

    if obs_delta < 0:
        p_one = float((perm_deltas <= obs_delta).mean())
        direction = f"SAE < Raw ({args.score_a} < {args.score_b})"
    else:
        p_one = float((perm_deltas >= obs_delta).mean())
        direction = f"SAE > Raw ({args.score_a} > {args.score_b})"
    p_two = float((np.abs(perm_deltas) >= abs(obs_delta)).mean())

    null_mean = float(perm_deltas.mean())
    null_std = float(perm_deltas.std(ddof=1))
    null_p025 = float(np.percentile(perm_deltas, 2.5))
    null_p975 = float(np.percentile(perm_deltas, 97.5))

    summary = {
        "score_a": args.score_a,
        "score_b": args.score_b,
        "n_test": int(len(y)),
        "auroc_a": float(auc_a),
        "auroc_b": float(auc_b),
        "observed_delta": float(obs_delta),
        "direction": direction,
        "B": args.B,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_2p5_pct": null_p025,
        "null_97p5_pct": null_p975,
    }

    print(f"=== Permutation test ({args.score_a} − {args.score_b}) ===")
    print(f"  n_test = {len(y)}, B = {args.B}")
    print(f"  AUROC({args.score_a}) = {auc_a:.4f}")
    print(f"  AUROC({args.score_b}) = {auc_b:.4f}")
    print(f"  Observed delta = {obs_delta:+.4f}  ({direction})")
    print(f"  Null distribution: mean={null_mean:+.4f}  sd={null_std:.4f}  "
          f"95% range [{null_p025:+.4f}, {null_p975:+.4f}]")
    print(f"  One-sided p-value = {p_one:.4g}")
    print(f"  Two-sided p-value = {p_two:.4g}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved {args.out_json}")


if __name__ == "__main__":
    main()
