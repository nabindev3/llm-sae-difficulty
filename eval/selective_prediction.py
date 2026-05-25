"""Selective answering / risk-coverage analysis for the LLM Bridge.

Even if SAE features don't add predictive power, the P1 stats probe serves
as a usable selective answering signal: abstaining on prompts with high predicted
error probability (difficulty) and answering only on easy ones decreases the average
error rate.

Risk = Error rate (fraction of incorrect answers on retained prompts).
Headline number: AURC (area under the risk-coverage curve, lower is better)
and error reduction at 50% coverage.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROBE_LABELS = {
    "pred_P1_InputStats":     ("P1 stats",       "#4c78a8"),
    "pred_P2_InputStats_Raw": ("P2 stats+raw",   "#f58518"),
    "pred_P3_InputStats_SAE": ("P3 stats+sae",   "#e45756"),
    "pred_P4_RawOnly":        ("P4 raw only",    "#72b7b2"),
    "pred_P5_SAEOnly":        ("P5 sae only",    "#54a24b"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_scores", default="activations/probe_scores.parquet")
    ap.add_argument("--metadata", default="activations/hellaswag_metadata.parquet")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for p in (args.probe_scores, args.metadata):
        if not os.path.exists(p):
            sys.exit(f"[selective] missing: {p}. Run the full pipeline first.")
            
    os.makedirs(args.out_dir, exist_ok=True)

    scores = pd.read_parquet(args.probe_scores)
    meta = pd.read_parquet(args.metadata)
    
    # Merge on window_id to pair predictions with true correctness labels
    test_meta = meta[meta["split"] == "test"][["window_id", "difficulty"]].rename(
        columns={"difficulty": "error_label"}
    )
    df = scores.merge(test_meta, on="window_id")
    if len(df) == 0:
        sys.exit("[selective] no overlap between probe_scores and test metadata.")

    probe_cols = [c for c in df.columns if c.startswith("pred_")]
    if not probe_cols:
        sys.exit("[selective] no pred_* columns in probe_scores.")

    n = len(df)
    errors_all = df["error_label"].values.astype(float)
    mean_err_all = float(errors_all.mean())
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 4)

    # Oracle: sort by TRUE correctness ascending (easy/correct first, i.e. 0s first, then 1s)
    sorted_truth = np.sort(errors_all)
    oracle_curve = np.array([
        sorted_truth[:max(1, int(round(c * n)))].mean() for c in coverages
    ])

    # Random baseline: averaged over many random prompt selections
    rand_curves = []
    for _ in range(args.n_bootstrap):
        perm = rng.permutation(n)
        rand_errs = errors_all[perm]
        rand_curves.append([rand_errs[:max(1, int(round(c * n)))].mean()
                            for c in coverages])
    rand_curves = np.array(rand_curves)
    random_curve = rand_curves.mean(axis=0)

    results = {}
    for col in probe_cols:
        order = np.argsort(df[col].values)        # ascending predicted P(hard)
        sorted_errors = errors_all[order]
        curve, lo, hi = [], [], []
        for c in coverages:
            k = max(1, int(round(c * n)))
            kept = sorted_errors[:k]
            curve.append(float(kept.mean()))
            
            # Bootstrap CI of error rate at this coverage
            boots = [kept[rng.integers(0, k, k)].mean()
                     for _ in range(args.n_bootstrap)]
            lo.append(float(np.percentile(boots, 2.5)))
            hi.append(float(np.percentile(boots, 97.5)))
            
        results[col] = {
            "curve": curve, 
            "ci95_lower": lo, 
            "ci95_upper": hi,
            "aurc": float(np.trapezoid(curve, coverages))
        }

    oracle_aurc = float(np.trapezoid(oracle_curve, coverages))
    random_aurc = float(np.trapezoid(random_curve, coverages))

    # Error rate reduction at 50% coverage
    i50 = int(np.argmin(np.abs(coverages - 0.5)))

    summary = {
        "n_test": n,
        "mean_error_no_abstention": mean_err_all,
        "coverages": coverages.tolist(),
        "oracle_curve": oracle_curve.tolist(),
        "oracle_aurc": oracle_aurc,
        "random_curve": random_curve.tolist(),
        "random_aurc": random_aurc,
        "probes": results,
        "at_coverage_0p5": {
            "no_abstention": mean_err_all,
            "oracle": float(oracle_curve[i50]),
            "random": float(random_curve[i50]),
            **{col: {
                "mean_error": results[col]["curve"][i50],
                "reduction_pct": 100 * (mean_err_all - results[col]["curve"][i50]) / (mean_err_all + 1e-8)
               } for col in probe_cols},
        },
    }
    
    with open(os.path.join(args.out_dir, "selective_prediction.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axhline(mean_err_all, color="gray", linestyle=":",
               label=f"No abstention ({mean_err_all:.3f})")
    ax.plot(coverages, random_curve, color="black", linestyle="--",
            label=f"Random (AURC {random_aurc:.3f})")
    ax.plot(coverages, oracle_curve, color="black", linestyle="-",
            linewidth=2, label=f"Oracle (AURC {oracle_aurc:.3f})")
            
    for col in probe_cols:
        lbl, c = PROBE_LABELS.get(col, (col, "purple"))
        ax.plot(coverages, results[col]["curve"], color=c, marker="o",
                markersize=4,
                label=f"{lbl} (AURC {results[col]['aurc']:.3f})")
        ax.fill_between(coverages, results[col]["ci95_lower"],
                        results[col]["ci95_upper"], alpha=0.12, color=c)
                        
    ax.set_xlabel("Coverage (fraction of validation prompts answered)")
    ax.set_ylabel("Mean Error Rate on retained prompts (lower better)")
    ax.set_title("Selective answering on HellaSwag — pythia-410m")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    
    out_png = os.path.join(args.out_dir, "risk_coverage.png")
    fig.savefig(out_png, dpi=150)

    # Console summary
    print(f"n_test = {n}")
    print(f"No-abstention error rate = {mean_err_all:.4f}")
    print(f"Oracle AURC              = {oracle_aurc:.4f}")
    print(f"Random AURC              = {random_aurc:.4f}")
    print("Probe AURCs (lower better):")
    for col in probe_cols:
        print(f"  {PROBE_LABELS.get(col, (col, ''))[0]:16s}  "
              f"AURC = {results[col]['aurc']:.4f}")
              
    print(f"\nAt 50% coverage:")
    for col in probe_cols:
        d = summary["at_coverage_0p5"][col]
        print(f"  {PROBE_LABELS.get(col, (col, ''))[0]:16s}  "
              f"mean error = {d['mean_error']:.4f}  "
              f"(ΔError = {-d['reduction_pct']:+.1f}% vs no abstention)")
    print(f"  {'Oracle':16s}  mean error = {oracle_curve[i50]:.4f}")
    print(f"  {'Random':16s}  mean error = {random_curve[i50]:.4f}")
    print(f"\nSaved {out_png}\nSaved {os.path.join(args.out_dir, 'selective_prediction.json')}")


if __name__ == "__main__":
    main()
