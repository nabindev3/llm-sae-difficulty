"""Feature-routed cascade: route between a cheap (Pythia-410M) and an expensive (Pythia-2.8B) LLM
using a trained difficulty probe as the routing signal.

We sweep the routing threshold τ in [0, 1] and plot the resulting Pareto curve on
(mean inference cost, mean error rate). Compared against:
  - **always cheap**  : Pythia-410M, cost=1.0
  - **always base**   : Pythia-2.8B,  cost=5.0
  - **random routing**: average over permutations at each routing fraction
  - **oracle routing**: route to base the prompts where Pythia-2.8B actually helps 
                        most (i.e. base is correct but small is incorrect).
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _route_threshold(scores, tau, err_cheap, err_base, cost_cheap, cost_base):
    to_base = scores >= tau
    final = np.where(to_base, err_base, err_cheap)
    cost = np.where(to_base, cost_base, cost_cheap)
    return float(final.mean()), float(cost.mean()), float(to_base.mean())


def _route_random(err_cheap, err_base, cost_cheap, cost_base, n_trials=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(err_cheap)
    fractions = np.linspace(0.0, 1.0, 21)
    curve = []
    for f in fractions:
        k = int(round(f * n))
        errors, costs = [], []
        for _ in range(n_trials):
            idx = rng.choice(n, size=k, replace=False) if k > 0 else np.array([], dtype=int)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            errors.append(np.where(mask, err_base, err_cheap).mean())
            costs.append(np.where(mask, cost_base, cost_cheap).mean())
        curve.append((float(np.mean(costs)), float(np.mean(errors)), float(f)))
    return curve


def _route_oracle(err_cheap, err_base, cost_cheap, cost_base):
    # Route to base first when base is correct (0 error) and cheap is incorrect (1 error).
    # Gap is err_cheap - err_base:
    # 1 - 0 = +1 (base helps most)
    # 0 - 0 =  0 (both correct, no benefit)
    # 1 - 1 =  0 (both incorrect, no benefit)
    # 0 - 1 = -1 (base is worse, negative benefit)
    gap = err_cheap - err_base
    order = np.argsort(-gap) # descending: +1 first, then 0s, then -1s
    n = len(gap)
    fractions = np.linspace(0.0, 1.0, 21)
    curve = []
    for f in fractions:
        k = int(round(f * n))
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        err = np.where(mask, err_base, err_cheap).mean()
        cost = np.where(mask, cost_base, cost_cheap).mean()
        curve.append((float(cost), float(err), float(f)))
    return curve


def _probe_curve(scores, err_cheap, err_base, cost_cheap, cost_base, n_taus=41):
    taus = np.linspace(0.0, 1.0, n_taus)
    pts = []
    for tau in taus:
        err, cost, frac = _route_threshold(scores, tau, err_cheap, err_base,
                                            cost_cheap, cost_base)
        pts.append({"tau": float(tau), "frac_to_base": frac,
                    "mean_cost": cost, "mean_error": err})
    return pts


def _dominating_points(pts, cheap_anchor, base_anchor):
    """Count probe-driven Pareto points strictly below the cheap↔base interpolation line."""
    c0, y0 = cheap_anchor
    c1, y1 = base_anchor
    dom = []
    for p in pts:
        c = p["mean_cost"]
        if not (c0 < c < c1):
            continue
        t = (c - c0) / (c1 - c0 + 1e-12)
        y_line = y0 + t * (y1 - y0)
        if p["mean_error"] < y_line - 1e-9:
            dom.append(p)
    return dom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small_metadata", type=str, default="activations/hellaswag_metadata.parquet")
    p.add_argument("--base_metadata", type=str, default="activations_base/hellaswag_metadata.parquet")
    p.add_argument("--probe_scores", type=str, default="activations/probe_scores.parquet")
    p.add_argument("--score_cols", type=str, nargs="+",
                   default=["pred_P3_InputStats_SAE", "pred_P1_InputStats"],
                   help="Probe score columns to evaluate as routing signals.")
    p.add_argument("--cost_cheap", type=float, default=1.0)
    p.add_argument("--cost_base", type=float, default=5.0)
    p.add_argument("--output_dir", type=str, default="eval/results")
    p.add_argument("--n_random_trials", type=int, default=500)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for path in (args.small_metadata, args.base_metadata, args.probe_scores):
        if not os.path.exists(path):
            raise SystemExit(f"[cascade] missing input: {path}. Run both extractions and probe first.")

    df_small = pd.read_parquet(args.small_metadata)[["window_id", "difficulty"]].rename(
        columns={"difficulty": "err_small"}
    )
    df_base = pd.read_parquet(args.base_metadata)[["window_id", "difficulty"]].rename(
        columns={"difficulty": "err_base"}
    )
    df_probe = pd.read_parquet(args.probe_scores)
    
    if "window_id" not in df_probe.columns:
        raise SystemExit("[cascade] probe_scores missing 'window_id' column.")

    available = [c for c in args.score_cols if c in df_probe.columns]
    missing = [c for c in args.score_cols if c not in df_probe.columns]
    if missing:
        print(f"[cascade] WARNING: missing score columns in probe_scores: {missing}")
    if not available:
        raise SystemExit("[cascade] no requested score columns present in probe_scores.")

    keep_cols = ["window_id"] + available
    df = df_probe[keep_cols].merge(df_small, on="window_id").merge(df_base, on="window_id")
    if len(df) == 0:
        raise SystemExit("[cascade] zero-row join — window_ids don't overlap.")
        
    n = len(df)
    err_small = df["err_small"].values.astype(float)
    err_base = df["err_base"].values.astype(float)
    
    print(f"[cascade] evaluating on {n} test prompts")
    print(f"  mean error small = {err_small.mean():.4f}")
    print(f"  mean error base  = {err_base.mean():.4f}")
    print(f"  win rate base    = {(err_base < err_small).mean():.2%}  (fraction where base beats small)")

    cheap_anchor = (args.cost_cheap, float(err_small.mean()))
    base_anchor = (args.cost_base,  float(err_base.mean()))

    # Baselines
    random_curve = _route_random(err_small, err_base, args.cost_cheap,
                                 args.cost_base, n_trials=args.n_random_trials)
    oracle_curve = _route_oracle(err_small, err_base, args.cost_cheap, args.cost_base)

    # Probe-driven curves
    probe_pts = {}
    summary = {
        "n_windows": n,
        "always_cheap": {"mean_error": cheap_anchor[1], "cost": cheap_anchor[0]},
        "always_base":  {"mean_error": base_anchor[1],  "cost": base_anchor[0]},
        "win_rate_base": float((err_base < err_small).mean()),
        "random_curve": random_curve,
        "oracle_curve": oracle_curve,
        "probes": {}
    }
    
    for col in available:
        pts = _probe_curve(df[col].values, err_small, err_base,
                            args.cost_cheap, args.cost_base)
        dom = _dominating_points(pts, cheap_anchor, base_anchor)
        best_dom = min(dom, key=lambda p: p["mean_error"]) if dom else None
        probe_pts[col] = pts
        summary["probes"][col] = {
            "frontier": pts,
            "n_dominating_points": len(dom),
            "best_dominating": best_dom,
        }
        print(f"  {col}: {len(dom)} Pareto-dominating points  "
              f"(best: {best_dom})" if dom else f"  {col}: 0 dominating points")

    # Save JSON
    with open(os.path.join(args.output_dir, "cascade_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([cheap_anchor[0], base_anchor[0]],
            [cheap_anchor[1], base_anchor[1]],
            color="gray", linestyle=":", label="linear interp (random equiv.)")
            
    rx, ry = [c for c, _, _ in random_curve], [y for _, y, _ in random_curve]
    ox, oy = [c for c, _, _ in oracle_curve], [y for _, y, _ in oracle_curve]
    ax.plot(rx, ry, color="#999", linestyle="--", linewidth=1.5,
            label=f"random routing (500-trial avg)")
    ax.plot(ox, oy, color="black", linewidth=2,
            label="oracle (best choice)")

    colors = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#72b7b2"]
    for col, c in zip(available, colors):
        cx = [pt["mean_cost"] for pt in probe_pts[col]]
        cy = [pt["mean_error"] for pt in probe_pts[col]]
        ax.plot(cx, cy, color=c, marker="o", markersize=4,
                label=f"routed by {col.replace('pred_', '')}")

    ax.scatter([cheap_anchor[0]], [cheap_anchor[1]], color="#4c78a8",
               s=80, zorder=6, edgecolor="black", label="always cheap (pythia-410m)")
    ax.scatter([base_anchor[0]], [base_anchor[1]], color="#e45756",
               s=80, zorder=6, edgecolor="black", label="always base (pythia-2.8b)")
               
    ax.set_xlabel(f"Mean inference cost  (cheap={args.cost_cheap}, base={args.cost_base})")
    ax.set_ylabel("Mean Error Rate on test (lower better)")
    ax.set_title("Feature-routed cascade: pythia-410m  ↔  pythia-2.8b")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    
    fig.savefig(os.path.join(args.output_dir, "pareto_frontier.png"), dpi=150)
    print(f"\nSaved {os.path.join(args.output_dir, 'pareto_frontier.png')}")
    print(f"Saved {os.path.join(args.output_dir, 'cascade_results.json')}")


if __name__ == "__main__":
    main()
