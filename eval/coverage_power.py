"""Causal-intervention COVERAGE sweep + power analysis.

The paper asserts "detectable per-feature SAE causal effects come from
intervention coverage, not SAE fidelity." Reviewers want this *quantified*, not
asserted. This sweeps the number of patched prompt positions K (via
causal_ablation.py --coverage K) and, at each K, measures:

  * the aggregate reconstruction penalty delta_recon_natural(K) (dose-response), and
  * per top-5 feature: effect size e_K and bootstrap CI.

Then a power analysis. From the bootstrap CI, the standard error is
  sigma_K ~ (CI_upper - CI_lower) / (2 * 1.96),
and the two-sided power to detect the measured effect at alpha=0.05 is
  power_K = Phi(e_K/sigma_K - 1.96) + Phi(-e_K/sigma_K - 1.96).
Plotting power_K vs K turns "coverage drives detectability" into a curve with
an explicit coverage threshold for 80% power.

Runs on the all-position SAE (sae/checkpoints_allpos_hellaswag) + all-position
activations, HellaSwag, continuous metric. Resumable per K.

Usage:
    python eval/coverage_power.py --ks 1 2 4 8 16 32 64
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE = "sae/checkpoints_allpos_hellaswag/sae_topk_32.pt"
ACTS = "activations_allpos/hellaswag_activations.safetensors"
META = "activations_allpos/hellaswag_metadata.parquet"


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-3000:] + p.stderr[-3000:])
        raise RuntimeError(f"failed: {' '.join(cmd)}")
    return p.stdout


def two_sided_power(effect, se, alpha=0.05):
    """Power to reject H0: effect=0 given measured |effect| and se."""
    if se <= 0:
        return float("nan")
    z = abs(effect) / se
    crit = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z - crit) + norm.cdf(-z - crit))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out_json", default="eval/results/coverage/coverage_power.json")
    args = ap.parse_args()
    out_path = os.path.join(ROOT, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = json.load(open(out_path)) if os.path.exists(out_path) else {"by_k": {}}

    for K in args.ks:
        if str(K) in summary["by_k"]:
            continue
        kdir = os.path.join("eval/results/coverage", f"k{K}")
        print(f"=== coverage K={K} ===", flush=True)
        run([args.python, "eval/causal_ablation.py", "--dataset", "hellaswag",
             "--metric", "continuous", "--coverage", str(K),
             "--hellaswag_boundary", "last_prompt",
             "--sae_ckpt", SAE, "--activations", ACTS, "--metadata", META,
             "--out_dir", kdir])
        r = json.load(open(os.path.join(ROOT, kdir, "hellaswag_causal_ablation.json")))
        feats = {}
        for f, fe in r["feature_effects"].items():
            e = fe["delta_error"]
            se = (fe["ci_upper"] - fe["ci_lower"]) / (2 * 1.959964)
            feats[f] = {"effect": e, "ci": [fe["ci_lower"], fe["ci_upper"]],
                        "se": se, "power": two_sided_power(e, se),
                        "detected": bool(fe["ci_lower"] > 0 or fe["ci_upper"] < 0)}
        summary["by_k"][str(K)] = {
            "K": K,
            "delta_recon_natural": r["delta_recon_natural"],
            "delta_recon_natural_ci": [r["delta_recon_natural_ci_lower"], r["delta_recon_natural_ci_upper"]],
            "n_features_detected": sum(v["detected"] for v in feats.values()),
            "mean_power": float(np.nanmean([v["power"] for v in feats.values()])),
            "features": feats,
        }
        json.dump(summary, open(out_path, "w"), indent=2)
        s = summary["by_k"][str(K)]
        print(f"  K={K}: recon_penalty={s['delta_recon_natural']:+.4f}  "
              f"features_detected={s['n_features_detected']}/{len(feats)}  "
              f"mean_power={s['mean_power']:.3f}", flush=True)

    # Coverage threshold for 80% mean power (linear interpolation on K).
    ks = sorted(int(k) for k in summary["by_k"])
    powers = [summary["by_k"][str(k)]["mean_power"] for k in ks]
    thresh = None
    for i in range(1, len(ks)):
        if powers[i - 1] < 0.8 <= powers[i]:
            frac = (0.8 - powers[i - 1]) / (powers[i] - powers[i - 1] + 1e-12)
            thresh = ks[i - 1] + frac * (ks[i] - ks[i - 1])
            break
    summary["coverage_for_80pct_power"] = thresh
    json.dump(summary, open(out_path, "w"), indent=2)

    # Plots: recon penalty vs K and mean power vs K.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot(ks, [summary["by_k"][str(k)]["delta_recon_natural"] for k in ks], "o-")
        a1.set_xlabel("coverage K (patched positions)"); a1.set_ylabel("recon penalty (nats)")
        a1.set_xscale("log", base=2); a1.set_title("Dose-response: penalty vs coverage")
        a2.plot(ks, powers, "o-"); a2.axhline(0.8, color="gray", ls="--", lw=1, label="80% power")
        a2.set_xlabel("coverage K (patched positions)"); a2.set_ylabel("mean detection power")
        a2.set_xscale("log", base=2); a2.set_ylim(-0.02, 1.02); a2.legend()
        a2.set_title("Detectability vs coverage")
        plt.tight_layout()
        fig_path = out_path.replace(".json", ".png")
        plt.savefig(fig_path, dpi=130)
        print(f"Wrote plot {fig_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nCOVERAGE POWER DONE. 80%-power coverage ~ {thresh}. Wrote {out_path}")


if __name__ == "__main__":
    main()
