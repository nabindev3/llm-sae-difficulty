"""Run the extension configs (Limitations 1/6/7) end-to-end.

For each (model, dataset, layer) config: extract activations (extract_general.py)
-> train a TopK-32 SAE (train_sae.py) -> probe raw vs SAE (probe.py), and record
the headline Delta(SAE-Raw). Covers:

  * Limitation 6 (binary above-chance task): ARC-Easy on Pythia-410M, L12 + L18.
  * Limitation 7 (3rd continuous benchmark): TriviaQA on Pythia-410M, L12 + L18.
  * Limitation 1 (2nd backbone family / instruct): Qwen2.5-0.5B-Instruct on
    SQuAD (L12+L20) and ARC-Easy (L12).

Resumable at every stage (skips extract/SAE/probe whose output already exists),
so a kill never repeats finished work. Sequential (one small model resident at a
time) to stay within the 16GB Mac. Small models only -- no Gemma-size blowup.

Usage:
    python eval/extension_run.py                 # all configs
    python eval/extension_run.py --only arc_pythia_l12 triviaqa_pythia_l12
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "activations_ext"

# label -> (model, dataset, layer_idx, task)
CONFIGS = {
    "arc_pythia_l12":      ("EleutherAI/pythia-410m",      "arc_easy", 11, "binary"),
    "arc_pythia_l18":      ("EleutherAI/pythia-410m",      "arc_easy", 17, "binary"),
    "triviaqa_pythia_l12": ("EleutherAI/pythia-410m",      "triviaqa", 11, "continuous"),
    "triviaqa_pythia_l18": ("EleutherAI/pythia-410m",      "triviaqa", 17, "continuous"),
    "squad_qwen_l12":      ("Qwen/Qwen2.5-0.5B-Instruct",  "squad",    12, "continuous"),
    "squad_qwen_l20":      ("Qwen/Qwen2.5-0.5B-Instruct",  "squad",    20, "continuous"),
    "arc_qwen_l12":        ("Qwen/Qwen2.5-0.5B-Instruct",  "arc_easy", 12, "binary"),
}


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        raise RuntimeError(f"failed: {' '.join(cmd)}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out_json", default="eval/results/extension/extension_summary.json")
    args = ap.parse_args()

    out_path = os.path.join(ROOT, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = json.load(open(out_path)) if os.path.exists(out_path) else {"cells": []}
    done = {c["label"] for c in summary["cells"]}

    for label in args.only:
        if label in done:
            print(f"=== {label}: already in summary, skip ==="); continue
        model, dataset, layer, task = CONFIGS[label]
        outdir = os.path.join(BASE, label)
        acts = os.path.join(outdir, f"{dataset}_activations.safetensors")
        meta = os.path.join(outdir, f"{dataset}_metadata.parquet")
        sae_ckpt = os.path.join(outdir, "sae", "sae_topk_32.pt")
        rj = os.path.join(outdir, "probe_results.json")
        print(f"\n=== {label}  ({model}, {dataset}, L-idx {layer}, {task}) ===", flush=True)

        if not os.path.exists(acts):
            print("  [1/3] extracting ...", flush=True)
            run([args.python, "extract_general.py", "--model", model, "--dataset", dataset,
                 "--layer_idx", str(layer), "--max_samples", str(args.max_samples),
                 "--output_dir", outdir])
        if not os.path.exists(sae_ckpt):
            print("  [2/3] training SAE ...", flush=True)
            run([args.python, "sae/train_sae.py", "--activations", acts, "--metadata", meta,
                 "--k", "32", "--output_dir", os.path.join(outdir, "sae")])
        if not os.path.exists(rj):
            print("  [3/3] probing ...", flush=True)
            run([args.python, "probing/probe.py", "--activations", acts, "--metadata", meta,
                 "--sae_ckpt", sae_ckpt, "--results_json", rj,
                 "--scores_parquet", os.path.join(outdir, "scores.parquet")])

        r = json.load(open(os.path.join(ROOT, rj)) if not os.path.isabs(rj) else open(rj))
        best = max(r["P1_AUROC"], r["P2_AUROC"], r["P4_RawOnly_AUROC"], r["P5_SAEOnly_AUROC"])
        cell = {"label": label, "model": model, "dataset": dataset, "layer_idx": layer,
                "task": task, "test_hard_rate": r.get("hard_fraction"),
                "P1_AUROC": r["P1_AUROC"], "P4_RawOnly_AUROC": r["P4_RawOnly_AUROC"],
                "P5_SAEOnly_AUROC": r["P5_SAEOnly_AUROC"], "best_AUROC": best,
                "delta_raw": r["delta_raw"], "delta_sae_over_raw": r["delta_sae_over_raw"],
                "delta_sae_over_raw_CI": [r["delta_sae_over_raw_CI_lower"],
                                          r["delta_sae_over_raw_CI_upper"]]}
        summary["cells"].append(cell)
        json.dump(summary, open(out_path, "w"), indent=2)
        print(f"  -> hard_rate={cell['test_hard_rate']:.3f} P4_raw={cell['P4_RawOnly_AUROC']:.3f} "
              f"P5_sae={cell['P5_SAEOnly_AUROC']:.3f} bestAUROC={best:.3f} "
              f"Δ(SAE-Raw)={cell['delta_sae_over_raw']:+.4f} CI={cell['delta_sae_over_raw_CI']}", flush=True)

    # Verdict per limitation: any config where SAE significantly beats raw?
    sig = [c for c in summary["cells"] if c["delta_sae_over_raw_CI"][0] > 0]
    summary["n_cells"] = len(summary["cells"])
    summary["n_sae_beats_raw_sig"] = len(sig)
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\nEXTENSION DONE: {len(summary['cells'])} cells, "
          f"{len(sig)} where SAE significantly beats raw.\nWrote {out_path}")


if __name__ == "__main__":
    main()
