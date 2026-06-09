"""Populate a difficulty / cascade report template with numbers from the pipeline.

One populator, parametrized by ``--dataset`` instead of forking per dataset:

  * hellaswag  multiple-choice difficulty. Adds a cross-layer probing table
               (Layer 12 mid vs Layer 18 late) and a positive/null narrative.
  * squad      generative perplexity difficulty + cascade routing. Single-layer
               table; reads its result JSONs from a ``squad/`` subdirectory.

The two used to live in populate_report.py and populate_report_squad.py as
near-identical copies; the only real differences are captured in DATASETS below.
"""
import os
import json
import argparse


# Per-dataset configuration. Everything the two forks used to differ on lives
# here; the body of main() is identical for both datasets.
DATASETS = {
    "hellaswag": {
        "template": "eval/report_template.md",
        "output": "eval/report.md",
        "probe_results": "probing/results/probe_results.json",
        # Second layer for the cross-layer table; None means single-layer.
        "late_probe_results": "probing/results/probe_results_late.json",
        "results_dir": "eval/results",
        "ablation_filename": "causal_ablation.json",
        "narrative": True,
        "cascade_target": "[Optional] Figure 3: cascade Pareto — `eval/results/cascade_pareto.png` comparing Pythia-410M ↔ Pythia-2.8B.",
        "cascade_probe_key": "pred_P1_InputStats",
        "cascade_probe_label": "P1",
    },
    "squad": {
        "template": "eval/report_template_squad.md",
        "output": "eval/report_squad.md",
        "probe_results": "probing/results/squad_probe_results.json",
        "late_probe_results": None,
        "results_dir": "eval/results/squad",
        "ablation_filename": "squad_causal_ablation.json",
        "narrative": False,
        "cascade_target": "[Optional] Figure 3: cascade Pareto — `eval/results/pareto_frontier.png`",
        "cascade_probe_key": "pred_P3_InputStats_SAE",
        "cascade_probe_label": "P3",
    },
}


def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def build_probe_table(probe_res, probe_res_late):
    """Probing-results table. Two AUROC columns when late-layer results are
    supplied (cross-layer robustness), one column otherwise."""
    def cell(res, prefix):
        return (f"{res.get(prefix + '_AUROC', 0.0):.3f} "
                f"({res.get(prefix + '_CI_lower', 0.0):.3f}, {res.get(prefix + '_CI_upper', 0.0):.3f})")

    rows = [
        ("P1 Input Stats", "P1"),
        ("P2 Stats + Raw", "P2"),
        ("P3 Stats + SAE", "P3"),
        ("P4 Raw Only (diag.)", "P4_RawOnly"),
        ("P5 SAE Only (diag.)", "P5_SAEOnly"),
    ]

    if probe_res_late is not None:
        lines = [
            "\n### Cross-Layer Robustness Probing Results",
            "We evaluate difficulty prediction at two pre-registered layers: **Layer 12 (mid)** and **Layer 18 (late)** of Pythia-410M.",
            "",
            "| Probe | Layer 12 Mid AUROC (95% CI) | Layer 18 Late AUROC (95% CI) |",
            "| :--- | :--- | :--- |",
        ]
        lines += [f"| {label} | {cell(probe_res, prefix)} | {cell(probe_res_late, prefix)} |"
                  for label, prefix in rows]
    else:
        lines = [
            "\n### Probing continuous perplexity difficulty",
            "We evaluate difficulty prediction at Layer 12 (mid) of Pythia-410M on SQuAD.",
            "",
            "| Probe | Layer 12 Mid AUROC (95% CI) |",
            "| :--- | :--- |",
        ]
        lines += [f"| {label} | {cell(probe_res, prefix)} |" for label, prefix in rows]
    return "\n".join(lines)


def build_calibration_block(cal_res):
    lines = ["\n### Calibration Results"]
    if cal_res:
        lines.append("| Probe | ECE (raw) | Brier (raw) |")
        lines.append("| :--- | :--- | :--- |")
        for col in ["pred_P1_InputStats", "pred_P3_InputStats_SAE"]:
            p_data = cal_res.get("probes", {}).get(col, {})
            name = col.replace("pred_", "").replace("_", " ")
            lines.append(f"| {name} | {p_data.get('ece', 0.0):.3f} | {p_data.get('brier', 0.0):.3f} |")
    return "\n".join(lines)


def build_recalibration_block(recal_res):
    lines = ["\n### Platt & Isotonic Recalibration Results"]
    if recal_res:
        lines.append("| Probe | Raw ECE | Platt Recal ECE | Isotonic Recal ECE |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for name in ["P1_InputStats", "P3_InputStats_SAE"]:
            p_data = recal_res.get(name, {})
            lines.append(
                f"| {name.replace('_', ' ')} | {p_data.get('raw', {}).get('ece', 0.0):.3f} | "
                f"{p_data.get('platt', {}).get('ece', 0.0):.3f} | {p_data.get('isotonic', {}).get('ece', 0.0):.3f} |"
            )
    return "\n".join(lines)


def build_selective_block(sel_res):
    lines = ["\n### Selective Answering Metrics"]
    if sel_res:
        p1_stats = sel_res.get("probes", {}).get("pred_P1_InputStats", {})
        p3_sae = sel_res.get("probes", {}).get("pred_P3_InputStats_SAE", {})
        lines.append(f"- No-Abstention Error Rate: {sel_res.get('mean_error_no_abstention', 0.0):.2%}")
        lines.append(f"- Oracle selective AURC: {sel_res.get('oracle_aurc', 0.0):.3f}")
        lines.append(f"- Random selective AURC: {sel_res.get('random_aurc', 0.0):.3f}")
        lines.append(f"- P1 (Stats) selective AURC: {p1_stats.get('aurc', 0.0):.3f}")
        lines.append(f"- P3 (SAE) selective AURC: {p3_sae.get('aurc', 0.0):.3f}")
    return "\n".join(lines)


def build_ablation_block(abl_res):
    lines = [
        "\n### मिश्रा-Style Causal Ablation Findings",
        f"- Natural error: {abl_res.get('natural_error_mean', 0.0):.2%}",
        f"- SAE reconstructed error: {abl_res.get('recon_error_mean', 0.0):.2%}",
        f"- Reconstruction penalty delta: {abl_res.get('delta_recon_natural', 0.0):+.2%}",
    ]
    lines.append("\n**Individual Feature Effects (Mean Delta Error vs Recon):**")
    for feat, d in abl_res.get("feature_effects", {}).items():
        lines.append(
            f"- Feature {feat}: {d.get('delta_error', 0.0):+.2%} "
            f"(95% CI [{d.get('ci_lower', 0.0):+.2%}, {d.get('ci_upper', 0.0):+.2%}])"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="hellaswag")
    ap.add_argument("--template", default=None, help="override the dataset's template path")
    ap.add_argument("--output", default=None, help="override the dataset's output path")
    ap.add_argument("--results_dir", default=None, help="override the dataset's eval results directory")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    template_path = args.template or cfg["template"]
    output_path = args.output or cfg["output"]
    results_dir = args.results_dir or cfg["results_dir"]

    # Load all generated JSON stats.
    probe_res = load_json_safe(cfg["probe_results"])
    probe_res_late = (
        load_json_safe(cfg["late_probe_results"]) if cfg["late_probe_results"] else None
    )
    sel_res = load_json_safe(os.path.join(results_dir, "selective_prediction.json"))
    cal_res = load_json_safe(os.path.join(results_dir, "calibration_results.json"))
    recal_res = load_json_safe(os.path.join(results_dir, "recalibration_results.json"))
    casc_res = load_json_safe(os.path.join(results_dir, "cascade_results.json"))
    abl_res = load_json_safe(os.path.join(results_dir, cfg["ablation_filename"]))

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found!")
        return

    with open(template_path, "r") as f:
        report = f.read()

    # Abstract replacements using Layer 12 (mid) as primary
    delta_sae = f"{probe_res.get('delta_sae', 0.0):+.3f}"
    delta_sae_ci_l = f"{probe_res.get('delta_sae_CI_lower', 0.0):+.3f}"
    delta_sae_ci_u = f"{probe_res.get('delta_sae_CI_upper', 0.0):+.3f}"

    delta_sae_over_raw = f"{probe_res.get('delta_sae_over_raw', 0.0):+.3f}"
    delta_sor_ci_l = f"{probe_res.get('delta_sae_over_raw_CI_lower', 0.0):+.3f}"
    delta_sor_ci_u = f"{probe_res.get('delta_sae_over_raw_CI_upper', 0.0):+.3f}"

    report = report.replace(
        "[P3−P1 ΔAUROC = X, 95% CI (a,b)]",
        f"P3−P1 ΔAUROC = {delta_sae} (95% CI [{delta_sae_ci_l}, {delta_sae_ci_u}])"
    )
    report = report.replace(
        "[SAE vs raw: P3−P2 ΔAUROC = Y]",
        f"SAE vs raw: P3−P2 ΔAUROC = {delta_sae_over_raw} (95% CI [{delta_sor_ci_l}, {delta_sor_ci_u}])"
    )

    # Positive/null narrative (hellaswag only).
    if cfg["narrative"]:
        is_null = float(probe_res.get('delta_sae_over_raw', 0.0)) <= 0.0
        narrative = (
            "We report a clean cross-modality **null result**: SAE features do not "
            "provide predictive gains over raw representations or prompt statistics, "
            "suggesting the interpretability of SAEs does not represent an additional "
            "predictive signal."
            if is_null else
            "We report a **positive result**: SAE features outperform raw activations on HellaSwag "
            "difficulty prediction, highlighting that sparse compression extracts clean predictive signals."
        )
        report = report.replace("[State honestly: positive / null.]", narrative)

    # Section 4: Setup replacements
    n_train = str(probe_res.get("n_train", "[FILL]"))
    n_test = str(probe_res.get("n_test", "[FILL]"))
    report = report.replace("n_train=[ ]", f"n_train={n_train}")
    report = report.replace("n_test=[ ]", f"n_test={n_test}")

    # Probing results table (cross-layer for hellaswag, single-layer for squad)
    report = report.replace(
        "Table 1: AUROC ± CI for P1/P2/P3.",
        build_probe_table(probe_res, probe_res_late),
    )

    report += "\n" + build_calibration_block(cal_res)
    report += "\n" + build_recalibration_block(recal_res)
    report += "\n" + build_selective_block(sel_res)

    # Cascade
    if casc_res:
        report = report.replace(
            cfg["cascade_target"],
            f"**Small-to-Base Cascade Pareto Routing results:**\n"
            f"- Cheap model: Pythia-410M (Error rate: {casc_res.get('always_cheap', {}).get('mean_error', 0.0):.2%}, Cost: 1.0)\n"
            f"- Base model: Pythia-2.8B (Error rate: {casc_res.get('always_base', {}).get('mean_error', 0.0):.2%}, Cost: 5.0)\n"
            f"- {cfg['cascade_probe_label']} routing dominates the linear baseline, finding "
            f"**{casc_res.get('probes', {}).get(cfg['cascade_probe_key'], {}).get('n_dominating_points', 0)}** Pareto-optimal points."
        )

    # Causal Ablation
    if abl_res:
        report += "\n" + build_ablation_block(abl_res)

    # Write output
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Successfully populated and wrote {output_path}")


if __name__ == "__main__":
    main()
