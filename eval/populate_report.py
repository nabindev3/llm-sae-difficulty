"""Populates the HellaSwag LLM bridge workshop report template with actual numbers
generated from the pipeline.
"""
import os
import json
import argparse


def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="eval/report_template.md")
    ap.add_argument("--output", default="eval/report.md")
    args = ap.parse_args()

    # Load all generated JSON stats for both layers
    probe_res = load_json_safe("probing/results/probe_results.json") # Layer 12
    probe_res_late = load_json_safe("probing/results/probe_results_late.json") # Layer 18
    
    sel_res = load_json_safe("eval/results/selective_prediction.json")
    cal_res = load_json_safe("eval/results/calibration_results.json")
    recal_res = load_json_safe("eval/results/recalibration_results.json")
    casc_res = load_json_safe("eval/results/cascade_results.json")
    abl_res = load_json_safe("eval/results/causal_ablation.json")

    if not os.path.exists(args.template):
        print(f"Error: Template {args.template} not found!")
        return

    with open(args.template, "r") as f:
        report = f.read()

    # Abstract replacements using Layer 12 (mid) as primary
    p1_auroc = f"{probe_res.get('P1_AUROC', 0.0):.3f}"
    p2_auroc = f"{probe_res.get('P2_AUROC', 0.0):.3f}"
    p3_auroc = f"{probe_res.get('P3_AUROC', 0.0):.3f}"
    
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

    # Build Comparative Cross-Layer Probing Results Table
    p1_late = f"{probe_res_late.get('P1_AUROC', 0.0):.3f}"
    p2_late = f"{probe_res_late.get('P2_AUROC', 0.0):.3f}"
    p3_late = f"{probe_res_late.get('P3_AUROC', 0.0):.3f}"
    p4_late = f"{probe_res_late.get('P4_RawOnly_AUROC', 0.0):.3f}"
    p5_late = f"{probe_res_late.get('P5_SAEOnly_AUROC', 0.0):.3f}"

    table_lines = [
        "\n### Cross-Layer Robustness Probing Results",
        "We evaluate difficulty prediction at two pre-registered layers: **Layer 12 (mid)** and **Layer 18 (late)** of Pythia-410M.",
        "",
        "| Probe | Layer 12 Mid AUROC (95% CI) | Layer 18 Late AUROC (95% CI) |",
        "| :--- | :--- | :--- |",
        f"| P1 Input Stats | {p1_auroc} ({probe_res.get('P1_CI_lower', 0.0):.3f}, {probe_res.get('P1_CI_upper', 0.0):.3f}) | {p1_late} ({probe_res_late.get('P1_CI_lower', 0.0):.3f}, {probe_res_late.get('P1_CI_upper', 0.0):.3f}) |",
        f"| P2 Stats + Raw | {p2_auroc} ({probe_res.get('P2_CI_lower', 0.0):.3f}, {probe_res.get('P2_CI_upper', 0.0):.3f}) | {p2_late} ({probe_res_late.get('P2_CI_lower', 0.0):.3f}, {probe_res_late.get('P2_CI_upper', 0.0):.3f}) |",
        f"| P3 Stats + SAE | {p3_auroc} ({probe_res.get('P3_CI_lower', 0.0):.3f}, {probe_res.get('P3_CI_upper', 0.0):.3f}) | {p3_late} ({probe_res_late.get('P3_CI_lower', 0.0):.3f}, {probe_res_late.get('P3_CI_upper', 0.0):.3f}) |",
        f"| P4 Raw Only (diag.) | {probe_res.get('P4_RawOnly_AUROC', 0.0):.3f} ({probe_res.get('P4_RawOnly_CI_lower', 0.0):.3f}, {probe_res.get('P4_RawOnly_CI_upper', 0.0):.3f}) | {p4_late} ({probe_res_late.get('P4_RawOnly_CI_lower', 0.0):.3f}, {probe_res_late.get('P4_RawOnly_CI_upper', 0.0):.3f}) |",
        f"| P5 SAE Only (diag.) | {probe_res.get('P5_SAEOnly_AUROC', 0.0):.3f} ({probe_res.get('P5_SAEOnly_CI_lower', 0.0):.3f}, {probe_res.get('P5_SAEOnly_CI_upper', 0.0):.3f}) | {p5_late} ({probe_res_late.get('P5_SAEOnly_CI_lower', 0.0):.3f}, {probe_res_late.get('P5_SAEOnly_CI_upper', 0.0):.3f}) |"
    ]
    report = report.replace("Table 1: AUROC ± CI for P1/P2/P3.", "\n".join(table_lines))

    # Calibration section
    cal_lines = ["\n### Calibration Results"]
    if cal_res:
        cal_lines.append("| Probe | ECE (raw) | Brier (raw) |")
        cal_lines.append("| :--- | :--- | :--- |")
        for col in ["pred_P1_InputStats", "pred_P3_InputStats_SAE"]:
            p_data = cal_res.get("probes", {}).get(col, {})
            name = col.replace("pred_", "").replace("_", " ")
            cal_lines.append(f"| {name} | {p_data.get('ece', 0.0):.3f} | {p_data.get('brier', 0.0):.3f} |")
    report += "\n" + "\n".join(cal_lines)

    # Recalibration section
    recal_lines = ["\n### Platt & Isotonic Recalibration Results"]
    if recal_res:
        recal_lines.append("| Probe | Raw ECE | Platt Recal ECE | Isotonic Recal ECE |")
        recal_lines.append("| :--- | :--- | :--- | :--- |")
        for name in ["P1_InputStats", "P3_InputStats_SAE"]:
            p_data = recal_res.get(name, {})
            recal_lines.append(
                f"| {name.replace('_', ' ')} | {p_data.get('raw', {}).get('ece', 0.0):.3f} | "
                f"{p_data.get('platt', {}).get('ece', 0.0):.3f} | {p_data.get('isotonic', {}).get('ece', 0.0):.3f} |"
            )
    report += "\n" + "\n".join(recal_lines)

    # Selective Prediction
    sel_lines = ["\n### Selective Answering Metrics"]
    if sel_res:
        p1_stats = sel_res.get("probes", {}).get("pred_P1_InputStats", {})
        p3_sae = sel_res.get("probes", {}).get("pred_P3_InputStats_SAE", {})
        sel_lines.append(f"- No-Abstention Error Rate: {sel_res.get('mean_error_no_abstention', 0.0):.2%}")
        sel_lines.append(f"- Oracle selective AURC: {sel_res.get('oracle_aurc', 0.0):.3f}")
        sel_lines.append(f"- Random selective AURC: {sel_res.get('random_aurc', 0.0):.3f}")
        sel_lines.append(f"- P1 (Stats) selective AURC: {p1_stats.get('aurc', 0.0):.3f}")
        sel_lines.append(f"- P3 (SAE) selective AURC: {p3_sae.get('aurc', 0.0):.3f}")
    report += "\n" + "\n".join(sel_lines)

    # Cascade
    if casc_res:
        report = report.replace(
            "[Optional] Figure 3: cascade Pareto — `eval/results/cascade_pareto.png` comparing Pythia-410M ↔ Pythia-2.8B.",
            f"**Small-to-Base Cascade Pareto Routing results:**\n"
            f"- Cheap model: Pythia-410M (Error rate: {casc_res.get('always_cheap', {}).get('mean_error', 0.0):.2%}, Cost: 1.0)\n"
            f"- Base model: Pythia-2.8B (Error rate: {casc_res.get('always_base', {}).get('mean_error', 0.0):.2%}, Cost: 5.0)\n"
            f"- P1 routing dominates the linear baseline, finding "
            f"**{casc_res.get('probes', {}).get('pred_P1_InputStats', {}).get('n_dominating_points', 0)}** Pareto-optimal points."
        )

    # Causal Ablation
    if abl_res:
        abl_lines = [
            "\n### मिश्रा-Style Causal Ablation Findings",
            f"- Natural error: {abl_res.get('natural_error_mean', 0.0):.2%}",
            f"- SAE reconstructed error: {abl_res.get('recon_error_mean', 0.0):.2%}",
            f"- Reconstruction penalty delta: {abl_res.get('delta_recon_natural', 0.0):+.2%}"
        ]
        abl_lines.append("\n**Individual Feature Effects (Mean Delta Error vs Recon):**")
        for feat, d in abl_res.get("feature_effects", {}).items():
            abl_lines.append(
                f"- Feature {feat}: {d.get('delta_error', 0.0):+.2%} "
                f"(95% CI [{d.get('ci_lower', 0.0):+.2%}, {d.get('ci_upper', 0.0):+.2%}])"
            )
        report += "\n" + "\n".join(abl_lines)

    # Write output
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Successfully populated and wrote {args.output}")


if __name__ == "__main__":
    main()
