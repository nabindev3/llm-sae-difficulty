#!/usr/bin/env bash
# End-to-end SQuAD Generation Perplexity & Cascade Routing Pipeline (Week 4 Mitigation)
# Stops at the first failure.

set -euo pipefail

PY=${PY:-python3}

echo "== [1/9] SQuAD: Extracting Layer 12 activations & gold perplexity labels =="
$PY extract_activations.py --dataset squad --layer_idx 11 --max_samples 5000 --output_dir activations

echo "== [2/9] SQuAD: Training TopK SAE on train split prompt tokens only =="
$PY sae/train_sae.py --activations activations/squad_activations.safetensors \
    --metadata activations/squad_metadata.parquet --epochs 5 --output_dir sae/checkpoints

echo "== [3/9] SQuAD: Running difficulty L1-logistic probe =="
$PY probing/probe.py --activations activations/squad_activations.safetensors \
    --metadata activations/squad_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --results_json probing/results/squad_probe_results.json --scores_parquet activations/squad_scores.parquet

echo "== [4/9] SQuAD: Extracting Pythia-2.8B base correctness on test split prompts =="
$PY eval/extract_base.py --dataset squad --small_metadata activations/squad_metadata.parquet --output_dir activations_base

echo "== [5/9] SQuAD: Running Small <-> Base Cascade Pareto Routing =="
$PY eval/cascade.py --small_metadata activations/squad_metadata.parquet \
    --base_metadata activations_base/squad_metadata.parquet \
    --probe_scores activations/squad_scores.parquet \
    --score_cols pred_P3_InputStats_SAE pred_P1_InputStats \
    --output_dir eval/results/squad

echo "== [6/9] SQuAD: Running selective-answering risk-coverage analysis =="
$PY eval/selective_prediction.py --probe_scores activations/squad_scores.parquet \
    --metadata activations/squad_metadata.parquet \
    --out_dir eval/results/squad

echo "== [7/9] SQuAD: Running calibration diagnostics & Platt/Isotonic recalibration =="
$PY eval/calibration.py --probe_scores activations/squad_scores.parquet \
    --out_dir eval/results/squad
$PY eval/recalibrate.py --activations activations/squad_activations.safetensors \
    --metadata activations/squad_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --out_dir eval/results/squad

echo "== [8/9] SQuAD: Running Mishra-style hook-based causal ablation on Layer 12 =="
$PY eval/causal_ablation.py --dataset squad --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --out_dir eval/results/squad

echo "== [9/9] SQuAD: Compiling SQuAD results and populating report_squad.md =="
$PY eval/populate_report_squad.py --squad_results_dir eval/results/squad

echo
echo "=========================================================="
echo "SQuAD Pipeline completed successfully!"
echo "All continuous perplexity metrics and Pareto frontiers generated!"
echo "=========================================================="
