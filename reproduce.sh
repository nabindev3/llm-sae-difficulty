#!/usr/bin/env bash
# Dual-layer pipeline for the LLM Bridge Project (Layer 12 Mid and Layer 18 Late).
# Stops at the first failure.

set -euo pipefail

PY=${PY:-python3}

echo "== [1/12] Running lightweight smoke test =="
$PY smoke_test.py

echo "== [2/12] Layer 12 (Mid): Extracting activations + HellaSwag labels =="
$PY extract_activations.py --layer_idx 11 --max_samples 5000 --output_dir activations

echo "== [3/12] Layer 12 (Mid): Training TopK SAE on TRAIN split prompt tokens =="
$PY sae/train_sae.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --epochs 5 --output_dir sae/checkpoints

echo "== [4/12] Layer 12 (Mid): Running difficulty logistic probe =="
$PY probing/probe.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --results_json probing/results/probe_results.json --scores_parquet activations/probe_scores.parquet

echo "== [5/12] Layer 18 (Late): Extracting activations + HellaSwag labels =="
$PY extract_activations.py --layer_idx 17 --max_samples 5000 --output_dir activations_late

echo "== [6/12] Layer 18 (Late): Training TopK SAE on TRAIN split prompt tokens =="
$PY sae/train_sae.py --activations activations_late/hellaswag_activations.safetensors \
    --metadata activations_late/hellaswag_metadata.parquet --epochs 5 --output_dir sae/checkpoints_late

echo "== [7/12] Layer 18 (Late): Running difficulty logistic probe =="
$PY probing/probe.py --activations activations_late/hellaswag_activations.safetensors \
    --metadata activations_late/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints_late/sae_topk_32.pt \
    --results_json probing/results/probe_results_late.json --scores_parquet activations_late/probe_scores.parquet

echo "== [8/12] SQuAD/HellaSwag: Extracting Pythia-2.8B base correctness on test split prompts =="
$PY eval/extract_base.py --dataset hellaswag --small_metadata activations/hellaswag_metadata.parquet --output_dir activations_base

echo "== [9/12] Running Small <-> Base Cascade Pareto Routing on HellaSwag =="
$PY eval/cascade.py --small_metadata activations/hellaswag_metadata.parquet \
    --base_metadata activations_base/hellaswag_metadata.parquet \
    --probe_scores activations/probe_scores.parquet \
    --score_cols pred_P3_InputStats_SAE pred_P1_InputStats

echo "== [10/12] Running selective-prediction analysis on primary Layer 12 =="
$PY eval/selective_prediction.py --probe_scores activations/probe_scores.parquet \
    --metadata activations/hellaswag_metadata.parquet

echo "== [11/12] Running calibration diagnostics & Platt/Isotonic recalibration =="
$PY eval/calibration.py --probe_scores activations/probe_scores.parquet
$PY eval/recalibrate.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt

echo "== [12/12] Running Mishra-style hook-based causal ablation on Layer 12 =="
$PY eval/causal_ablation.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt

echo "== [Final] Compiling cross-layer results & populating report.md =="
$PY eval/populate_report.py

echo
echo "=========================================================="
echo "Done! The dual-layer pipeline has executed successfully."
echo "Comparative cross-layer AUROC metrics are compiled in eval/report.md."
echo "=========================================================="
