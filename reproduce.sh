#!/usr/bin/env bash
# Dual-layer pipeline for the LLM Bridge Project (Layer 12 Mid and Layer 18 Late).
# Stops at the first failure.

set -euo pipefail

PY=${PY:-python3}

echo "== [1/10] Running lightweight smoke test =="
$PY smoke_test.py

echo "== [2/10] Layer 12 (Mid): Extracting activations + HellaSwag labels =="
$PY extract_activations.py --layer_idx 11 --max_samples 1000 --output_dir activations

echo "== [3/10] Layer 12 (Mid): Training TopK SAE on TRAIN split prompt tokens =="
$PY sae/train_sae.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --epochs 5 --output_dir sae/checkpoints

echo "== [4/10] Layer 12 (Mid): Running difficulty logistic probe =="
$PY probing/probe.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --results_json probing/results/probe_results.json --scores_parquet activations/probe_scores.parquet

echo "== [5/10] Layer 18 (Late): Extracting activations + HellaSwag labels =="
$PY extract_activations.py --layer_idx 17 --max_samples 1000 --output_dir activations_late

echo "== [6/10] Layer 18 (Late): Training TopK SAE on TRAIN split prompt tokens =="
$PY sae/train_sae.py --activations activations_late/hellaswag_activations.safetensors \
    --metadata activations_late/hellaswag_metadata.parquet --epochs 5 --output_dir sae/checkpoints_late

echo "== [7/10] Layer 18 (Late): Running difficulty logistic probe =="
$PY probing/probe.py --activations activations_late/hellaswag_activations.safetensors \
    --metadata activations_late/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints_late/sae_topk_32.pt \
    --results_json probing/results/probe_results_late.json --scores_parquet activations_late/probe_scores.parquet

echo "== [8/10] Running selective-prediction analysis on primary Layer 12 =="
$PY eval/selective_prediction.py --probe_scores activations/probe_scores.parquet \
    --metadata activations/hellaswag_metadata.parquet

echo "== [9/10] Running calibration diagnostics & Platt/Isotonic recalibration =="
$PY eval/calibration.py --probe_scores activations/probe_scores.parquet
$PY eval/recalibrate.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt

echo "== [10/10] Running Mishra-style hook-based causal ablation on Layer 12 =="
$PY eval/causal_ablation.py --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet --sae_ckpt sae/checkpoints/sae_topk_32.pt

echo "== [Final] Compiling cross-layer results & populating report.md =="
$PY eval/populate_report.py

echo
echo "=========================================================="
echo "Done! The dual-layer pipeline has executed successfully."
echo "Comparative cross-layer AUROC metrics are compiled in eval/report.md."
echo "=========================================================="
