#!/usr/bin/env bash
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

echo "== Step 1 [1/7]: Train HellaSwag-matched L12 SAE =="
$PY sae/train_sae.py \
  --activations activations/hellaswag_activations.safetensors \
  --metadata activations/hellaswag_metadata.parquet \
  --k 32 \
  --output_dir sae/checkpoints_hellaswag_l12

echo "== Step 1 [2/7]: HellaSwag probe with dataset-matched L12 SAE =="
$PY probing/probe.py \
  --activations activations/hellaswag_activations.safetensors \
  --metadata activations/hellaswag_metadata.parquet \
  --sae_ckpt sae/checkpoints_hellaswag_l12/sae_topk_32.pt \
  --results_json probing/results/probe_results.json \
  --scores_parquet activations/probe_scores.parquet

echo "== Step 1 [3/7]: HellaSwag cascade (uncalibrated) =="
$PY eval/cascade.py \
  --small_metadata activations/hellaswag_metadata.parquet \
  --base_metadata activations_base/hellaswag_metadata.parquet \
  --probe_scores activations/probe_scores.parquet \
  --score_cols pred_P3_InputStats_SAE pred_P1_InputStats \
  --output_dir eval/results

echo "== Step 1 [4/7]: HellaSwag selective prediction =="
$PY eval/selective_prediction.py \
  --probe_scores activations/probe_scores.parquet \
  --metadata activations/hellaswag_metadata.parquet \
  --out_dir eval/results

echo "== Step 1 [5/7]: HellaSwag calibration =="
$PY eval/calibration.py \
  --probe_scores activations/probe_scores.parquet \
  --out_dir eval/results

echo "== Step 1 [6/7]: HellaSwag recalibrate (with --probe_scores so cascade can route on Platt for parity) =="
$PY eval/recalibrate.py \
  --activations activations/hellaswag_activations.safetensors \
  --metadata activations/hellaswag_metadata.parquet \
  --sae_ckpt sae/checkpoints_hellaswag_l12/sae_topk_32.pt \
  --probe_scores activations/probe_scores.parquet \
  --out_dir eval/results

echo "== Step 1 [6b/7]: HellaSwag calibrated cascade (closes Plan 3 parity gap) =="
$PY eval/cascade.py \
  --small_metadata activations/hellaswag_metadata.parquet \
  --base_metadata activations_base/hellaswag_metadata.parquet \
  --probe_scores activations/probe_scores.parquet \
  --score_cols pred_P3_InputStats_SAE_platt pred_P1_InputStats_platt pred_P3_InputStats_SAE pred_P1_InputStats \
  --output_dir eval/results/calibrated

echo "== Step 1 [7/7]: HellaSwag populate report =="
$PY eval/populate_report.py

echo "=== Step 1 complete ==="
