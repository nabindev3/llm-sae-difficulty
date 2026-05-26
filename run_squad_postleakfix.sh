#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-../tsfm-sae-routing/venv/bin/python3}

echo "== [1/6] SQuAD probe (post-leakage-fix) =="
$PY probing/probe.py \
   --activations activations/squad_activations.safetensors \
   --metadata activations/squad_metadata.parquet \
   --sae_ckpt sae/checkpoints/sae_topk_32.pt \
   --results_json probing/results/squad_probe_results.json \
   --scores_parquet activations/squad_scores.parquet

echo "== [2/6] SQuAD cascade =="
$PY eval/cascade.py \
   --small_metadata activations/squad_metadata.parquet \
   --base_metadata activations_base/squad_metadata.parquet \
   --probe_scores activations/squad_scores.parquet \
   --score_cols pred_P3_InputStats_SAE pred_P1_InputStats \
   --output_dir eval/results/squad

echo "== [3/6] SQuAD selective prediction =="
$PY eval/selective_prediction.py \
   --probe_scores activations/squad_scores.parquet \
   --metadata activations/squad_metadata.parquet \
   --out_dir eval/results/squad

echo "== [4/6] SQuAD calibration =="
$PY eval/calibration.py \
   --probe_scores activations/squad_scores.parquet \
   --out_dir eval/results/squad

echo "== [5/6] SQuAD recalibration =="
$PY eval/recalibrate.py \
   --activations activations/squad_activations.safetensors \
   --metadata activations/squad_metadata.parquet \
   --sae_ckpt sae/checkpoints/sae_topk_32.pt \
   --out_dir eval/results/squad

echo "== [6/6] SQuAD report populate =="
$PY eval/populate_report_squad.py --squad_results_dir eval/results/squad

echo
echo "=== Post-leakage-fix SQuAD pipeline complete ==="
