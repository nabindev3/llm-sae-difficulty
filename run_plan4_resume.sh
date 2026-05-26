#!/usr/bin/env bash
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

echo "== Plan 4 [2/3]: Train Layer-18 SAE on SQuAD activations =="
$PY sae/train_sae.py \
  --activations activations_late/squad_activations.safetensors \
  --metadata activations_late/squad_metadata.parquet \
  --k 32 \
  --output_dir sae/checkpoints_late_squad

echo "== Plan 4 [3/3]: Probe Layer-18 SQuAD =="
$PY probing/probe.py \
  --activations activations_late/squad_activations.safetensors \
  --metadata activations_late/squad_metadata.parquet \
  --sae_ckpt sae/checkpoints_late_squad/sae_topk_32.pt \
  --results_json probing/results/squad_probe_late_results.json \
  --scores_parquet activations_late/squad_scores.parquet

echo "=== Plan 4 resume complete ==="
