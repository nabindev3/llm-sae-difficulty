#!/usr/bin/env bash
# Plan 4 chain: waits for Layer-18 SQuAD extract to finish, then runs SAE training and probe.
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

# Wait for extract output (the safetensors file is the last thing written)
until [ -f activations_late/squad_activations.safetensors ] && [ -f activations_late/squad_metadata.parquet ]; do
  sleep 10
done
sleep 5  # let any final flushes settle

echo "== Plan 4 [2/3]: Train Layer-18 SAE on SQuAD activations =="
cd sae
$PY train_sae.py \
  --activations ../activations_late/squad_activations.safetensors \
  --metadata ../activations_late/squad_metadata.parquet \
  --k 32 \
  --output_dir checkpoints_late_squad
cd ..

echo "== Plan 4 [3/3]: Probe Layer-18 SQuAD =="
$PY probing/probe.py \
  --activations activations_late/squad_activations.safetensors \
  --metadata activations_late/squad_metadata.parquet \
  --sae_ckpt sae/checkpoints_late_squad/sae_topk_32.pt \
  --results_json probing/results/squad_probe_late_results.json \
  --scores_parquet activations_late/squad_scores.parquet

echo "=== Plan 4 chain complete ==="
