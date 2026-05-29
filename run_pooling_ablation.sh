#!/usr/bin/env bash
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

echo "=== HellaSwag L12 all-position per-pooling ablation ==="
$PY eval/pooling_ablation.py \
  --activations activations_allpos/hellaswag_activations.safetensors \
  --metadata activations_allpos/hellaswag_metadata.parquet \
  --sae_ckpt sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
  --label hellaswag_l12_allpos \
  --out_json eval/results/pooling/hellaswag_l12_allpos.json

echo
echo "=== SQuAD L12 all-position per-pooling ablation ==="
$PY eval/pooling_ablation.py \
  --activations activations_allpos/squad_activations.safetensors \
  --metadata activations_allpos/squad_metadata.parquet \
  --sae_ckpt sae/checkpoints_allpos_squad/sae_topk_32.pt \
  --label squad_l12_allpos \
  --out_json eval/results/squad/pooling/squad_l12_allpos.json

echo
echo "=== Done ==="
