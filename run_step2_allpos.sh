#!/usr/bin/env bash
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

echo "== Step 2 [1/6]: Extract HellaSwag all-position prompt activations =="
$PY extract_prompt_sequences.py \
  --dataset hellaswag \
  --base_metadata activations/hellaswag_metadata.parquet \
  --layer_idx 11 \
  --output_dir activations_allpos

echo "== Step 2 [2/6]: Extract SQuAD all-position prompt activations =="
$PY extract_prompt_sequences.py \
  --dataset squad \
  --base_metadata activations/squad_metadata.parquet \
  --layer_idx 11 \
  --output_dir activations_allpos

echo "== Step 2 [3/6]: Train HellaSwag all-position L12 SAE =="
$PY sae/train_sae.py \
  --activations activations_allpos/hellaswag_activations.safetensors \
  --metadata activations_allpos/hellaswag_metadata.parquet \
  --k 32 \
  --output_dir sae/checkpoints_allpos_hellaswag

echo "== Step 2 [4/6]: Train SQuAD all-position L12 SAE =="
$PY sae/train_sae.py \
  --activations activations_allpos/squad_activations.safetensors \
  --metadata activations_allpos/squad_metadata.parquet \
  --k 32 \
  --output_dir sae/checkpoints_allpos_squad

echo "== Step 2 [5/6]: HellaSwag continuous causal at ALL prompt positions =="
$PY eval/causal_ablation.py \
  --dataset hellaswag \
  --metric continuous \
  --positions all \
  --activations activations_allpos/hellaswag_activations.safetensors \
  --metadata activations_allpos/hellaswag_metadata.parquet \
  --sae_ckpt sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
  --out_dir eval/results/allpos

echo "== Step 2 [6/6]: SQuAD continuous causal at ALL prompt positions =="
$PY eval/causal_ablation.py \
  --dataset squad \
  --metric continuous \
  --positions all \
  --activations activations_allpos/squad_activations.safetensors \
  --metadata activations_allpos/squad_metadata.parquet \
  --sae_ckpt sae/checkpoints_allpos_squad/sae_topk_32.pt \
  --out_dir eval/results/squad/allpos

echo "=== Step 2 complete ==="
