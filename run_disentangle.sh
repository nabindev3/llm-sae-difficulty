#!/usr/bin/env bash
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

echo "== Disentangle [1/2]: SQuAD — all-position SAE + BOUNDARY-only intervention =="
# All-position SAE applied at a single token position (prompt_len-1, the last prompt token).
# Comparing this to Step 2's all-position-intervention result isolates the
# "intervene at more positions" effect from the "more faithful SAE" effect.
$PY eval/causal_ablation.py \
  --dataset squad \
  --metric continuous \
  --positions boundary \
  --activations activations_allpos/squad_activations.safetensors \
  --metadata activations_allpos/squad_metadata.parquet \
  --sae_ckpt sae/checkpoints_allpos_squad/sae_topk_32.pt \
  --out_dir eval/results/squad/disentangle

echo "== Disentangle [2/2]: HellaSwag — all-position SAE + BOUNDARY at last-prompt-token =="
# HellaSwag's original boundary (first ending token, position prompt_len) is OOD for the
# all-position SAE (which only saw prompt tokens). Use last-prompt boundary instead so the
# SAE is applied in-distribution. This is a different position than the original boundary
# run; the comparison is "all-position SAE applied at 1 in-distribution position" vs
# "all-position SAE applied at all in-distribution positions".
$PY eval/causal_ablation.py \
  --dataset hellaswag \
  --metric continuous \
  --positions boundary \
  --hellaswag_boundary last_prompt \
  --activations activations_allpos/hellaswag_activations.safetensors \
  --metadata activations_allpos/hellaswag_metadata.parquet \
  --sae_ckpt sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
  --out_dir eval/results/disentangle

echo "=== Disentangle complete ==="
