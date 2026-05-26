#!/usr/bin/env bash
# Plan 2 SQuAD chain: waits for HellaSwag continuous to finish, then runs SQuAD continuous.
set -euo pipefail
PY=../tsfm-sae-routing/venv/bin/python3

until [ -f eval/results/continuous/hellaswag_causal_ablation.json ]; do
  sleep 30
done
sleep 5

echo "== Plan 2 [SQuAD continuous] =="
$PY eval/causal_ablation.py \
  --dataset squad \
  --metric continuous \
  --out_dir eval/results/squad/continuous

echo "=== Plan 2 SQuAD continuous complete ==="
