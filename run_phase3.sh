#!/bin/bash
# Phase 3: after phase 2 (its last artifact = position_matched/summary.json),
# run the explanatory analyses sequentially: #11 direct-logit-attribution,
# #12 feature-steering. (#13 P1 breakdown already ran.)
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/.venvs/tsfm-sae-difficulty/bin/python3}"
MARK=eval/results/position_matched/summary.json

echo "[phase3] waiting for phase2 to finish ..."
for i in $(seq 1 480); do
  [ -f "$MARK" ] && { echo "[phase3] phase2 done"; break; }
  sleep 30
done

echo "[phase3] === #11 direct-logit-attribution (hellaswag top causal features) ==="
"$PY" eval/dla.py --dataset hellaswag \
  --sae_ckpt sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
  --activations activations_allpos/hellaswag_activations.safetensors \
  --metadata activations_allpos/hellaswag_metadata.parquet \
  --causal_json eval/results/allpos/hellaswag_causal_ablation.json \
  --out_json eval/results/dla/hellaswag.json

echo "[phase3] === #12 feature-steering (squad) ==="
"$PY" eval/steering.py --n_eval 400 --top_k 3 --out_json eval/results/steering/squad.json

echo "[phase3] PHASE3 DONE"
