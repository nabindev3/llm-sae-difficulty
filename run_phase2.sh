#!/bin/bash
# Phase 2: after the layer-depth sweep finishes, run the remaining rigor steps
# sequentially (one model on MPS at a time): #4 stronger baseline (L12=idx11,
# L18=idx17), #5 causal-coverage power analysis, #3 position-matched disentangle.
set -u
cd "$(dirname "$0")"
PY="${PY:-$HOME/.venvs/tsfm-sae-difficulty/bin/python3}"
SUM=eval/results/layersweep/layersweep_summary.json

echo "[phase2] waiting for layer sweep (24 layers) ..."
for i in $(seq 1 240); do
  n=$("$PY" -c "import json,os;print(len(json.load(open('$SUM'))['cells']) if os.path.exists('$SUM') else 0)" 2>/dev/null)
  if [ "${n:-0}" -ge 24 ]; then echo "[phase2] layer sweep done ($n/24)"; break; fi
  echo "[phase2] layer sweep $n/24; waiting 30s"; sleep 30
done

echo "[phase2] === #4 stronger baseline (L12=idx11) ==="
"$PY" eval/strong_baseline.py \
  --activations eval/results/layersweep/_work/layer11_acts.safetensors \
  --metadata activations_layersweep/squad_metadata.parquet \
  --sae_ckpt eval/results/layersweep/_work/layer11_sae/sae_topk_32.pt \
  --out_json eval/results/strong_baseline/squad_l12.json

echo "[phase2] === #4 stronger baseline (L18=idx17) ==="
"$PY" eval/strong_baseline.py \
  --activations eval/results/layersweep/_work/layer17_acts.safetensors \
  --metadata activations_layersweep/squad_metadata.parquet \
  --sae_ckpt eval/results/layersweep/_work/layer17_sae/sae_topk_32.pt \
  --out_json eval/results/strong_baseline/squad_l18.json

echo "[phase2] === #5 causal-coverage power analysis ==="
"$PY" eval/coverage_power.py --ks 1 2 4 8 16 32 64

echo "[phase2] === #3 position-matched disentanglement ==="
"$PY" eval/position_matched.py

echo "[phase2] PHASE2 DONE"
