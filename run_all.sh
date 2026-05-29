#!/usr/bin/env bash
# =============================================================================
# End-to-end reproducer for the LLM Routing Probe pipeline.
#
# Runs every stage in dependency order, with idempotency checks (skips a stage
# if its terminal output already exists) and SHA256 checksums of key artifacts
# logged to eval/reproduction_manifest.txt.
#
# To force a re-run of a stage, delete its terminal artifact (listed under
# "Stage outputs" in each phase) and re-run this script.
#
# Total wall-clock from scratch on Apple Silicon MPS: ~3.5 hours.
# - extraction (5 datasets):                ~12 min
# - SAE training (6 SAEs):                  ~30 min
# - probing (3 layers/datasets):            ~20 min
# - causal ablation (5 configurations):     ~140 min
# - downstream eval (cascade/cal/recal):    ~5 min
# - permutation + pooling + chosen_C:       ~10 min
# =============================================================================
set -euo pipefail

PY=${PY:-../tsfm-sae-routing/venv/bin/python3}
MANIFEST=eval/reproduction_manifest.txt
mkdir -p eval logs

if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "ERROR: python interpreter '$PY' not found. Set PY=/path/to/python before invoking." >&2
  exit 1
fi

log_phase() { printf "\n\n============================================================\n  Phase %s\n============================================================\n" "$1"; }
log_skip()  { printf "  [skip] %s already exists\n" "$1"; }

# Append "TIMESTAMP  phase=N  artifact=<path>  sha256=<hash>" lines to the manifest.
sha256_of()  { shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }
log_artifact() {
  local phase="$1"; local artifact="$2"
  if [ ! -e "$artifact" ]; then
    echo "  [warn] $artifact missing — not adding to manifest"
    return
  fi
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local hash; hash=$(sha256_of "$artifact")
  printf "%s\tphase=%s\tartifact=%s\tsha256=%s\n" "$ts" "$phase" "$artifact" "$hash" >> "$MANIFEST"
  printf "  [ok] %s  %s…\n" "$artifact" "${hash:0:12}"
}

t0_total=$SECONDS

# -----------------------------------------------------------------------------
log_phase "1 — Boundary-token activation extraction (Pythia-410M, L12 + L18)"
# Stage outputs:
#   activations/{hellaswag,squad}_activations.safetensors
#   activations/{hellaswag,squad}_metadata.parquet
#   activations_late/{hellaswag,squad}_activations.safetensors
#   activations_late/{hellaswag,squad}_metadata.parquet
for ds in hellaswag squad; do
  if [ -f "activations/${ds}_activations.safetensors" ]; then
    log_skip "activations/${ds}_activations.safetensors"
  else
    $PY extract_activations.py --dataset "$ds" --layer_idx 11 --max_samples 5000 --output_dir activations
  fi
  log_artifact 1 "activations/${ds}_activations.safetensors"
  log_artifact 1 "activations/${ds}_metadata.parquet"
done
for ds in hellaswag squad; do
  if [ -f "activations_late/${ds}_activations.safetensors" ]; then
    log_skip "activations_late/${ds}_activations.safetensors"
  else
    $PY extract_activations.py --dataset "$ds" --layer_idx 17 --max_samples 5000 --output_dir activations_late
  fi
  log_artifact 1 "activations_late/${ds}_activations.safetensors"
  log_artifact 1 "activations_late/${ds}_metadata.parquet"
done

# -----------------------------------------------------------------------------
log_phase "2 — Boundary-token SAE training (dataset-matched, per layer)"
# Stage outputs:
#   sae/checkpoints_hellaswag_l12/sae_topk_32.pt
#   sae/checkpoints/sae_topk_32.pt                   (SQuAD-trained L12 — historical default path)
#   sae/checkpoints_late/sae_topk_32.pt              (HellaSwag-trained L18)
#   sae/checkpoints_late_squad/sae_topk_32.pt        (SQuAD-trained L18)
declare -A SAE_TARGETS=(
  ["sae/checkpoints_hellaswag_l12"]="activations/hellaswag_activations.safetensors activations/hellaswag_metadata.parquet"
  ["sae/checkpoints"]="activations/squad_activations.safetensors activations/squad_metadata.parquet"
  ["sae/checkpoints_late"]="activations_late/hellaswag_activations.safetensors activations_late/hellaswag_metadata.parquet"
  ["sae/checkpoints_late_squad"]="activations_late/squad_activations.safetensors activations_late/squad_metadata.parquet"
)
for ckpt_dir in "${!SAE_TARGETS[@]}"; do
  read -r acts meta <<<"${SAE_TARGETS[$ckpt_dir]}"
  ckpt="$ckpt_dir/sae_topk_32.pt"
  if [ -f "$ckpt" ]; then
    log_skip "$ckpt"
  else
    $PY sae/train_sae.py --activations "$acts" --metadata "$meta" --k 32 --output_dir "$ckpt_dir"
  fi
  log_artifact 2 "$ckpt"
done

# -----------------------------------------------------------------------------
log_phase "3 — Pythia-2.8B base evaluation (for cascade win-rate)"
# Stage outputs:
#   activations_base/{hellaswag,squad}_metadata.parquet
for ds in hellaswag squad; do
  if [ -f "activations_base/${ds}_metadata.parquet" ]; then
    log_skip "activations_base/${ds}_metadata.parquet"
  else
    $PY eval/extract_base.py --dataset "$ds" --model EleutherAI/pythia-2.8b --output_dir activations_base
  fi
  log_artifact 3 "activations_base/${ds}_metadata.parquet"
done

# -----------------------------------------------------------------------------
log_phase "4 — Probes (L12 HellaSwag, L12 SQuAD, L18 SQuAD)"
# Stage outputs:
#   probing/results/{probe_results,squad_probe_results,squad_probe_late_results}.json
#   activations/{hellaswag,squad}_scores.parquet, activations_late/squad_scores.parquet
if [ -f probing/results/probe_results.json ]; then log_skip "HellaSwag L12 probe"; else
  $PY probing/probe.py \
    --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet \
    --sae_ckpt sae/checkpoints_hellaswag_l12/sae_topk_32.pt \
    --results_json probing/results/probe_results.json \
    --scores_parquet activations/probe_scores.parquet
fi
log_artifact 4 probing/results/probe_results.json
log_artifact 4 activations/probe_scores.parquet

if [ -f probing/results/squad_probe_results.json ]; then log_skip "SQuAD L12 probe"; else
  $PY probing/probe.py \
    --activations activations/squad_activations.safetensors \
    --metadata activations/squad_metadata.parquet \
    --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --results_json probing/results/squad_probe_results.json \
    --scores_parquet activations/squad_scores.parquet
fi
log_artifact 4 probing/results/squad_probe_results.json
log_artifact 4 activations/squad_scores.parquet

if [ -f probing/results/squad_probe_late_results.json ]; then log_skip "SQuAD L18 probe"; else
  $PY probing/probe.py \
    --activations activations_late/squad_activations.safetensors \
    --metadata activations_late/squad_metadata.parquet \
    --sae_ckpt sae/checkpoints_late_squad/sae_topk_32.pt \
    --results_json probing/results/squad_probe_late_results.json \
    --scores_parquet activations_late/squad_scores.parquet
fi
log_artifact 4 probing/results/squad_probe_late_results.json

# -----------------------------------------------------------------------------
log_phase "5 — Cascade + selective + calibration + recalibration (SQuAD path is fuller)"
# HellaSwag downstream
if [ -f eval/results/cascade_results.json ]; then log_skip "HellaSwag cascade"; else
  $PY eval/cascade.py \
    --small_metadata activations/hellaswag_metadata.parquet \
    --base_metadata activations_base/hellaswag_metadata.parquet \
    --probe_scores activations/probe_scores.parquet \
    --score_cols pred_P3_InputStats_SAE pred_P1_InputStats \
    --output_dir eval/results
fi
log_artifact 5 eval/results/cascade_results.json
if [ ! -f eval/results/selective_prediction.json ]; then
  $PY eval/selective_prediction.py --probe_scores activations/probe_scores.parquet --metadata activations/hellaswag_metadata.parquet --out_dir eval/results
fi
log_artifact 5 eval/results/selective_prediction.json
if [ ! -f eval/results/calibration_results.json ]; then
  $PY eval/calibration.py --probe_scores activations/probe_scores.parquet --out_dir eval/results
fi
log_artifact 5 eval/results/calibration_results.json
if [ ! -f eval/results/recalibration_results.json ]; then
  $PY eval/recalibrate.py \
    --activations activations/hellaswag_activations.safetensors \
    --metadata activations/hellaswag_metadata.parquet \
    --sae_ckpt sae/checkpoints_hellaswag_l12/sae_topk_32.pt \
    --probe_scores activations/probe_scores.parquet \
    --out_dir eval/results
fi
log_artifact 5 eval/results/recalibration_results.json

# SQuAD downstream
if [ -f eval/results/squad/cascade_results.json ]; then log_skip "SQuAD cascade"; else
  $PY eval/cascade.py \
    --small_metadata activations/squad_metadata.parquet \
    --base_metadata activations_base/squad_metadata.parquet \
    --probe_scores activations/squad_scores.parquet \
    --score_cols pred_P3_InputStats_SAE pred_P1_InputStats \
    --output_dir eval/results/squad
fi
log_artifact 5 eval/results/squad/cascade_results.json
if [ ! -f eval/results/squad/selective_prediction.json ]; then
  $PY eval/selective_prediction.py --probe_scores activations/squad_scores.parquet --metadata activations/squad_metadata.parquet --out_dir eval/results/squad
fi
log_artifact 5 eval/results/squad/selective_prediction.json
if [ ! -f eval/results/squad/calibration_results.json ]; then
  $PY eval/calibration.py --probe_scores activations/squad_scores.parquet --out_dir eval/results/squad
fi
log_artifact 5 eval/results/squad/calibration_results.json
if [ ! -f eval/results/squad/recalibration_results.json ]; then
  $PY eval/recalibrate.py \
    --activations activations/squad_activations.safetensors \
    --metadata activations/squad_metadata.parquet \
    --sae_ckpt sae/checkpoints/sae_topk_32.pt \
    --probe_scores activations/squad_scores.parquet \
    --out_dir eval/results/squad
fi
log_artifact 5 eval/results/squad/recalibration_results.json
# Plan 3 parity: calibrated cascade with pred_*_platt columns
if [ ! -f eval/results/squad/calibrated/cascade_results.json ]; then
  $PY eval/cascade.py \
    --small_metadata activations/squad_metadata.parquet \
    --base_metadata activations_base/squad_metadata.parquet \
    --probe_scores activations/squad_scores.parquet \
    --score_cols pred_P3_InputStats_SAE_platt pred_P1_InputStats_platt pred_P3_InputStats_SAE pred_P1_InputStats \
    --output_dir eval/results/squad/calibrated
fi
log_artifact 5 eval/results/squad/calibrated/cascade_results.json

# -----------------------------------------------------------------------------
log_phase "6 — Causal ablation (boundary-only, binary metric — legacy reference)"
for ds in hellaswag squad; do
  out_dir="eval/results"
  if [ "$ds" = "squad" ]; then out_dir="eval/results/squad"; fi
  if [ -f "$out_dir/${ds}_causal_ablation.json" ]; then log_skip "$ds boundary binary causal"; else
    if [ "$ds" = "squad" ]; then
      $PY eval/causal_ablation.py --dataset squad --metric binary --positions boundary --out_dir "$out_dir"
    else
      $PY eval/causal_ablation.py --dataset hellaswag --metric binary --positions boundary --out_dir "$out_dir"
    fi
  fi
  log_artifact 6 "$out_dir/${ds}_causal_ablation.json"
done

# -----------------------------------------------------------------------------
log_phase "7 — Causal ablation (boundary-only, continuous metric)"
mkdir -p eval/results/continuous eval/results/squad/continuous
for ds in hellaswag squad; do
  out_dir="eval/results/continuous"
  if [ "$ds" = "squad" ]; then out_dir="eval/results/squad/continuous"; fi
  if [ -f "$out_dir/${ds}_causal_ablation.json" ]; then log_skip "$ds boundary continuous causal"; else
    $PY eval/causal_ablation.py --dataset "$ds" --metric continuous --positions boundary --out_dir "$out_dir"
  fi
  log_artifact 7 "$out_dir/${ds}_causal_ablation.json"
done

# -----------------------------------------------------------------------------
log_phase "8 — All-position extraction + SAE training"
mkdir -p activations_allpos sae/checkpoints_allpos_hellaswag sae/checkpoints_allpos_squad
for ds in hellaswag squad; do
  if [ -f "activations_allpos/${ds}_activations.safetensors" ]; then log_skip "all-pos $ds extract"; else
    $PY extract_prompt_sequences.py --dataset "$ds" --base_metadata "activations/${ds}_metadata.parquet" --layer_idx 11 --output_dir activations_allpos
  fi
  log_artifact 8 "activations_allpos/${ds}_activations.safetensors"
  log_artifact 8 "activations_allpos/${ds}_metadata.parquet"
done
for ds in hellaswag squad; do
  ckpt="sae/checkpoints_allpos_${ds}/sae_topk_32.pt"
  if [ -f "$ckpt" ]; then log_skip "all-pos $ds SAE"; else
    $PY sae/train_sae.py --activations "activations_allpos/${ds}_activations.safetensors" --metadata "activations_allpos/${ds}_metadata.parquet" --k 32 --output_dir "sae/checkpoints_allpos_${ds}"
  fi
  log_artifact 8 "$ckpt"
done

# -----------------------------------------------------------------------------
log_phase "9 — Causal ablation (all-position SAE + all-position intervention)"
mkdir -p eval/results/allpos eval/results/squad/allpos
for ds in hellaswag squad; do
  out_dir="eval/results/allpos"
  if [ "$ds" = "squad" ]; then out_dir="eval/results/squad/allpos"; fi
  if [ -f "$out_dir/${ds}_causal_ablation.json" ]; then log_skip "$ds all-pos causal"; else
    $PY eval/causal_ablation.py \
      --dataset "$ds" --metric continuous --positions all \
      --activations "activations_allpos/${ds}_activations.safetensors" \
      --metadata    "activations_allpos/${ds}_metadata.parquet" \
      --sae_ckpt    "sae/checkpoints_allpos_${ds}/sae_topk_32.pt" \
      --out_dir     "$out_dir"
  fi
  log_artifact 9 "$out_dir/${ds}_causal_ablation.json"
done

# -----------------------------------------------------------------------------
log_phase "10 — Disentanglement (all-position SAE + boundary-only intervention)"
mkdir -p eval/results/disentangle eval/results/squad/disentangle
if [ ! -f eval/results/squad/disentangle/squad_causal_ablation.json ]; then
  $PY eval/causal_ablation.py --dataset squad --metric continuous --positions boundary \
    --activations activations_allpos/squad_activations.safetensors \
    --metadata    activations_allpos/squad_metadata.parquet \
    --sae_ckpt    sae/checkpoints_allpos_squad/sae_topk_32.pt \
    --out_dir     eval/results/squad/disentangle
fi
log_artifact 10 eval/results/squad/disentangle/squad_causal_ablation.json

if [ ! -f eval/results/disentangle/hellaswag_causal_ablation.json ]; then
  $PY eval/causal_ablation.py --dataset hellaswag --metric continuous --positions boundary --hellaswag_boundary last_prompt \
    --activations activations_allpos/hellaswag_activations.safetensors \
    --metadata    activations_allpos/hellaswag_metadata.parquet \
    --sae_ckpt    sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
    --out_dir     eval/results/disentangle
fi
log_artifact 10 eval/results/disentangle/hellaswag_causal_ablation.json

# -----------------------------------------------------------------------------
log_phase "11 — Permutation tests on Δ(SAE − Raw) AUROC"
mkdir -p eval/results/permutation eval/results/squad/permutation
declare -A PERM=(
  ["eval/results/permutation/hellaswag_l12_p3_vs_p2.json"]="activations/probe_scores.parquet activations/hellaswag_metadata.parquet"
  ["eval/results/squad/permutation/squad_l12_p3_vs_p2.json"]="activations/squad_scores.parquet activations/squad_metadata.parquet"
  ["eval/results/squad/permutation/squad_l18_p3_vs_p2.json"]="activations_late/squad_scores.parquet activations_late/squad_metadata.parquet"
)
for out in "${!PERM[@]}"; do
  read -r scores meta <<<"${PERM[$out]}"
  if [ -f "$out" ]; then log_skip "perm test $(basename "$out" .json)"; else
    $PY eval/permutation_test.py --probe_scores "$scores" --metadata "$meta" \
      --score_a pred_P3_InputStats_SAE --score_b pred_P2_InputStats_Raw \
      --B 10000 --out_json "$out"
  fi
  log_artifact 11 "$out"
done

# -----------------------------------------------------------------------------
log_phase "12 — Per-pooling ablation (Task 1)"
mkdir -p eval/results/pooling eval/results/squad/pooling
if [ ! -f eval/results/pooling/hellaswag_l12_allpos.json ]; then
  $PY eval/pooling_ablation.py \
    --activations activations_allpos/hellaswag_activations.safetensors \
    --metadata    activations_allpos/hellaswag_metadata.parquet \
    --sae_ckpt    sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
    --label hellaswag_l12_allpos \
    --out_json eval/results/pooling/hellaswag_l12_allpos.json
fi
log_artifact 12 eval/results/pooling/hellaswag_l12_allpos.json
if [ ! -f eval/results/squad/pooling/squad_l12_allpos.json ]; then
  $PY eval/pooling_ablation.py \
    --activations activations_allpos/squad_activations.safetensors \
    --metadata    activations_allpos/squad_metadata.parquet \
    --sae_ckpt    sae/checkpoints_allpos_squad/sae_topk_32.pt \
    --label squad_l12_allpos \
    --out_json eval/results/squad/pooling/squad_l12_allpos.json
fi
log_artifact 12 eval/results/squad/pooling/squad_l12_allpos.json

# -----------------------------------------------------------------------------
log_phase "13 — chosen_C variance diagnosis (Task 2)"
mkdir -p eval/results/chosen_c
if [ ! -f eval/results/chosen_c/squad_l12.json ]; then
  $PY eval/chosen_c_diagnosis.py \
    --cases \
      "hellaswag_l12:activations/hellaswag_metadata.parquet:activations/hellaswag_activations.safetensors:sae/checkpoints_hellaswag_l12/sae_topk_32.pt" \
      "squad_l12:activations/squad_metadata.parquet:activations/squad_activations.safetensors:sae/checkpoints/sae_topk_32.pt" \
      "squad_l18:activations_late/squad_metadata.parquet:activations_late/squad_activations.safetensors:sae/checkpoints_late_squad/sae_topk_32.pt" \
    --out_dir eval/results/chosen_c
fi
log_artifact 13 eval/results/chosen_c/hellaswag_l12.json
log_artifact 13 eval/results/chosen_c/squad_l12.json
log_artifact 13 eval/results/chosen_c/squad_l18.json

# -----------------------------------------------------------------------------
log_phase "14 — Report population"
if [ ! -f eval/report.md ];       then $PY eval/populate_report.py;                                          fi
if [ ! -f eval/report_squad.md ]; then $PY eval/populate_report_squad.py --squad_results_dir eval/results/squad; fi
log_artifact 14 eval/report.md
log_artifact 14 eval/report_squad.md

# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo "  All phases complete."
printf "  Total wall-clock: %d s (%.1f min)\n" $((SECONDS - t0_total)) "$(echo "($SECONDS - $t0_total) / 60" | bc -l)"
echo "  Manifest with SHA256 checksums: $MANIFEST"
echo "============================================================"
