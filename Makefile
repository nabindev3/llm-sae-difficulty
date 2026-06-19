# llm-sae-difficulty — consolidated entrypoints for the run_*.sh zoo.
# `make help` lists presets. Override the interpreter with `make PY=path squad`.
# Large tensors/checkpoints live in an HF dataset: run `make artifacts` first
# for any target that touches activations_allpos/ or sae/checkpoints*/.
PY ?= python3
export PY
.DEFAULT_GOAL := help

.PHONY: help artifacts all hellaswag squad multiseed sweep layersweep extensions \
        coverage position-matched strong-baseline dla steering baseline-shap test clean

help:  ## List available presets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-17s\033[0m %s\n",$$1,$$2}'

artifacts:  ## Download tensors + SAE checkpoints from the HF dataset
	bash download_artifacts.sh

# --- Canonical reproducers -------------------------------------------------
all:  ## Full 14-phase reproducer (extraction → … → report) [~3.5h]
	bash run_all.sh
hellaswag:  ## HellaSwag dual-layer (L12+L18) boundary pipeline
	bash reproduce.sh
squad:  ## SQuAD continuous-perplexity + cascade pipeline (AUROC 0.716 gate)
	bash reproduce_squad.sh

# --- Robustness battery (§5) ----------------------------------------------
multiseed:  ## Training-induced variance, 5 seeds × 3 configs
	$(PY) eval/multiseed.py --seeds 0 1 2 3 4
sweep:  ## SAE design space: expansion × k × {topk,gated,jumprelu}
	$(PY) eval/sae_sweep.py
layersweep:  ## Δ(SAE−Raw) vs depth (all 24 layers) + plot
	$(PY) extract_layersweep.py --max_samples 5000 --output_dir activations_layersweep
	$(PY) eval/layer_sweep.py
extensions:  ## ARC-Easy + TriviaQA + Qwen2.5-0.5B backbone
	$(PY) eval/extension_run.py
coverage:  ## Causal-coverage power analysis (needs `make artifacts`)
	$(PY) eval/coverage_power.py --ks 1 2 4 8 16 32 64
position-matched:  ## Position-controlled fidelity-vs-coverage (needs `make artifacts`)
	$(PY) eval/position_matched.py

# --- Explanatory analyses --------------------------------------------------
strong-baseline:  ## Harder P1 (confidence+entropy+emb-dist); needs `make layersweep`
	$(PY) eval/strong_baseline.py \
	  --activations eval/results/layersweep/_work/layer11_acts.safetensors \
	  --metadata activations_layersweep/squad_metadata.parquet \
	  --sae_ckpt eval/results/layersweep/_work/layer11_sae/sae_topk_32.pt \
	  --out_json eval/results/strong_baseline/squad_l12.json
dla:  ## Direct-logit attribution on top causal features (needs `make artifacts`)
	$(PY) eval/dla.py --dataset hellaswag \
	  --sae_ckpt sae/checkpoints_allpos_hellaswag/sae_topk_32.pt \
	  --activations activations_allpos/hellaswag_activations.safetensors \
	  --metadata activations_allpos/hellaswag_metadata.parquet \
	  --causal_json eval/results/allpos/hellaswag_causal_ablation.json \
	  --out_json eval/results/dla/hellaswag.json
steering:  ## Feature-steering output shift (needs `make artifacts`)
	$(PY) eval/steering.py --n_eval 400 --top_k 3 --out_json eval/results/steering/squad.json
baseline-shap:  ## P1 coefficient/SHAP recipe
	$(PY) eval/baseline_shap.py --out_json eval/results/baseline_shap/squad_l12.json

# --- Dev -------------------------------------------------------------------
test:  ## Fast unit tests + synthetic smoke + leakage audit
	$(PY) -m pytest tests/ -q
clean:  ## Remove regeneratable work dirs (keeps summaries)
	rm -rf eval/results/*/_work* activations_layersweep activations_ext

# Note: the remaining run_*.sh (run_step*, run_plan*, run_from_step4, *_quick,
# *_postleakfix, run_phase[23], run_gemma_resumable) are kept verbatim for
# provenance — they document exactly which one-off produced which artifact.
