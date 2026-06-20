# Model / Artifact Card — LLM SAE Difficulty Probe & Selective-QA

A **scope-and-honest-claims** card for this repository. It documents what the
result *does* and *does not* say, and the intended use of the one deployable
component (`selective_qa.PlattSelectiveQA`). Every number here is reproducible
from committed JSON via `notebooks/results_summary.ipynb`.

> ⚠️ Superseded by [`fm-difficulty-probe`](https://github.com/nabindev3/fm-difficulty-probe).
> This card covers the LLM (Pythia) side specifically.

## Summary

Two artifacts:
1. **A negative result (the science):** on the models tested, sparse-autoencoder
   (SAE) features extracted from the residual stream add **no incremental
   self-difficulty signal over raw activations**, for cost-quality routing.
2. **A positive, deployable tool (the engineering):** Platt-recalibrated
   **raw-activation** selective QA — packaged as `PlattSelectiveQA` — that gives
   useful coverage-based answer/abstain routing.

## What the result DOES say

- On **Pythia-410M**, TopK-32 SAE features **decrease** probe AUROC vs raw
  activations for difficulty on SQuAD (Δ(SAE−Raw) ≈ −0.08, label-permutation
  _p_ < 10⁻⁴), and this is **layer-invariant** (negative at all 24 layers).
- The null is **robust** (§5): it holds across 5 SAE seeds, ~80 SAE design
  points (expansion, sparsity, gated/JumpReLU/transcoder), a second
  instruction-tuned backbone (**Qwen2.5-0.5B**), two extra benchmarks
  (**ARC-Easy** where the model is well above chance, **TriviaQA**), a hardened
  baseline, and — addressing the biggest objection — an off-the-shelf **frontier
  SAE (Gemma Scope on Gemma-2-2B)**, which *still* loses to raw (Δ = −0.045,
  CI [−0.062, −0.028]). **Zero of all configurations tested** make SAE features
  significantly beat raw.
- The same SAE features are **causally active** under multi-position
  intervention; detectability of per-feature effects is governed by
  **intervention coverage** (80% power at ≈ 4 patched positions), not SAE
  fidelity.
- **Deployable:** Platt-recalibrated raw-activation selective QA captures
  **≈ 41% of the oracle AURC** improvement on SQuAD with a **~76% ECE reduction**.

## What the result does NOT say

- It does **not** say "SAEs are useless." It is a statement about one
  instrumental task (beating raw activations at *difficulty probing* for
  routing), not about SAE interpretability in general.
- It does **not** establish the *probe-level* null at frontier model scale.
  Probes were tested on ≤ 2.6B-parameter models; frontier scale was tested only
  for **SAE quality** (Gemma Scope), not for a frontier-model difficulty probe.
- It does **not** claim raw activations are a strong difficulty signal in
  absolute terms — best AUROC is ~0.67–0.82 depending on model/layer; only that
  raw ≥ SAE.
- The deployable result is **not** state-of-the-art routing. 41% of oracle AURC
  on SQuAD is modest and honest; generalization to other tasks/models/cost
  structures is not guaranteed and must be re-validated.
- Causal feature effects are **small** (max ≈ 0.009 nats) and **barely
  actionable**: steering the most difficulty-aligned features shifts the gold
  answer by ≤ 0.0066 nats. "Causally active" ≠ "a usable control knob."
- Difficulty labels are **quantized perplexity thresholds / MC correctness** — a
  proxy for difficulty, not ground truth.

## Intended use

- **Research baseline** for SAE-vs-raw probing and selective prediction.
- **`PlattSelectiveQA`** for cost-quality routing when you already have (a) a 1-D
  difficulty score from any upstream probe and (b) a labelled calibration split.
  It calibrates the score (Platt) and answers the easiest `coverage` fraction.

## Out of scope / cautions

- **Not** a safety-critical abstention system. Selective-QA thresholds must be
  validated on your own distribution; calibration does not transfer for free.
- **Not** evidence for or against SAEs as an interpretability method broadly.
- Do not read the causal-ablation effects as a steering/control result.

## Data, models, evaluation

- **Models:** Pythia-410M / -2.8B, Qwen2.5-0.5B-Instruct, Gemma-2-2B (+ Gemma
  Scope SAE). **Data:** HellaSwag, SQuAD, ARC-Easy, TriviaQA (public).
- **Eval:** L1-logistic probes, paired bootstrap CIs, label-permutation
  _p_-values, 5-fold OOF Platt/Isotonic calibration, risk-coverage AURC,
  multi-position causal ablation. Leakage controls: prompt-perplexity
  contamination purge, TF-IDF dedup, and an automated single-feature-AUROC
  leakage audit (`tests/test_leakage_audit.py`).

## Limitations & reproducibility

See `paper_draft.md` §7 (Limitations, each marked addressed/residual) and §5
(Robustness). Reproduce numbers from JSON with `notebooks/results_summary.ipynb`;
restore raw artifacts with `bash download_artifacts.sh`; pinned environment via
`Dockerfile`.

## Citation / contact

Nabin Prasad Dev — see the repository and its successor `fm-difficulty-probe`.
Licensed MIT.
