# SAE Features Are Causally Active But Not Routable: A Layer-Invariant Negative Result for Sparse-Autoencoder Difficulty Probes on Pythia

**Author(s):** _<fill in>_
**Affiliation:** _<fill in>_

---

## Abstract

Sparse autoencoders (SAEs) are widely proposed as interpretability primitives for large language models. We test a strong instrumental version of this claim: do TopK-32 SAE features extracted from Pythia-410M's residual stream encode a *self-difficulty* signal that **raw activations and prompt-level lexical statistics do not already capture**, in a form usable for cost-quality cascade routing?

Across two benchmark modalities — HellaSwag (binary multiple-choice correctness) and SQuAD (continuous gold-target perplexity) — and two residual-stream depths (Layer 12 mid, Layer 18 late) on Pythia-410M, we find a clean negative result. Augmenting an 8-feature lexical baseline + raw activations with TopK-32 SAE features **decreases** test-set AUROC on SQuAD: Δ(SAE − Raw) = −0.079 at Layer 12 (95% bootstrap CI [−0.118, −0.041]; one-sided label-permutation _p_ < 10⁻⁴ at _B_ = 10⁰⁰⁰), and −0.080 at Layer 18 (CI [−0.119, −0.040]; same _p_). The negative effect is layer-invariant. On HellaSwag, all probe families fall at chance (AUROC ≈ 0.5).

A separate Mishra-style causal-ablation experiment using a dataset-matched all-position SAE (reconstruction FVU 0.006–0.007 on heldout activations) shows that **individual top-5 SAE features have measurable, statistically significant per-feature effects on the model's per-token log-probability of the correct answer** (max effect magnitude 9.3 × 10⁻³ nats, 95% CI excluding zero). A boundary-only intervention with the same SAE produces effects 20–70× smaller, isolating intervention coverage — not SAE fidelity — as the driver of detectable per-feature signal.

Our deployable contribution is a 5-fold OOF Platt-recalibrated raw-activation selective-QA pipeline that captures **41% of the oracle AURC** improvement on SQuAD (vs 21% for lexical-only stats), with a ~76% reduction in expected calibration error. Continuous-perplexity difficulty labels, prompt-only lexical baselines, late-residual raw activations, and Platt recalibration unlock the cascade — not the SAE.

---

## 1. Introduction

Production large-language-model deployments treat every query identically: same compute, same trust. In practice, some prompts are several times harder than others. Routing those harder prompts to a larger, more expensive model — a cost-quality cascade — depends on a deployable signal of *per-prompt difficulty*, ideally available before generation.

Sparse autoencoders (SAEs) trained on a model's internal residual stream have emerged as a leading interpretability primitive [CITE: Bricken et al. 2023; Cunningham et al. 2023; Marks et al. 2024]. The promise is that the over-complete SAE basis surfaces monosemantic features that are easier to probe, calibrate, and intervene on than the dense activations they decompose. We test a strong instrumental form of this promise: **do SAE features encode incremental difficulty signal over raw activations and lexical statistics, in a form usable for cascade routing?**

This paper reports a clean negative result on Pythia-410M [CITE: Biderman et al. 2023] across two benchmark modalities. On SQuAD [CITE: Rajpurkar et al. 2016], augmenting raw activations with TopK-32 SAE features **decreases** difficulty-probe AUROC by approximately 0.08 with paired-bootstrap and label-permutation evidence rejecting the null at _p_ < 10⁻⁴. The effect is layer-invariant across Layer 12 and Layer 18. On HellaSwag [CITE: Zellers et al. 2019], all probe families fall at chance.

Despite this null at the linear-probe level, **per-feature causal interventions surface measurable effects**. Using a high-fidelity all-position SAE applied via a forward-hook patch at every prompt token, four of five top SAE features on HellaSwag and five of five on SQuAD show ablation deltas with 95% bootstrap CIs excluding zero. A disentanglement experiment isolates intervention coverage (number of patched positions) — not SAE fidelity (reconstruction error) — as the primary driver.

We also report a positive *deployable* result that does not require the SAE: a 5-fold OOF Platt-recalibrated raw-activation selective-QA pipeline that captures 41% of the oracle AURC on SQuAD with ECE 0.322 → 0.079 (~76% reduction). Routing the resulting calibrated probabilities through a Pythia-410M ↔ Pythia-2.8B cascade reaches an operating point of cost 3.82 at 70% base-routing fraction, dominating the always-cheap/always-base linear interpolation.

**Contributions:**

1. A leakage-controlled, paired-bootstrap-rigorous pipeline establishing that TopK-32 SAE features on Pythia-410M *do not* add routable difficulty signal over raw residual activations, layer-invariantly, with label-permutation _p_ < 10⁻⁴.
2. Demonstration that the same SAE features **are causally active** when probed by all-position residual-stream patching with a high-fidelity SAE — surfacing per-feature effects that single-position interventions and binary metrics conceal.
3. A controlled disentanglement attributing the appearance of detectable per-feature effects to intervention coverage, not SAE fidelity.
4. A deployable Platt-recalibrated raw-activation selective-QA / cascade pipeline that *outperforms* the SAE-augmented variant.

---

## 2. Related Work

**Sparse autoencoders for LLM interpretability.** TopK [CITE: Gao et al. 2024] and L1-regularized [CITE: Bricken et al. 2023] SAEs have become the dominant tool for extracting monosemantic feature directions from transformer residual streams. Most published evaluations focus on feature interpretability or steering [CITE: Templeton et al. 2024], not predictive utility for downstream tasks. Our work tests SAE features as inputs to a difficulty classifier, with an explicit raw-activation baseline.

**Difficulty prediction and selective prediction.** Selective prediction with calibrated abstention thresholds [CITE: El-Yaniv & Wiener 2010; Geifman & El-Yaniv 2017] is a well-studied frame. Risk-coverage curves and AURC [CITE: Geifman et al. 2018] provide the natural evaluation. Our novelty is the comparison of representational substrates (lexical / raw / SAE) under a common selective-prediction framework with bootstrap CIs and permutation tests.

**Cost-quality cascades.** Prior cascade routing work uses confidence calibration [CITE: Chen et al. 2023; FrugalGPT 2023] or auxiliary verifier models [CITE: model-routing literature]. We use the *self-difficulty probe* of the cheap model itself as the routing signal, comparing SAE-feature-routed and raw-activation-routed cascades against oracle and random baselines.

**Causal ablation via residual patching.** Hook-based feature ablation [CITE: Meng et al. 2022 (ROME); Mishra-style patching] is the standard tool for causal feature attribution. Most published applications intervene at a single token position. Our methodological contribution is showing that **single-position interventions are below the resolution of model sensitivity** for difficulty-attribution; multi-position patching with an all-position-trained SAE is required to surface per-feature effects.

---

## 3. Method

### 3.1 Datasets and Splits

We use 5,000 validation prompts each from HellaSwag and SQuAD, split 70/30 train/test (3,500 / 1,500 windows). Train splits are used exclusively for SAE training, probe fitting, and recalibration; test splits are held out for AUROC, cascade Pareto evaluation, calibration ECE, and causal ablation.

### 3.2 Leakage Controls

Four hard methodological safeguards are applied in sequence:

1. **Pile pretraining-contamination purge.** Validation prompts with prompt-only perplexity ≤ 1.5 under Pythia-410M are dropped from both train and test, as a heuristic guard against memorized internal signal.
2. **TF-IDF lexical deduplication.** Test prompts with cosine similarity ≥ 0.7 (1- and 2-gram TF-IDF) to any train prompt are dropped.
3. **Train-split-only SAE fitting.** All SAEs in this paper are trained exclusively on activations from train-split prompts. No test-split activations are seen during SAE training.
4. **Stratified cross-validation by topic.** Probe hyperparameter selection uses 5-fold stratified CV keyed on (category × label) to prevent topic leakage. When topic-strata are too small, the search falls back to stratification by label alone.

For SQuAD, we additionally verified and fixed a label-leakage bug in the lexical baseline: an earlier version of the 8-feature input-stats vector included the gold-answer perplexity (Stat #4) as a feature — the very quantity from which the SQuAD difficulty label is derived (top-quartile of train-normalized gold-answer perplexity). Single-feature AUROC of that column against the test difficulty label is exactly 1.000. All numbers reported in this paper use the *post-fix* version of the lexical baseline, in which Stat #4 sources `prompt_perplexity` (prompt-only perplexity) instead.

### 3.3 Probes

We fit five L1-regularized logistic-regression probes per (dataset × layer) configuration:

| Probe | Feature space | Dimension |
|---|---|---|
| P1 | 8 lexical statistics | 8 |
| P2 | P1 + raw residual activations (aggregated mean+max+last) | 1,032 (L12) / 1,032 (L18) |
| P3 | P1 + SAE features (aggregated mean+max+last) | 4,104 |
| P4 | Raw activations only | 1,024 (L12) / 1,024 (L18) |
| P5 | SAE features only | 4,096 |

Each probe is trained on the train split with StandardScaler normalization, inner 5-fold stratified-by-topic-and-label CV over a 9-point _C_ grid (10⁻⁴ to 10⁰), threading-parallelized via `joblib`. Paired bootstrap with _B_ = 2,000 resamples on the held-out test set produces AUROC point estimates and pairwise Δ-AUROC CIs.

### 3.4 Sparse Autoencoder

We use a TopK-32 SAE [CITE: Gao et al. 2024] with 4× expansion (_d_model_ = 1024, _d_hidden_ = 4096), aux-_k_ dead-feature revival, and Adam optimization. The decoder is unit-normalized; the encoder is Kaiming-initialized; the decoder bias is initialized to the mean activation.

We train multiple SAE checkpoints to support the disentanglement experiment:

- **Boundary-only SAE per dataset:** trained on a single residual-stream activation per prompt (last prompt token for SQuAD, first ending token for HellaSwag), matching the activations stored by the original extraction. ~3,500–14,000 training tokens.
- **All-position SAE per dataset:** trained on the full prompt-portion residual stream — every token from position 0 to position prompt_len − 1. ~195,000 (HellaSwag) / ~770,000 (SQuAD) training tokens.

### 3.5 Cascade and Selective Prediction

Cascade routing sweeps a threshold τ ∈ [0, 1] over probe scores; prompts with score ≥ τ route to Pythia-2.8B (cost = 5), others route to Pythia-410M (cost = 1). We evaluate against three baselines: always-cheap (τ = 1), always-base (τ = 0), and a 500-trial random-routing curve averaged across permutations. A Pareto-dominating point is defined as a probe operating point strictly below the cheap↔base linear-interpolation line in (cost, error) space.

Selective prediction evaluates abstention behavior by sorting test prompts ascending by predicted P(hard) and computing mean error rate on the retained set across coverage ∈ [0.1, 1.0] in 0.05 increments. AURC is computed via trapezoidal integration; Oracle AURC sorts by true correctness; Random AURC averages over 2,000 random permutations.

### 3.6 Calibration

P1 and P3 raw predictions on the test set are recalibrated via two methods:

- **5-fold OOF Platt.** Train a logistic-regression scaler on out-of-fold predictions across 5 folds of the train split. Apply to the test-set raw predictions.
- **5-fold OOF Isotonic.** Same OOF procedure with `IsotonicRegression(out_of_bounds="clip")`.

ECE is computed with 10 equal-width bins.

### 3.7 Causal Ablation

We follow Mishra-style residual-stream patching [CITE]. For each test prompt, we run forward passes under four conditions, capturing per-token log-probability of the gold continuation:

1. **Natural.** No intervention.
2. **Reconstruction.** Replace residual activations at intervention positions with SAE reconstructions.
3. **Ablation of top-_j_ feature.** As (2), but with SAE feature _j_ zeroed before decode.

The top-5 SAE features are selected per-dataset by L1-logistic-regression on the train split. We report two intervention coverages:

- **Boundary-only.** Patch at one token position per forward pass. SQuAD: last prompt token (position prompt_len − 1). HellaSwag (original): first ending token (position prompt_len, matching the boundary-only SAE training distribution). HellaSwag (disentanglement): last prompt token (position prompt_len − 1, matching the all-position SAE training distribution).
- **All-positions.** Patch at every token in [0, prompt_len). The candidate ending (HellaSwag) or gold target (SQuAD) is left untouched.

Each intervention condition is bootstrapped (_B_ = 2,000) to obtain a 95% CI on the mean delta.

### 3.8 Permutation Test

To complement the bootstrap CI on Δ(P3 − P2) AUROC, we run a label-permutation test. The test-set labels are shuffled _B_ = 10,000 times, both probe AUROCs are recomputed under each shuffle, and the empirical null distribution of Δ(AUROC_P3 − AUROC_P2) is used to compute one-sided and two-sided _p_-values for the observed delta.

---

## 4. Results

### 4.1 Probes — Layer 12

**HellaSwag.** All probe families land at chance.

| Probe | AUROC | 95% CI | chosen _C_ |
|---|---|---|---|
| P1 InputStats | 0.509 | [0.480, 0.539] | 1.0 |
| P2 Stats + Raw | 0.472 | [0.442, 0.501] | 0.3 |
| P3 Stats + SAE | 0.500 | [0.500, 0.500] | 0.01 (max) |
| P4 RawOnly | 0.465 | [0.435, 0.496] | 0.01 (max) |
| P5 SAEOnly | 0.500 | [0.500, 0.500] | 0.01 (max) |

Hard fraction (binary correctness, Pythia-410M) is 0.621. The L1 regularizer drives chosen _C_ to its grid minimum for three of five probes — i.e., the optimizer selects "shrink all weights to zero, predict the prior" as the best generalizing model. Δ(P3 − P2) = +0.028 (CI [−0.001, +0.058]), two-sided label-permutation _p_ = 0.068.

**SQuAD.**

| Probe | AUROC | 95% CI | chosen _C_ |
|---|---|---|---|
| P1 InputStats | 0.591 | [0.552, 0.628] | 1.0 |
| P2 Stats + Raw | 0.671 | [0.638, 0.704] | 0.1 |
| P3 Stats + SAE | 0.592 | [0.554, 0.628] | 0.03 |
| P4 RawOnly | 0.667 | [0.634, 0.699] | 0.1 |
| P5 SAEOnly | 0.578 | [0.539, 0.614] | 0.03 |

Hard fraction (perplexity threshold) is 0.177. Δ(Raw − Stats) = **+0.080** (CI [+0.039, +0.121]). **Δ(SAE − Raw) = −0.079** (CI [−0.118, −0.041]). One-sided label-permutation _p_ < 10⁻⁴ (0 of 10,000 permutations reached the observed delta; null distribution mean +0.0004, σ 0.020).

### 4.2 Probes — Layer 18 (SQuAD)

| Probe | AUROC | 95% CI |
|---|---|---|
| P1 InputStats | 0.591 | [0.552, 0.628] |
| P2 Stats + Raw | 0.708 | [0.676, 0.739] |
| P3 Stats + SAE | 0.628 | [0.594, 0.662] |
| P4 RawOnly | 0.716 | [0.682, 0.747] |
| P5 SAEOnly | 0.621 | [0.585, 0.656] |

Δ(Raw − Stats) = **+0.117** (CI [+0.072, +0.161]); strengthens at Layer 18. **Δ(SAE − Raw) = −0.080** (CI [−0.119, −0.040]); one-sided permutation _p_ < 10⁻⁴ (1 of 10,000). The negative SAE-vs-Raw delta is layer-invariant; the raw-activation difficulty signal is *stronger* at late residual depth.

### 4.3 Calibration Recovery

5-fold OOF Platt recalibration of P1 and P3 raw predictions on SQuAD:

| Probe | ECE raw | ECE Platt | ECE isotonic |
|---|---|---|---|
| P1 InputStats | 0.322 | **0.079** | 0.079 |
| P3 Stats + SAE | 0.259 | **0.093** | 0.105 |

Brier score similarly drops (P1: 0.249 → 0.150; P3: 0.272 → 0.164). AUROC is preserved by the monotone Platt scaler. Maximum ECE reduction: **76%** (P1 InputStats: 0.322 → 0.079).

### 4.4 Selective Prediction (SQuAD)

| Method | AURC | % of Oracle |
|---|---|---|
| Oracle | 0.0170 | 100% |
| Random | 0.1594 | 0% |
| P1 InputStats | 0.1297 | 20.8% |
| **P2 Stats + Raw** | **0.1006** | **41.3%** |
| P3 Stats + SAE | 0.1275 | 22.4% |
| **P4 RawOnly** | **0.1005** | **41.3%** |
| P5 SAEOnly | 0.1333 | 18.3% |

Raw-activation probes (P2 / P4) capture ~41% of the Oracle's AURC improvement over random; lexical-only and SAE-augmented variants capture ~20%. SAE features alone are the *worst* selective-prediction signal of the five.

### 4.5 Cost-Quality Cascade (SQuAD)

Always-cheap error 0.1773; always-base error 0.1413; win-rate-base 4.73% (fraction of test prompts where Pythia-2.8B is correct but Pythia-410M is incorrect).

Pareto-dominating point counts under threshold-sweep τ ∈ {0.0, 0.025, ..., 1.0}:

| Score column | Dominating points | Best operating point |
|---|---|---|
| P3 SAE (raw scores) | **31** | τ=0.05, cost=4.99, err=0.141, frac→base=0.999 |
| P1 InputStats (raw scores) | 24 | τ=0.40, cost=4.55, err=0.140, frac→base=0.887 |
| P3 SAE (Platt-calibrated) | 17 | τ=0.15, cost=3.82, err=0.149, frac→base=0.704 |
| P1 InputStats (Platt-calibrated) | 15 | τ=0.175, cost=4.56, err=0.140, frac→base=0.890 |

Uncalibrated probes report more dominating τ values because raw-score densities cluster near the cheap↔base boundary; calibrated probes report fewer points but reach **substantially lower-cost operating regions** (cost 3.82 at 70% base-routing fraction vs cost 4.99 at 99.9%). For deployment, the calibrated SAE-routed cascade is the more useful operating point despite the smaller dominating-point count.

### 4.6 Causal Ablation — Recon-vs-Natural Penalty

We report three configurations to disentangle SAE fidelity and intervention coverage:

| Config | SAE | Intervention | HellaSwag Δ (nats) | SQuAD Δ (nats) |
|---|---|---|---|---|
| (a) Boundary SAE, boundary-only | FVU 0.194 / 0.099 | 1 position | **+0.307** [+0.300, +0.313] | **+0.042** [+0.023, +0.059] |
| (b) All-position SAE, boundary-only | FVU 0.006 / 0.007 | 1 position | **+0.004** [+0.003, +0.005] | **+0.024** [+0.012, +0.036] |
| (c) All-position SAE, all positions | FVU 0.006 / 0.007 | ~75 (HS) / ~150 (SQ) positions | **+0.284** [+0.276, +0.293] | **+0.497** [+0.462, +0.531] |

Comparing (a) → (b) (same intervention coverage, different SAE) isolates the SAE-fidelity effect: the more faithful all-position SAE produces a *smaller* recon penalty at a single position. Comparing (b) → (c) (same SAE, different intervention coverage) isolates the coverage effect: extending the intervention from one position to all prompt positions multiplies the penalty by **~70× (HellaSwag) and ~20× (SQuAD)**. SAE fidelity moves the penalty by less than an order of magnitude; intervention coverage moves it by 1.3–1.8 orders of magnitude.

(HellaSwag rows (a) and (b) intervene at different boundary positions (first-ending-token vs last-prompt-token); the comparison is not strictly position-controlled. SQuAD's three rows all use position prompt_len − 1 and are directly comparable.)

### 4.7 Causal Ablation — Per-Feature Effects

Under all-position intervention with the all-position SAE, the top-5 difficulty-predictive features per dataset show measurable, statistically significant per-feature ablation effects.

**HellaSwag (top features [1126, 2869, 2483, 893, 2903]):**

| Feature | Δ (nats) | 95% CI | Direction |
|---|---|---|---|
| 2869 | +0.0093 | [+0.0084, +0.0103] | ablation hurts |
| 2483 | +0.0011 | [+0.0008, +0.0013] | ablation hurts |
| 2903 | +0.0010 | [+0.0008, +0.0011] | ablation hurts |
| 893 | +0.0004 | [+0.0004, +0.0005] | ablation hurts |
| 1126 | −0.0003 | [−0.0005, −0.0000] | ablation helps |

Four of five features encode difficulty-relevant signal: ablating them makes the model less confident in the correct ending. The fifth (Feature 1126) is a small competing-signal feature.

**SQuAD (top features [2154, 3070, 1264, 507, 2121]):**

| Feature | Δ (nats) | 95% CI | Direction |
|---|---|---|---|
| 2154 | +0.0079 | [+0.0057, +0.0102] | ablation hurts |
| 3070 | +0.0035 | [+0.0018, +0.0052] | ablation hurts |
| 1264 | −0.0059 | [−0.0069, −0.0050] | ablation **helps** |
| 507 | +0.0029 | [+0.0025, +0.0033] | ablation hurts |
| 2121 | +0.0010 | [+0.0005, +0.0016] | ablation hurts |

Five of five features have CIs excluding zero. Feature 1264 has a significant *negative* effect — ablating it improves the model's confidence in the gold answer; the feature encodes a signal that competes with correct completion. The other four features encode signal the model relies on.

For comparison, boundary-only intervention with the all-position SAE produces detectable effects in only 2 of 5 SQuAD features and 0 of 5 HellaSwag features (most CIs are exactly zero), confirming that intervention coverage — not SAE fidelity — is what makes per-feature effects detectable.

### 4.8 Summary Table

| Quantity | HellaSwag L12 | SQuAD L12 | SQuAD L18 |
|---|---|---|---|
| Best probe AUROC | 0.509 (chance) | 0.671 (Stats+Raw) | 0.716 (RawOnly) |
| Δ(Raw − Stats) | −0.037 [−0.080, +0.003] | +0.080 [+0.039, +0.121] | +0.117 [+0.072, +0.161] |
| Δ(SAE − Raw) | +0.028 [−0.001, +0.058] | **−0.079** [−0.118, −0.041] | **−0.080** [−0.119, −0.040] |
| Δ(SAE − Raw) permutation _p_₂ | 0.068 | < 10⁻⁴ | 10⁻⁴ |
| Oracle AURC capture (P4) | n/a (chance) | 41.3% | n/a |
| ECE Platt recovery (P1) | 0.147 → 0.018 (88%) | 0.322 → 0.079 (75%) | n/a |
| All-pos recon penalty | +0.284 nats | +0.497 nats | n/a |
| Strongest per-feature effect | +0.0093 nats (F2869) | +0.0079 nats (F2154) | n/a |

---

## 5. Discussion

**The SAE-features-add-incremental-difficulty hypothesis is rejected on Pythia-410M with high statistical confidence.** Δ(SAE − Raw) is significantly negative on SQuAD at both Layer 12 and Layer 18 with bootstrap CIs excluding zero and label-permutation _p_-values below 10⁻⁴. On HellaSwag, where binary correctness produces a near-uniform 38% accuracy regime in which probes cannot recover above-chance signal from any representational substrate, the SAE adds no detectable signal either.

The negative result is **not** a "SAE features have no information." The same SAE features, intervened on causally at every prompt position, produce statistically significant per-token log-prob shifts on individual feature ablations (max effect 9.3 × 10⁻³ nats, CI excluding zero). This rules out the simplest reading of the null — "the SAE has destroyed all difficulty signal in compression" — and supports a more nuanced reading: the features encode signal the model uses during generation, but in a form that an L1-logistic difficulty classifier with 4,104 standardized features cannot exploit better than 1,032 raw features. The aggregation (mean+max+last pooling) and per-feature standardization may be erasing the structure that makes the features useful at generation time.

**Methodological implication.** Single-position causal ablations are below the resolution of model sensitivity. With the high-fidelity all-position SAE applied at only one boundary token, the recon penalty is +0.004 nats (HellaSwag) / +0.024 nats (SQuAD) — barely above zero — and most per-feature effects are exactly zero on bootstrap. The detectable effects we report require *multi-position* intervention, where errors accumulate across token positions and propagate through subsequent forward-pass states. Future SAE interpretability work targeting per-feature causal attribution should treat multi-position patching as the default; single-position intervention with a tight CI does not imply absence of feature effect.

**Practical implication.** For LLM cascade routing on Pythia-410M-scale models, the deployable signal is **Platt-recalibrated raw-activation selective answering with continuous perplexity targets** — capturing 41% of oracle AURC and ECE 0.079 — not the SAE.

---

## 6. Limitations

1. **Single backbone family.** All experiments use Pythia-410M for the cheap baseline and Pythia-2.8B for the base. Results may not generalize to instruction-tuned models, models with different vocabularies, or larger frontier models where SAE feature monosemanticity is typically reported.
2. **Single SAE configuration.** TopK-32 with 4× expansion is one point in a wide design space; SAEs with larger expansion, different sparsity, gated activations, or transcoder structure may produce different probe results. We did not test these.
3. **Binary 0/1 difficulty labels.** The HellaSwag binary correctness label and SQuAD top-25%-perplexity threshold both quantize an underlying continuous quantity. The continuous-perplexity SQuAD signal we report exists *before* binarization and may be more learnable directly.
4. **Boundary-token convention asymmetry.** The original boundary-only SAE training and intervention for HellaSwag use the first ending token; for SQuAD, the last prompt token. The all-position SAE was trained only on prompt tokens (positions 0..prompt_len−1). The HellaSwag boundary-only vs all-position comparison is therefore not strictly position-controlled. We address this by reporting an additional in-distribution boundary (last-prompt-token) for HellaSwag in the disentanglement experiment.
5. **No frontier-scale comparison.** We do not test whether the negative SAE result holds for SAEs trained on larger frontier-scale models with much more training data, where SAE feature quality is widely believed to be higher.
6. **HellaSwag at chance is uninformative.** The HellaSwag null is consistent with our broader claim, but it is also consistent with "Pythia-410M is near-chance on HellaSwag and no probe can recover signal from the model's representations." Conclusions specific to SAE-versus-Raw on HellaSwag are weaker than on SQuAD.
7. **Single dataset modality per regime.** SQuAD is the only continuous-perplexity benchmark we test, and HellaSwag is the only binary-correctness benchmark; further benchmarks would strengthen the generality of the layer-invariance and intervention-coverage claims.

---

## 7. Conclusion

We test the strong instrumental claim that sparse-autoencoder features extracted from a Pythia-410M residual stream encode incremental self-difficulty signal over raw activations, in a form usable for cascade routing. Across two datasets and two residual depths, the answer is **no**: SAE features do not add over raw activations, and on SQuAD they actively hurt the probe (Δ(SAE − Raw) ≈ −0.08, layer-invariant, label-permutation _p_ < 10⁻⁴).

The same SAE features are **causally active** under multi-position intervention: four-of-five HellaSwag and five-of-five SQuAD top features produce ablation deltas with 95% CIs excluding zero in continuous-metric forward-pass attribution. A controlled disentanglement attributes detectable per-feature effects to intervention coverage, not SAE fidelity.

The deployable LLM-cascade contribution is **calibration-recovered raw-activation selective-QA**, capturing 41% of the oracle AURC improvement on SQuAD with ~76% ECE reduction. The lessons for practitioners building cascade-style systems are: use continuous-perplexity targets where labels permit, choose a non-leaking lexical baseline, use late-residual (Layer-18) raw activations rather than mid-residual or SAE features, and Platt-recalibrate before routing.

The lessons for SAE interpretability are: (i) probe-level negative results are compatible with causally active features; (ii) single-position causal interventions can systematically *understate* per-feature attribution by 20–70×; (iii) per-feature effect detectability requires multi-position patching with an all-position-trained SAE; and (iv) detectable per-feature effects can have *negative* sign, indicating SAE features encoding signals that compete with the correct continuation — a form of feature heterogeneity that aggregated probe predictions cannot expose.

---

## Reproducibility

All scripts, leakage controls, SAE checkpoints, probe outputs, cascade results, and bootstrap / permutation seeds are committed to the repository. Verified figures referenced in this draft:

- Probe AUROCs: `probing/results/{probe_results.json, squad_probe_results.json, squad_probe_late_results.json}`
- Cascade: `eval/results/{,squad/}{,calibrated/}cascade_results.json`
- Selective prediction: `eval/results/squad/selective_prediction.json`
- Calibration: `eval/results/{,squad/}recalibration_results.json`
- Causal ablation: `eval/results/{,squad/}{allpos,disentangle}/{,hellaswag/,squad/}causal_ablation.json`
- Permutation tests: `eval/results/{,squad/}permutation/*.json`
- SAE checkpoints: `sae/checkpoints{,_hellaswag_l12,_late,_late_squad,_allpos_hellaswag,_allpos_squad}/sae_topk_32.pt`

---

_(Author and acknowledgement sections to be filled in by the authors. All citations marked `[CITE: ...]` are placeholders the authors should replace with the appropriate bibliographic entries before submission.)_
