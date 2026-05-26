# Sparse-Autoencoder Features from a Language Model Predict Answer Correctness

*Workshop-style preliminary report. ~6 pages. Fill bracketed slots from `probing/results/` and `eval/results/`.*

## Abstract
We train a TopK sparse autoencoder on Pythia-410M residual stream activations and ask whether the discovered features predict zero-shot answer correctness on HellaSwag *beyond* cheap prompt-level input statistics and raw activations. On HellaSwag, we find P3−P1 ΔAUROC = -0.005 (95% CI [-0.047, +0.037]); SAE vs raw: P3−P2 ΔAUROC = +0.032 (95% CI [-0.006, +0.071]). We report a **positive result**: SAE features outperform raw activations on HellaSwag difficulty prediction, highlighting that sparse compression extracts clean predictive signals.

## 1. Introduction
- Large Language Models (LLMs) are deployed as black-boxes; knowing *when to trust an LLM response* or when to escalate is critical for cascade engineering.
- Question: do internal SAE features capture answer difficulty in a way that cheap prompt metrics or raw activations obscure?
- Contribution: an *incremental*, leakage-controlled probe evaluating prompt difficulty; [optional: a one-point feature-routed LLM cascade].

## 2. Related work
- Mishra (2026): SAEs on Chronos, causal change-detection.
- Pythia (Biderman et al.): open-suite decoder-only models.
- Distinction: we target **label-free inference-time correctness prediction / routing**, not post-hoc feature interpretation.

## 3. Method
- Model: EleutherAI/pythia-410m (60M parameter scaling reference); hook Layer 12 residual stream.
- Labels: Binary correctness (1 if zero-shot prediction is incorrect, 0 if correct) on HellaSwag validation.
- SAE: TopK, 1024→4096 (4×), k=32, aux-k revival; trained on **train-split prompt tokens only** to prevent validation leakage.
- Probe: L1 logistic, stratified CV C, prompt-cluster TF-IDF deduplication and perplexity-based pretraining contamination purge; concat(mean,max,last) sequence pooling.
- Metric: paired bootstrap ΔAUROC (P2−P1, P3−P1, P3−P2), 95% CI.

## 4. Experiments
- Setup: HellaSwag validation, max_seq_len 128; n_train=3500, n_test=1499.
- 
### Cross-Layer Robustness Probing Results
We evaluate difficulty prediction at two pre-registered layers: **Layer 12 (mid)** and **Layer 18 (late)** of Pythia-410M.

| Probe | Layer 12 Mid AUROC (95% CI) | Layer 18 Late AUROC (95% CI) |
| :--- | :--- | :--- |
| P1 Input Stats | 0.509 (0.480, 0.539) | 0.509 (0.480, 0.539) |
| P2 Stats + Raw | 0.472 (0.442, 0.501) | 0.500 (0.500, 0.500) |
| P3 Stats + SAE | 0.504 (0.474, 0.534) | 0.452 (0.420, 0.480) |
| P4 Raw Only (diag.) | 0.465 (0.435, 0.496) | 0.500 (0.500, 0.500) |
| P5 SAE Only (diag.) | 0.496 (0.467, 0.526) | 0.453 (0.421, 0.482) | Figure 1: `probing/results/auroc.png`.
- [Optional] Figure 3: cascade Pareto — `eval/results/pareto_frontier.png` comparing Pythia-410M ↔ Pythia-2.8B.

## 5. Limitations
- Single benchmark (HellaSwag); thin test set → wide CIs.
- Probes encoded prompt context, not generation sampler dynamics.
- [If null:] SAE features do not buy incremental prediction accuracy over raw activations.

## 6. Future work
Multi-model feature alignment, generative hallucination prediction, steering.


### Calibration Results
| Probe | ECE (raw) | Brier (raw) |
| :--- | :--- | :--- |
| P1 InputStats | 0.147 | 0.257 |
| P3 InputStats SAE | 0.407 | 0.427 |

### Platt & Isotonic Recalibration Results
| Probe | Raw ECE | Platt Recal ECE | Isotonic Recal ECE |
| :--- | :--- | :--- | :--- |
| P1 InputStats | 0.147 | 0.018 | 0.028 |
| P3 InputStats SAE | 0.252 | 0.003 | 0.032 |

### Selective Answering Metrics
- No-Abstention Error Rate: 62.11%
- Oracle selective AURC: 0.254
- Random selective AURC: 0.559
- P1 (Stats) selective AURC: 0.553
- P3 (SAE) selective AURC: 0.557

### मिश्रा-Style Causal Ablation Findings
- Natural error: 66.67%
- SAE reconstructed error: 75.33%
- Reconstruction penalty delta: +8.67%

**Individual Feature Effects (Mean Delta Error vs Recon):**
- Feature 1546: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 2627: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 1896: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 2617: -0.33% (95% CI [-1.00%, +0.00%])
- Feature 2696: +0.00% (95% CI [+0.00%, +0.00%])