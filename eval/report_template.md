# Sparse-Autoencoder Features from a Language Model Predict Answer Correctness

*Workshop-style preliminary report. ~6 pages. Fill bracketed slots from `probing/results/` and `eval/results/`.*

## Abstract
We train a TopK sparse autoencoder on Pythia-410M residual stream activations and ask whether the discovered features predict zero-shot answer correctness on HellaSwag *beyond* cheap prompt-level input statistics and raw activations. On HellaSwag, we find [P3−P1 ΔAUROC = X, 95% CI (a,b)]; [SAE vs raw: P3−P2 ΔAUROC = Y]. [State honestly: positive / null.]

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
- Setup: HellaSwag validation, max_seq_len 128; n_train=[ ], n_test=[ ].
- Table 1: AUROC ± CI for P1/P2/P3. Figure 1: `probing/results/auroc.png`.
- [Optional] Figure 3: cascade Pareto — `eval/results/pareto_frontier.png` comparing Pythia-410M ↔ Pythia-2.8B.

## 5. Limitations
- Single benchmark (HellaSwag); thin test set → wide CIs.
- Probes encoded prompt context, not generation sampler dynamics.
- [If null:] SAE features do not buy incremental prediction accuracy over raw activations.

## 6. Future work
Multi-model feature alignment, generative hallucination prediction, steering.
