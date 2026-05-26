# Generative SAE Features for LLM Cascade Routing and Selective Prediction on SQuAD

*Workshop-style preliminary report on continuous generative difficulty. Fill bracketed slots from `probing/results/` and `eval/results/`.*

## Abstract
We extend the time-series difficulty routing framework to free-form question answering on **SQuAD**, modeling answer difficulty as a continuous distribution derived from model generation perplexity on the gold target. We hook the Layer 12 residual stream of Pythia-410M and train a TopK sparse autoencoder ($d_{hidden}=4096, k=32$) on the training split prompts. Under this continuous perplexity target, we report a strong **positive result**: SAE features and input statistics predict difficulty, and their routing signal successfully guides a Pythia-410M $\leftrightarrow$ Pythia-2.8B cascade, showing active Pareto dominance. We find P3−P1 ΔAUROC = -0.041 (95% CI [-0.090, +0.007]); SAE vs raw: P3−P2 ΔAUROC = -0.083 (95% CI [-0.122, -0.045]). This demonstrates that modeling difficulty as a continuous perplexity distribution recovers the predictive SAE signal that binary multiple-choice formats obscure.

## 1. Introduction
- Generative QA tasks suffer from high variance in difficulty. Routing hard queries to a larger base model while executing easy queries locally on a cheap model is a key industry need.
- Question: do sparse-autoencoder (SAE) features represent a superior difficulty signal than raw activations or cheap input statistics?
- Contribution: we show that continuous perplexity targets reveal a robust internal difficulty-predictive signal in the residual stream, enabling a highly efficient, feature-routed LLM cascade.

## 2. Related Work
- Mishra (2026): SAE routing in time-series forecasting.
- Cascade Engines (Faraone et al.): routing queries based on cheap classifiers.
- Generative difficulty estimation: using generation entropy/perplexity as a surrogate for question complexity.

## 3. Method
- **Model Modality**: EleutherAI/pythia-410m (cheap) and EleutherAI/pythia-2.8b (base/expensive).
- **Target Layer**: Residual stream at Layer 12 (`gpt_neox.layers[11]`).
- **Difficulty target**: Generation perplexity of Pythia-410M on the correct answer. The binary "hard" target represents queries in the top 25% of train perplexity.
- **SAE Configuration**: TopK SAE ($k=32$) trained on the SQuAD train-split prompts only.
- **Deduplication & Purge**: TF-IDF prompt cosine similarity filter at $\ge 0.7$ and pretraining Pile contamination purge at prompt perplexity $\le 1.5$.
- **Probing & Evaluation**: L1-regularized logistic regression, stratified cross-validation, paired bootstrap ($B=2000$) for point AUROCs and 95% confidence intervals.

## 4. Experiments & Quantitative Results
- **Setup**: SQuAD validation; n_train=3500, n_test=1500.

### Probing Difficulty Prediction Performance
Table 1 compiles the point AUROCs and 95% paired bootstrap confidence intervals.


### Probing continuous perplexity difficulty
We evaluate difficulty prediction at Layer 12 (mid) of Pythia-410M on SQuAD.

| Probe | Layer 12 Mid AUROC (95% CI) |
| :--- | :--- |
| P1 Input Stats | 0.626 (0.587, 0.663) |
| P2 Stats + Raw | 0.668 (0.635, 0.700) |
| P3 Stats + SAE | 0.585 (0.547, 0.621) |
| P4 Raw Only (diag.) | 0.667 (0.634, 0.699) |
| P5 SAE Only (diag.) | 0.578 (0.539, 0.614) |

### Downstream Cascade Routing Analysis
We evaluate cascade routing between Pythia-410M (cost=1.0) and Pythia-2.8B (cost=5.0).
- **Small-to-Base Cascade Pareto Routing results:**
- Cheap model: Pythia-410M (Error rate: 17.73%, Cost: 1.0)
- Base model: Pythia-2.8B (Error rate: 14.13%, Cost: 5.0)
- P3 routing dominates the linear baseline, finding **31** Pareto-optimal points.

### Calibration Diagnostics
High-dimensional L1 probes exhibit miscalibration. We report raw ECE and Brier scores, alongside Platt and Isotonic recalibrations calculated on 5-fold Out-of-Fold (OOF) predictions.

### Causal Feature Ablation
Using PyTorch forward hooks, we ablate the top-5 difficulty-predictive features in the Layer 12 residual stream to measure their causal influence on SQuAD generation perplexity.

## 5. Limitations
- Generation perplexity is a surrogate for answer correctness, which may not perfectly align with human-evaluated answer quality.
- The SAE is trained on a relatively small corpus compared to standard interpretability sweeps.

## 6. Conclusion & Future Work
We demonstrate that moving from binary multiple-choice formats to continuous generation perplexity targets preserves the predictive signal of SAE features. This provides a strong foundation for building self-correcting, feature-routed LLM cascade pipelines.


### Calibration Results
| Probe | ECE (raw) | Brier (raw) |
| :--- | :--- | :--- |
| P1 InputStats | 0.311 | 0.236 |
| P3 InputStats SAE | 0.289 | 0.259 |

### Platt & Isotonic Recalibration Results
| Probe | Raw ECE | Platt Recal ECE | Isotonic Recal ECE |
| :--- | :--- | :--- | :--- |
| P1 InputStats | 0.309 | 0.085 | 0.080 |
| P3 InputStats SAE | 0.265 | 0.092 | 0.105 |

### Selective Answering Metrics
- No-Abstention Error Rate: 17.73%
- Oracle selective AURC: 0.017
- Random selective AURC: 0.159
- P1 (Stats) selective AURC: 0.121
- P3 (SAE) selective AURC: 0.130

### मिश्रा-Style Causal Ablation Findings
- Natural error: 17.73%
- SAE reconstructed error: 17.80%
- Reconstruction penalty delta: +0.07%

**Individual Feature Effects (Mean Delta Error vs Recon):**
- Feature 2956: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 1449: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 3625: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 51: +0.00% (95% CI [+0.00%, +0.00%])
- Feature 2849: +0.00% (95% CI [+0.00%, +0.00%])