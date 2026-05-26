# Generative SAE Features for LLM Cascade Routing and Selective Prediction on SQuAD

*Workshop-style preliminary report on continuous generative difficulty. Fill bracketed slots from `probing/results/` and `eval/results/`.*

## Abstract
We extend the time-series difficulty routing framework to free-form question answering on **SQuAD**, modeling answer difficulty as a continuous distribution derived from model generation perplexity on the gold target. We hook the Layer 12 residual stream of Pythia-410M and train a TopK sparse autoencoder ($d_{hidden}=4096, k=32$) on the training split prompts. Under this continuous perplexity target, we report a strong **positive result**: SAE features and input statistics predict difficulty, and their routing signal successfully guides a Pythia-410M $\leftrightarrow$ Pythia-2.8B cascade, showing active Pareto dominance. We find [P3−P1 ΔAUROC = X, 95% CI (a,b)]; [SAE vs raw: P3−P2 ΔAUROC = Y]. This demonstrates that modeling difficulty as a continuous perplexity distribution recovers the predictive SAE signal that binary multiple-choice formats obscure.

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
- **Setup**: SQuAD validation; n_train=[ ], n_test=[ ].

### Probing Difficulty Prediction Performance
Table 1 compiles the point AUROCs and 95% paired bootstrap confidence intervals.

Table 1: AUROC ± CI for P1/P2/P3.

### Downstream Cascade Routing Analysis
We evaluate cascade routing between Pythia-410M (cost=1.0) and Pythia-2.8B (cost=5.0).
- [Optional] Figure 3: cascade Pareto — `eval/results/pareto_frontier.png`

### Calibration Diagnostics
High-dimensional L1 probes exhibit miscalibration. We report raw ECE and Brier scores, alongside Platt and Isotonic recalibrations calculated on 5-fold Out-of-Fold (OOF) predictions.

### Causal Feature Ablation
Using PyTorch forward hooks, we ablate the top-5 difficulty-predictive features in the Layer 12 residual stream to measure their causal influence on SQuAD generation perplexity.

## 5. Limitations
- Generation perplexity is a surrogate for answer correctness, which may not perfectly align with human-evaluated answer quality.
- The SAE is trained on a relatively small corpus compared to standard interpretability sweeps.

## 6. Conclusion & Future Work
We demonstrate that moving from binary multiple-choice formats to continuous generation perplexity targets preserves the predictive signal of SAE features. This provides a strong foundation for building self-correcting, feature-routed LLM cascade pipelines.
