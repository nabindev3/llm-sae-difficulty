"""Automated leakage audit: no single feature may ~perfectly predict the label.

Codifies the discipline currently living in code comments. The historical bug
(SQuAD Stat #4 sourcing the gold-answer perplexity from which the difficulty
label is derived) gave that one column a single-feature AUROC of 1.0000. This
test recomputes single-feature AUROC for every cheap input-stat feature and
FAILS LOUDLY if any is >= LEAK_THRESHOLD — so the leak can never silently
re-enter.

Runs in CI on the committed metadata parquets alone (no large tensors). If the
activation safetensors + an SAE checkpoint are present locally (after
`bash download_artifacts.sh`), it additionally audits the raw and SAE feature
columns; otherwise those are skipped.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from probing.features import compute_prompt_stats, INPUT_STAT_NAMES, aggregate_sequence

LEAK_THRESHOLD = 0.98   # a legitimate cheap feature is far below this; ~1.0 == leak

METADATA = [
    "activations/squad_metadata.parquet",
    "activations/hellaswag_metadata.parquet",
    "activations_late/squad_metadata.parquet",
]


def _single_feature_aurocs(X, y):
    """Direction-free single-feature AUROC for every column of X."""
    out = {}
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.ptp(col) == 0 or len(np.unique(y)) < 2:
            out[j] = 0.5
            continue
        a = roc_auc_score(y, col)
        out[j] = max(a, 1 - a)
    return out


@pytest.mark.parametrize("meta_path", METADATA)
def test_no_input_stat_leaks(meta_path):
    full = os.path.join(ROOT, meta_path)
    if not os.path.exists(full):
        pytest.skip(f"{meta_path} not present")
    df = pd.read_parquet(full)
    if "difficulty" not in df.columns:
        pytest.skip("no difficulty column")
    y = df["difficulty"].values
    X = compute_prompt_stats(df)
    aurocs = _single_feature_aurocs(X, y)
    worst_j = max(aurocs, key=aurocs.get)
    worst = aurocs[worst_j]
    assert worst < LEAK_THRESHOLD, (
        f"LEAKAGE in {meta_path}: feature '{INPUT_STAT_NAMES[worst_j]}' has "
        f"single-feature AUROC {worst:.4f} >= {LEAK_THRESHOLD}. "
        f"A cheap stat should not near-perfectly predict difficulty.")


def test_raw_and_sae_features_if_present():
    """Audit raw + SAE columns too, but only when the tensors are downloaded."""
    acts = os.path.join(ROOT, "activations/squad_activations.safetensors")
    ckpt = os.path.join(ROOT, "sae/checkpoints/sae_topk_32.pt")
    meta = os.path.join(ROOT, "activations/squad_metadata.parquet")
    if not (os.path.exists(acts) and os.path.exists(ckpt) and os.path.exists(meta)):
        pytest.skip("activation tensors / SAE checkpoint not present (run download_artifacts.sh)")

    import torch
    from safetensors.torch import load_file
    from sae.load_any import load_sae_any

    df = pd.read_parquet(meta)
    y = df["difficulty"].values
    raw = load_file(acts)["encoder_embeddings"]
    raw_agg = aggregate_sequence(raw.numpy(), df)
    sae, dh = load_sae_any(ckpt, 32, "cpu")
    with torch.no_grad():
        codes, _, _ = sae(raw[:, 0, :].float())
    sae_agg = aggregate_sequence(codes.unsqueeze(1).numpy(), df)

    for name, X in [("raw", raw_agg), ("sae", sae_agg)]:
        worst = max(_single_feature_aurocs(X, y).values())
        assert worst < LEAK_THRESHOLD, (
            f"LEAKAGE: a single {name} feature has AUROC {worst:.4f} >= {LEAK_THRESHOLD}")


if __name__ == "__main__":
    for m in METADATA:
        if os.path.exists(os.path.join(ROOT, m)):
            test_no_input_stat_leaks(m)
            print(f"{m}: no input-stat leak")
    print("leakage audit OK")
