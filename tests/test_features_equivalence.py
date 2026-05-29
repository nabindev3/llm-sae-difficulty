"""Verify the vectorized compute_prompt_stats produces identical output to
the original per-row implementation."""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from probing.features import compute_prompt_stats, INPUT_STAT_NAMES


# A faithful copy of the prior per-row implementation, kept here as a reference
# oracle to verify the vectorized version. If this ever needs to change, the
# vectorized version must change to match.
def compute_prompt_stats_reference_loop(df_meta: pd.DataFrame) -> np.ndarray:
    train_mask = (df_meta["split"] == "train").values
    train_prompts = (
        df_meta.loc[train_mask, "prompt"].values
        if train_mask.sum() > 0
        else df_meta["prompt"].values
    )
    global_train_bigrams = set()
    for prompt in train_prompts:
        words = prompt.lower().split()
        if len(words) >= 2:
            global_train_bigrams.update(
                (words[i], words[i + 1]) for i in range(len(words) - 1)
            )

    categories = df_meta["activity_label"].unique()
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}

    stats = []
    for _, row in df_meta.iterrows():
        prompt = row["prompt"]
        words = prompt.split()
        words_lower = [w.lower() for w in words]
        char_len = float(len(prompt))
        token_count = float(row["seq_len"])
        lexical_diversity = len(set(words_lower)) / (len(words) + 1e-8)
        if "prompt_perplexity" in df_meta.columns:
            perplexity = float(row["prompt_perplexity"])
        else:
            perplexity = float(row["perplexity"])
        char_to_token = char_len / (token_count + 1e-8)
        alpha_chars = [c for c in prompt if c.isalpha()]
        capitalization_ratio = sum(1 for c in alpha_chars if c.isupper()) / (
            len(alpha_chars) + 1e-8
        )
        cat_id = float(cat_to_id.get(row["activity_label"], 0.0))
        if len(words_lower) >= 2:
            prompt_bigrams = {
                (words_lower[i], words_lower[i + 1])
                for i in range(len(words_lower) - 1)
            }
            novel_count = sum(
                1 for bg in prompt_bigrams if bg not in global_train_bigrams
            )
            bigram_novelty = novel_count / len(prompt_bigrams)
        else:
            bigram_novelty = 1.0
        stats.append([
            char_len, token_count, lexical_diversity, perplexity,
            char_to_token, capitalization_ratio, cat_id, bigram_novelty,
        ])
    return np.array(stats, dtype=np.float64)


def test_against_dataset(label, metadata_path):
    df = pd.read_parquet(metadata_path)
    print(f"\n=== {label}  (N={len(df)}) ===")

    t0 = time.perf_counter()
    reference = compute_prompt_stats_reference_loop(df)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    vectorized = compute_prompt_stats(df)
    t_vec = time.perf_counter() - t0

    assert reference.shape == vectorized.shape, (
        f"Shape mismatch: ref={reference.shape}, vec={vectorized.shape}"
    )
    diffs = np.abs(reference - vectorized).max(axis=0)

    print(f"  Reference (iterrows):  {t_ref:6.3f} s")
    print(f"  Vectorized:            {t_vec:6.3f} s")
    print(f"  Speedup:               {t_ref / t_vec:5.1f}x")
    print(f"  Max |Δ| per column (8 stats):")
    for name, d in zip(INPUT_STAT_NAMES, diffs):
        print(f"    {name:22s}  {d:.3e}")

    np.testing.assert_allclose(
        reference, vectorized, rtol=1e-12, atol=1e-12,
        err_msg=f"Vectorized output diverged from reference loop on {label}",
    )
    print(f"  ✓ numerical equivalence (rtol=1e-12, atol=1e-12)")


if __name__ == "__main__":
    for label, path in [
        ("HellaSwag L12", "activations/hellaswag_metadata.parquet"),
        ("SQuAD L12",     "activations/squad_metadata.parquet"),
    ]:
        if os.path.exists(path):
            test_against_dataset(label, path)
        else:
            print(f"[skip] {label}: {path} not present")
