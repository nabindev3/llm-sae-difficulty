"""Feature extractors for the LLM difficulty probe.

Eight classical prompt-level cheap statistics:
  1. Character length
  2. Token count
  3. Lexical diversity (Type-Token Ratio)
  4. Perplexity under small model
  5. Character-to-token ratio
  6. Capitalization ratio
  7. Task category ID (label-encoded activity)
  8. Bigram novelty (fraction of prompt word bigrams that are novel vs train vocabulary)
"""
import numpy as np
import pandas as pd


def compute_prompt_stats(df_meta: pd.DataFrame) -> np.ndarray:
    """Compute 8 prompt-level cheap statistics for all rows in the metadata.

    Vectorized version: uses pandas string accessors and list comprehensions
    over numpy arrays, avoiding df.iterrows() Python overhead. Produces
    numerically identical output to the prior per-row implementation
    (verified by element-wise allclose; see tests/test_features_equivalence.py).

    Returns:
        np.ndarray of shape (N, 8) containing the cheap features.
    """
    # --- 1. Build global train bigram vocabulary for novelty (Stat 8) ----------
    train_mask = (df_meta["split"] == "train").values
    if train_mask.sum() > 0:
        train_prompts = df_meta.loc[train_mask, "prompt"].values
    else:
        train_prompts = df_meta["prompt"].values

    global_train_bigrams: set = set()
    for prompt in train_prompts:
        words = prompt.lower().split()
        if len(words) >= 2:
            global_train_bigrams.update(zip(words[:-1], words[1:]))

    # --- 2. Encode task categories to integer IDs (Stat 7) ---------------------
    categories = df_meta["activity_label"].unique()
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}

    # --- 3. Extract prompts once and pre-split for word-level stats ------------
    prompts = df_meta["prompt"].to_numpy()
    words_per_prompt = [p.split() for p in prompts]
    words_lower_per_prompt = [[w.lower() for w in ws] for ws in words_per_prompt]

    # Vectorized stats (Stats 1, 2, 5)
    char_len = np.array([len(p) for p in prompts], dtype=np.float64)
    token_count = df_meta["seq_len"].to_numpy(dtype=np.float64)
    char_to_token = char_len / (token_count + 1e-8)

    # Stat 3: Lexical diversity (Type-Token Ratio)
    lexical_diversity = np.array(
        [len(set(ws)) / (len(ws) + 1e-8) for ws in words_lower_per_prompt],
        dtype=np.float64,
    )

    # Stat 4: Perplexity — PROMPT-only, never the gold target.
    # For SQuAD, the `perplexity` column is the gold-answer perplexity from
    # which the difficulty label is derived (extract_activations.py:280) — using
    # it here would be direct label leakage (single-feature AUROC = 1.000).
    # `prompt_perplexity` is the prompt-only perplexity used for Pile-contamination
    # checks. HellaSwag stores prompt-only ppl in `perplexity` and has no
    # `prompt_perplexity` column, so we fall back to `perplexity` there.
    ppl_col = "prompt_perplexity" if "prompt_perplexity" in df_meta.columns else "perplexity"
    perplexity = df_meta[ppl_col].to_numpy(dtype=np.float64)

    # Stat 6: Capitalization ratio. We need (Unicode uppercase letters) /
    # (Unicode alphabetic chars). A pandas regex like r"[A-Z]"/r"[A-Za-z]"
    # would only match ASCII and undercount accented or non-Latin letters,
    # diverging from the reference implementation that uses str.isalpha()/
    # str.isupper() (which are Unicode-aware). We do a single-pass per-prompt
    # counter — still O(total chars) but ~2× faster than the prior reference
    # which built a separate alpha-char list before counting uppercase.
    def _cap_ratio(p: str) -> float:
        alpha_n = 0
        upper_n = 0
        for c in p:
            if c.isalpha():
                alpha_n += 1
                if c.isupper():
                    upper_n += 1
        return upper_n / (alpha_n + 1e-8)
    capitalization_ratio = np.fromiter(
        (_cap_ratio(p) for p in prompts),
        dtype=np.float64,
        count=len(prompts),
    )

    # Stat 7: Task category ID (numeric encoding)
    cat_id = df_meta["activity_label"].map(cat_to_id).fillna(0.0).to_numpy(dtype=np.float64)

    # Stat 8: Bigram novelty. Pre-compute per-prompt bigram sets and count novels.
    def _novelty(ws_lower):
        if len(ws_lower) < 2:
            return 1.0  # degenerate case (matches prior implementation)
        prompt_bigrams = set(zip(ws_lower[:-1], ws_lower[1:]))
        novel_count = sum(1 for bg in prompt_bigrams if bg not in global_train_bigrams)
        return novel_count / len(prompt_bigrams)

    bigram_novelty = np.array(
        [_novelty(ws) for ws in words_lower_per_prompt], dtype=np.float64
    )

    # Stack to (N, 8) in the same column order as INPUT_STAT_NAMES
    return np.column_stack([
        char_len,
        token_count,
        lexical_diversity,
        perplexity,
        char_to_token,
        capitalization_ratio,
        cat_id,
        bigram_novelty,
    ])


INPUT_STAT_NAMES = [
    "char_length", "token_count", "lexical_diversity", "perplexity",
    "char_to_token", "capitalization_ratio", "category_id", "bigram_novelty"
]


def aggregate_sequence(seq_tensor: np.ndarray, meta_df: pd.DataFrame = None) -> np.ndarray:
    """Pools sequence activations / SAE codes across prompt token lengths.
    
    Args:
        seq_tensor: np.ndarray of shape (N, max_seq_len, d)
        meta_df: Optional pd.DataFrame containing actual prompt token lengths ('seq_len')
                 to filter out zero padding in mean/max pooling.
                 
    Returns:
        np.ndarray of shape (N, 3*d) containing concat(mean, max, last) pooled features.
    """
    if seq_tensor.ndim != 3:
        raise ValueError(f"Expected (N, seq, d), got {seq_tensor.shape}")
        
    N, max_seq, d = seq_tensor.shape
    if max_seq == 1:
        return seq_tensor[:, 0, :]
        
    mean_list = []
    max_list = []
    last_list = []
    
    for i in range(N):
        # Determine valid prompt tokens to exclude zero padding from pooling
        seq_len = int(meta_df.iloc[i]["seq_len"]) if meta_df is not None else max_seq
        seq_len = min(seq_len, max_seq)
        
        valid_seq = seq_tensor[i, :seq_len, :] # Shape: (seq_len, d)
        
        mean_list.append(valid_seq.mean(axis=0))
        max_list.append(valid_seq.max(axis=0))
        last_list.append(valid_seq[-1, :]) # last prompt token (boundary token)
        
    mean = np.array(mean_list)
    mx = np.array(max_list)
    last = np.array(last_list)
    
    return np.concatenate([mean, mx, last], axis=1)
