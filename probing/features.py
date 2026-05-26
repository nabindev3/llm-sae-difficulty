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
    
    Returns:
        np.ndarray of shape (N, 8) containing the cheap features.
    """
    # 1. Build a global train bigram vocabulary to compute bigram novelty
    train_mask = (df_meta["split"] == "train").values
    train_prompts = df_meta.loc[train_mask, "prompt"].values if train_mask.sum() > 0 else df_meta["prompt"].values
    
    global_train_bigrams = set()
    for prompt in train_prompts:
        words = prompt.lower().split()
        if len(words) >= 2:
            bigrams = { (words[i], words[i+1]) for i in range(len(words)-1) }
            global_train_bigrams.update(bigrams)
            
    # 2. Encode task categories to integer IDs
    categories = df_meta["activity_label"].unique()
    cat_to_id = { cat: idx for idx, cat in enumerate(categories) }
    
    stats = []
    for _, row in df_meta.iterrows():
        prompt = row["prompt"]
        words = prompt.split()
        words_lower = [w.lower() for w in words]
        
        # Stat 1: Character length
        char_len = float(len(prompt))
        
        # Stat 2: Token count
        token_count = float(row["seq_len"])
        
        # Stat 3: Lexical diversity (Type-Token Ratio)
        lexical_diversity = len(set(words_lower)) / (len(words) + 1e-8)
        
        # Stat 4: Perplexity under Pythia-410M
        perplexity = float(row["perplexity"])
        
        # Stat 5: Character-to-token ratio
        char_to_token = char_len / (token_count + 1e-8)
        
        # Stat 6: Capitalization ratio (uppercase letters / total characters)
        alpha_chars = [c for c in prompt if c.isalpha()]
        capitalization_ratio = sum(1 for c in alpha_chars if c.isupper()) / (len(alpha_chars) + 1e-8)
        
        # Stat 7: Task category ID
        cat_id = float(cat_to_id.get(row["activity_label"], 0.0))
        
        # Stat 8: N-gram novelty (fraction of bigrams not found in training vocabulary)
        if len(words_lower) >= 2:
            prompt_bigrams = { (words_lower[i], words_lower[i+1]) for i in range(len(words_lower)-1) }
            novel_count = sum(1 for bg in prompt_bigrams if bg not in global_train_bigrams)
            bigram_novelty = novel_count / len(prompt_bigrams)
        else:
            bigram_novelty = 1.0  # degenerate case
            
        stats.append([
            char_len, token_count, lexical_diversity, perplexity,
            char_to_token, capitalization_ratio, cat_id, bigram_novelty
        ])
        
    return np.array(stats, dtype=np.float64)


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
