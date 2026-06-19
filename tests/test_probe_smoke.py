"""Synthetic smoke test for the probe ladder — no model, no large activations.

Exercises the full probing pipeline (build_ladder -> make_cv_splits -> fit_ladder
-> paired_bootstrap) on small random data so CI can verify the machinery imports
and runs end-to-end on every push, independently of the multi-GB activation files.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probing.probe import (build_ladder, make_cv_splits, fit_ladder,
                           paired_bootstrap, HEADLINE_PAIRS)


def test_probe_ladder_runs_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 240
    # A weak planted signal so AUROC is defined and slightly above chance.
    y = rng.integers(0, 2, size=n)
    signal = (y[:, None] * 0.6 + rng.normal(size=(n, 4)))
    input_stats = np.concatenate([signal[:, :2], rng.normal(size=(n, 6))], axis=1)  # (n, 8)
    raw_agg = np.concatenate([signal, rng.normal(size=(n, 12))], axis=1)            # (n, 16)
    sae_agg = rng.normal(size=(n, 32))                                              # (n, 32)

    df = pd.DataFrame({
        "activity_label": rng.choice(["a", "b", "c"], size=n),
        "split": ["train"] * 168 + ["test"] * 72,
    })
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    probes = build_ladder(input_stats, raw_agg, sae_agg)
    assert set(probes) == {"P1_InputStats", "P2_InputStats_Raw", "P3_InputStats_SAE",
                           "P4_RawOnly", "P5_SAEOnly"}

    splits = make_cv_splits(df, y[train_mask], train_mask, seed=0)
    results, preds = fit_ladder(probes, y[train_mask], y[test_mask], train_mask, test_mask, splits)
    names = list(probes.keys())
    auroc_ci, delta_ci = paired_bootstrap(preds, y[test_mask], names, HEADLINE_PAIRS, n_boot=200, seed=0)

    for nme in names:
        assert 0.0 <= results[nme]["AUROC"] <= 1.0
        lo, hi = auroc_ci[nme]
        assert lo <= hi
    # The planted-signal probes should clear chance on this easy synthetic set.
    assert results["P2_InputStats_Raw"]["AUROC"] > 0.55


if __name__ == "__main__":
    test_probe_ladder_runs_on_synthetic_data()
    print("probe-ladder smoke OK")
