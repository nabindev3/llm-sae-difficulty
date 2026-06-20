"""Tests for the importable PlattSelectiveQA wrapper (the deployable result)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selective_qa import PlattSelectiveQA


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)                      # 1 = hard/incorrect
    # A score that ranks harder items higher, plus noise (uncalibrated scale).
    scores = 3.0 * y + rng.normal(size=n) + 5.0
    return scores, y


def test_calibration_and_routing():
    s, y = _synthetic()
    cut = 1000
    qa = PlattSelectiveQA().fit(s[:cut], y[:cut])

    p = qa.predict_proba(s[cut:])
    assert p.min() >= 0.0 and p.max() <= 1.0
    # Platt output is a probability with mean near the base rate.
    assert abs(p.mean() - y[cut:].mean()) < 0.15

    # Routing answers ~the requested coverage and abstains on the rest.
    ans = qa.route(s[cut:], coverage=0.7)
    assert abs(ans.mean() - 0.7) < 0.05
    # Answered set should be EASIER (lower error) than the abstained set.
    assert y[cut:][ans].mean() < y[cut:][~ans].mean()


def test_aurc_beats_random_and_bounds():
    s, y = _synthetic()
    qa = PlattSelectiveQA()
    aurc = qa.aurc(s, y)
    oracle = qa.oracle_aurc(y)
    base = y.mean()                                    # AURC of answering everything
    assert oracle <= aurc <= base + 1e-9              # selective beats answer-all, worse than oracle
    frac = qa.fraction_of_oracle(s, y)
    assert 0.0 < frac <= 1.0 + 1e-9                    # captures some of the oracle gap

    # ECE needs the fitted calibrator; small on this well-specified synthetic set.
    qa.fit(s, y)
    assert qa.ece(s, y) < 0.1


if __name__ == "__main__":
    test_calibration_and_routing()
    test_aurc_beats_random_and_bounds()
    print("selective_qa tests OK")
