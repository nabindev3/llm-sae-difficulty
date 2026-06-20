"""PlattSelectiveQA: calibrate a difficulty score and route by coverage.

Reuses the exact logic validated in eval/recalibrate.py (Platt = logistic with
C=1e6) and eval/selective_prediction.py (risk-coverage AURC by answering the
lowest-difficulty fraction first). Inputs are 1-D difficulty *scores* — higher
means harder/more likely incorrect — so this stays model- and modality-agnostic:
you supply scores from whatever upstream probe (the study's deployable choice is
a raw-activation L1-logistic probe on continuous-perplexity labels).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _as_scores(s):
    s = np.asarray(s, dtype=float).ravel()
    if s.ndim != 1:
        raise ValueError("scores must be 1-D")
    return s


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-width probability bins (matches eval/recalibrate.py)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum():
            ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


class PlattSelectiveQA:
    """Platt-recalibrate a difficulty score and provide coverage-based routing.

    Parameters
    ----------
    coverages : array-like, optional
        Grid of coverage fractions for the risk-coverage curve / AURC
        (default 0.10..1.00 step 0.05, matching the paper).
    """

    def __init__(self, coverages=None):
        self.coverages = (np.round(np.arange(0.10, 1.0001, 0.05), 4)
                          if coverages is None else np.asarray(coverages, float))
        self._platt: LogisticRegression | None = None

    # -- calibration -------------------------------------------------------
    def fit(self, scores, labels) -> "PlattSelectiveQA":
        """Fit Platt scaling: a logistic map from raw score -> calibrated P(hard).

        ``labels`` are 1 for hard/incorrect, 0 for easy/correct. Fit on a
        held-out calibration split (ideally out-of-fold) to avoid optimism.
        """
        s = _as_scores(scores)
        y = np.asarray(labels).ravel().astype(int)
        if s.shape[0] != y.shape[0]:
            raise ValueError("scores and labels length mismatch")
        # C=1e6 ≈ unregularized: pure Platt (sigmoid) recalibration.
        self._platt = LogisticRegression(C=1e6).fit(s.reshape(-1, 1), y)
        return self

    def predict_proba(self, scores) -> np.ndarray:
        """Calibrated P(hard) for each score. Requires :meth:`fit`."""
        if self._platt is None:
            raise RuntimeError("call fit() first")
        return self._platt.predict_proba(_as_scores(scores).reshape(-1, 1))[:, 1]

    # -- routing -----------------------------------------------------------
    def threshold_for_coverage(self, scores, coverage: float) -> float:
        """Score cutoff that answers the easiest ``coverage`` fraction."""
        s = _as_scores(scores)
        coverage = float(np.clip(coverage, 0.0, 1.0))
        return float(np.quantile(s, coverage))

    def route(self, scores, coverage: float = 0.7) -> np.ndarray:
        """Boolean mask: True = answer (low difficulty), False = abstain/escalate.

        Answers the ``coverage`` fraction with the lowest difficulty scores.
        Calibration is monotone, so routing on raw scores == routing on
        calibrated probabilities; calibration matters for the *threshold's*
        interpretability, not the ranking.
        """
        s = _as_scores(scores)
        return s <= self.threshold_for_coverage(s, coverage)

    # -- evaluation --------------------------------------------------------
    def risk_coverage_curve(self, scores, labels):
        """Return (coverages, risks): error rate among the answered fraction."""
        s = _as_scores(scores)
        err = np.asarray(labels, float).ravel()       # 1 = incorrect
        order = np.argsort(s)                          # easiest (lowest score) first
        sorted_err = err[order]
        n = len(s)
        risks = np.array([sorted_err[:max(1, int(round(c * n)))].mean() for c in self.coverages])
        return self.coverages.copy(), risks

    def aurc(self, scores, labels) -> float:
        """Area under the risk-coverage curve (lower is better)."""
        cov, risk = self.risk_coverage_curve(scores, labels)
        return float(np.trapezoid(risk, cov))

    def oracle_aurc(self, labels) -> float:
        """Best achievable AURC: route by the true labels."""
        err = np.sort(np.asarray(labels, float).ravel())   # 0s (correct) first
        n = len(err)
        curve = np.array([err[:max(1, int(round(c * n)))].mean() for c in self.coverages])
        return float(np.trapezoid(curve, self.coverages))

    def fraction_of_oracle(self, scores, labels) -> float:
        """How much of the oracle's AURC improvement over 'answer all' is captured."""
        all_err = float(np.asarray(labels, float).mean())   # AURC of answering everything ≈ base error
        oracle = self.oracle_aurc(labels)
        got = self.aurc(scores, labels)
        denom = all_err - oracle
        return float((all_err - got) / denom) if denom > 0 else float("nan")

    def ece(self, scores, labels, n_bins: int = 10) -> float:
        """Expected calibration error of the calibrated probabilities."""
        return expected_calibration_error(np.asarray(labels, float).ravel(),
                                          self.predict_proba(scores), n_bins)
