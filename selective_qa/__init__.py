"""Platt-recalibrated selective QA — the one deployable result, importable.

The study's positive, deployable finding: a Platt-recalibrated raw-activation
difficulty score gives useful cost-quality routing (≈41% of oracle AURC on
SQuAD, with a ~76% ECE reduction). This package packages that as a tiny,
dependency-light wrapper you can ``pip install -e .`` and use on ANY difficulty
score (from a raw-activation probe, lexical stats, max-softmax, …).

    from selective_qa import PlattSelectiveQA
    qa = PlattSelectiveQA().fit(cal_scores, cal_labels)   # labels: 1 = hard/incorrect
    p_hard  = qa.predict_proba(test_scores)               # calibrated P(hard)
    answer  = qa.route(test_scores, coverage=0.7)         # bool mask: answer vs abstain
    print(qa.aurc(test_scores, test_labels))              # area under risk-coverage
"""
from .predictor import PlattSelectiveQA

__all__ = ["PlattSelectiveQA"]
