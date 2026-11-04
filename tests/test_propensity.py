"""Unit tests for propensity metrics, calibration, and decile lift."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from custintel.propensity import (
    _bootstrap_auc_ci,
    _bootstrap_diff_ci,
    _decile_analysis,
    _eval_scores,
)


def test_eval_scores_matches_sklearn():
    """Verify _eval_scores exactly matches scikit-learn standard metrics."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    proba = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.65, 0.15, 0.4, 0.85])

    res = _eval_scores(y_true, proba)

    expected_auc = round(float(roc_auc_score(y_true, proba)), 4)
    expected_pr = round(float(average_precision_score(y_true, proba)), 4)
    expected_brier = round(float(brier_score_loss(y_true, proba)), 4)

    assert res["roc_auc"] == expected_auc
    assert res["pr_auc"] == expected_pr
    assert res["brier"] == expected_brier


def test_bootstrap_auc_ci_contains_estimate():
    """Bootstrap CI must contain the point estimate."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.2, 200)
    proba = y_true * 0.4 + rng.uniform(0, 0.6, 200)

    point_auc = roc_auc_score(y_true, proba)
    ci = _bootstrap_auc_ci(y_true, proba, n_boot=200, alpha=0.05, seed=42)

    assert ci[0] <= point_auc <= ci[1]
    assert 0.0 <= ci[0] < ci[1] <= 1.0


def test_bootstrap_diff_ci():
    """Bootstrap difference CI for superior model must be strictly positive."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.3, 200)
    good_proba = y_true * 0.7 + rng.uniform(0, 0.3, 200)
    random_proba = rng.uniform(0, 1, 200)

    diff_ci = _bootstrap_diff_ci(y_true, good_proba, random_proba, n_boot=200, alpha=0.05, seed=42)
    assert diff_ci[0] > 0.0  # Superior model statistically separates from random


def test_decile_table_properties():
    """Decile table must partition samples and have monotonic cumulative positives."""
    y_true = np.array([1] * 20 + [0] * 80)
    proba = np.linspace(0.01, 0.99, 100)

    dec_table = _decile_analysis(y_true, proba)
    assert len(dec_table) == 10
    total_count = sum(d["count"] for d in dec_table)
    assert total_count == 100
    total_pos = sum(d["positives"] for d in dec_table)
    assert total_pos == 20
    assert dec_table[-1]["cum_positives"] == 20
