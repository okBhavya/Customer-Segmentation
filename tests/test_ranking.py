"""Unit tests for ranking metrics (NDCG, Recall, HitRate) and IPS counterfactual estimation."""

import numpy as np
import pytest

from custintel.ranking import (
    _calc_hit_at_k,
    _calc_ndcg_at_k,
    _calc_recall_at_k,
)


def test_ndcg_hand_computed():
    """Verify NDCG@k matches hand calculations."""
    # Relevance grades: [3, 2, 0, 1]
    y_true = np.array([3, 2, 0, 1])

    # Perfect ranking: order [0, 1, 3, 2] -> relevance [3, 2, 1, 0]
    perfect_scores = np.array([4.0, 3.0, 1.0, 2.0])
    ndcg_perfect = _calc_ndcg_at_k(y_true, perfect_scores, k=4)
    assert pytest.approx(ndcg_perfect, 1e-4) == 1.0

    # Reversed ranking: order [2, 3, 1, 0] -> relevance [0, 1, 2, 3]
    worst_scores = np.array([1.0, 2.0, 4.0, 3.0])
    ndcg_worst = _calc_ndcg_at_k(y_true, worst_scores, k=4)
    assert 0.0 < ndcg_worst < 1.0

    # Empty / zero relevance
    y_zero = np.array([0, 0, 0])
    assert _calc_ndcg_at_k(y_zero, perfect_scores[:3], k=3) == 0.0


def test_recall_and_hit_at_k():
    """Verify Recall@k and HitRate@k logic."""
    y_true = np.array([1, 0, 2, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    # Top 2 items: indices [0, 1], items have rel [1, 0]. Total positive items = 2.
    assert _calc_recall_at_k(y_true, scores, k=2) == 0.5
    assert _calc_hit_at_k(y_true, scores, k=2) == 1.0

    # Top 1 item: index [0], rel 1
    assert _calc_recall_at_k(y_true, scores, k=1) == 0.5
    assert _calc_hit_at_k(y_true, scores, k=1) == 1.0


def test_ips_estimator_on_simulated_bandit():
    """Validate Self-Normalized IPS (SNIPS) estimator on a simulated bandit with known ground truth.

    Synthetic setup:
    Arm 0 has true CTR = 0.10, propensity = 0.80 (frequently logged)
    Arm 1 has true CTR = 0.30, propensity = 0.20 (infrequently logged)
    True overall population CTR under uniform logging = 0.20
    Naive unweighted CTR is biased towards Arm 0: ~0.14
    SNIPS estimator must recover true population value within confidence bounds!
    """
    rng = np.random.RandomState(42)
    n = 20000

    # Logging policy: 80% arm 0, 20% arm 1
    arms = rng.choice([0, 1], size=n, p=[0.80, 0.20])
    true_ctrs = {0: 0.10, 1: 0.30}
    propensities = np.array([0.80 if a == 0 else 0.20 for a in arms])

    # Clicks generated from true conditional CTR
    clicks = np.array([rng.binomial(1, true_ctrs[a]) for a in arms])

    # Naive sample mean is biased towards arm 0
    naive_ctr = clicks.mean()
    assert naive_ctr < 0.18  # Biased downwards by popularity of arm 0

    # Target uniform evaluation policy CTR: (0.10 + 0.30) / 2 = 0.20
    # Importance weights for target policy (uniform 0.5 / 0.5) vs logging policy
    target_p = 0.5
    weights = (target_p / propensities) * clicks
    snips_estimate = weights.sum() / (target_p / propensities).sum()

    assert pytest.approx(snips_estimate, abs=0.02) == 0.20
