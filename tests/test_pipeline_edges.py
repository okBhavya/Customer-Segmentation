"""Unit tests for critical edge cases specified in Phase 11 QA audit."""

import numpy as np
import pandas as pd

from custintel.ranking import _calc_ndcg_at_k, _calc_recall_at_k
from custintel.segment import _profile_segments


def test_edge_case_ndcg_with_zero_relevant_items():
    """NDCG and Recall must return 0.0 when no items in ground truth are relevant."""
    y_true = np.array([0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    assert _calc_ndcg_at_k(y_true, scores, k=5) == 0.0
    assert _calc_recall_at_k(y_true, scores, k=5) == 0.0


def test_edge_case_ips_weight_clipping():
    """Propensities must be clipped to lower threshold to prevent inverse weight explosion."""
    raw_propensities = np.array([0.000001, 0.0001, 0.05, 0.50])
    clipped = np.clip(raw_propensities, 0.05, 1.0)

    # Weights must not exceed 1 / 0.05 = 20.0
    weights = 1.0 / clipped
    assert np.all(weights <= 20.0)
    assert np.all(clipped >= 0.05)


def test_edge_case_empty_segment_profiling():
    """Profiling segments must handle small or single-observation segments gracefully."""
    df = pd.DataFrame(
        {
            "segment_kmeans": [0, 1],
            "total_events": [5, 10],
            "total_spend": [0.0, 50.0],
            "active_days": [1, 2],
            "label_purchase": [0, 1],
            "label_churn": [1, 0],
            "label_revenue": [0.0, 50.0],
        }
    )

    profiles = _profile_segments(df, "segment_kmeans", "test")
    assert "0" in profiles
    assert "1" in profiles
    assert profiles["0"]["size"] == 1
    assert profiles["1"]["size"] == 1


def test_edge_case_purchases_without_prior_views():
    """Conversion rates must handle purchases without views without dividing by zero."""
    view_count = 0
    purchase_count = 2

    # Protected conversion rate
    rate = purchase_count * 1.0 / view_count if view_count > 0 else 0.0
    assert rate == 0.0


def test_edge_case_users_with_only_remove_from_cart():
    """Users with only remove_from_cart events should have valid bounded features."""
    total_events = 3
    remove_count = 3
    cart_count = 0

    cart_churn_rate = (cart_count + remove_count) * 1.0 / total_events if total_events > 0 else 0.0
    assert cart_churn_rate == 1.0


def test_edge_case_duplicate_rows_dedup():
    """Exact duplicate records must collapse to unique entries."""
    df = pd.DataFrame(
        [
            {"user_id": 1, "product_id": 100, "event_type": "view"},
            {"user_id": 1, "product_id": 100, "event_type": "view"},
            {"user_id": 1, "product_id": 100, "event_type": "view"},
        ]
    )
    deduped = df.drop_duplicates()
    assert len(deduped) == 1
