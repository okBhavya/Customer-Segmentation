"""Unit tests for customer segmentation, RFM rules, and clustering."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from custintel.segment import _rfm_label, _rfm_segment


def test_rfm_zero_inflated_monetary():
    """Users with zero purchases/monetary spend must not crash with qcut/cut ties."""
    df = pd.DataFrame(
        {
            "days_since_last_event": [1.0, 5.0, 10.0, 2.0, 8.0, 0.5, 9.0],
            "total_events": [1, 2, 1, 15, 3, 1, 2],
            "total_spend": [0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 150.0],  # mostly zeros
        }
    )

    scored = _rfm_segment(df, n_bins=4)
    assert "rfm_segment" in scored.columns
    assert "r_score" in scored.columns
    assert "m_score" in scored.columns
    # Zero spend users must get m_score == 0
    zero_spend_mask = scored["total_spend"] == 0.0
    assert (scored.loc[zero_spend_mask, "m_score"] == 0.0).all()
    # Users with spend must get m_score > 0
    assert (scored.loc[~zero_spend_mask, "m_score"] > 0.0).all()


def test_rfm_label_mapping():
    """Verify segment label mapping logic."""
    r_champ = pd.Series({"r_score": 4, "f_score": 4, "m_score": 4})
    assert _rfm_label(r_champ) == "champions"

    r_loyal = pd.Series({"r_score": 3, "f_score": 2, "m_score": 2})
    assert _rfm_label(r_loyal) == "loyal"

    r_window = pd.Series({"r_score": 1, "f_score": 1, "m_score": 0})
    assert _rfm_label(r_window) == "window_shoppers"


def test_kmeans_determinism_with_fixed_seed():
    """K-Means with fixed random_state must produce identical cluster labels."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)

    km1 = KMeans(n_clusters=4, random_state=42, n_init=10)
    l1 = km1.fit_predict(X)

    km2 = KMeans(n_clusters=4, random_state=42, n_init=10)
    l2 = km2.fit_predict(X)

    np.testing.assert_array_equal(l1, l2)
    assert adjusted_rand_score(l1, l2) == 1.0


def test_segment_variance_explained():
    """Variance explained (R^2) should be between 0 and 1."""
    df = pd.DataFrame(
        {
            "segment": [0, 0, 1, 1, 2, 2],
            "outcome": [10.0, 12.0, 100.0, 105.0, 50.0, 48.0],
        }
    )
    group_means = df.groupby("segment")["outcome"].transform("mean")
    ss_between = ((group_means - df["outcome"].mean()) ** 2).sum()
    ss_total = ((df["outcome"] - df["outcome"].mean()) ** 2).sum()
    r2 = ss_between / ss_total
    assert 0.90 <= r2 <= 1.0
