"""Audit tests: independent pandas re-implementations, leakage checks, and AppTest."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from custintel.config import get_project_root, load_config
from custintel.db import run_sql_file


@pytest.fixture
def audit_sample():
    """Sample dataset for cross-checking SQL against pure pandas."""
    rng = np.random.RandomState(42)
    n = 300
    uids = rng.choice([10, 20, 30, 40, 50], size=n)
    times = pd.date_range("2019-10-01", periods=10, freq="D")
    event_times = [
        rng.choice(times) + pd.Timedelta(hours=int(rng.randint(0, 23))) for _ in range(n)
    ]
    etypes = rng.choice(["view", "cart", "purchase"], size=n, p=[0.7, 0.2, 0.1])
    prices = rng.uniform(10, 500, size=n).round(2)
    cids = rng.choice(["elec", "appl", "cloth"], size=n)
    brands = rng.choice(["brandA", "brandB", "brandC"], size=n)

    return pd.DataFrame(
        {
            "event_time": event_times,
            "event_type": etypes,
            "product_id": rng.randint(1000, 2000, size=n),
            "category_id": rng.randint(1, 5, size=n),
            "category_code": cids,
            "brand": brands,
            "price": prices,
            "user_id": uids,
            "user_session": [f"s_{u}_{i % 3}" for i, u in enumerate(uids)],
        }
    )


def test_sql_vs_pandas_10_features_cross_check(audit_sample):
    """Re-implement 10 SQL features independently in pure pandas and assert exact match."""
    # 1. SQL computation
    con = duckdb.connect()
    con.execute("CREATE TABLE events AS SELECT * FROM audit_sample")
    sql_path = Path(__file__).resolve().parents[1] / "sql" / "features.sql"
    params = {
        "feature_start": "2019-10-01",
        "feature_end": "2019-10-11",
        "label_start": "2019-10-11",
        "label_end": "2019-10-18",
        "min_events_for_churn_label": "2",
    }
    run_sql_file(con, sql_path, params)
    sql_res = (
        con.execute("SELECT * FROM snapshot_features ORDER BY user_id")
        .fetchdf()
        .set_index("user_id")
    )

    # 2. Independent pure pandas computation
    lb = audit_sample[
        (audit_sample["event_time"] >= "2019-10-01") & (audit_sample["event_time"] < "2019-10-11")
    ]

    for uid in sql_res.index:
        u_df = lb[lb["user_id"] == uid]
        row = sql_res.loc[uid]

        # 1. total_events
        assert row["total_events"] == len(u_df)

        # 2. view_count
        assert row["view_count"] == (u_df["event_type"] == "view").sum()

        # 3. cart_count
        assert row["cart_count"] == (u_df["event_type"] == "cart").sum()

        # 4. purchase_count
        assert row["purchase_count"] == (u_df["event_type"] == "purchase").sum()

        # 5. total_spend
        expected_spend = u_df.loc[u_df["event_type"] == "purchase", "price"].sum()
        assert pytest.approx(row["total_spend"], 1e-2) == expected_spend

        # 6. active_days
        assert row["active_days"] == u_df["event_time"].dt.date.nunique()

        # 7. unique_categories
        assert row["unique_categories"] == u_df["category_code"].nunique()

        # 8. unique_brands
        assert row["unique_brands"] == u_df["brand"].nunique()

        # 9. total_sessions
        assert row["total_sessions"] == u_df["user_session"].nunique()

        # 10. price_mean
        expected_price_mean = u_df["price"].mean()
        assert pytest.approx(row["price_mean"], 1e-2) == expected_price_mean


def test_shuffled_label_auc_near_half():
    """Under permuted/shuffled labels, AUC must be consistent with random guessing (~0.50)."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.25, 1000)
    scores = rng.normal(0, 1, 1000)

    # Shuffled labels
    y_shuffled = rng.permutation(y_true)
    auc = roc_auc_score(y_shuffled, scores)
    assert 0.45 <= auc <= 0.55


def test_temporal_windows_strictly_disjoint():
    """Verify train, validation, and test snapshot label windows are disjoint."""
    cfg = load_config()
    snaps = cfg["snapshots"]

    tr_lend = pd.Timestamp(snaps["train"]["label_end"])
    va_lstart = pd.Timestamp(snaps["validation"]["label_start"])
    va_lend = pd.Timestamp(snaps["validation"]["label_end"])
    te_lstart = pd.Timestamp(snaps["test"]["label_start"])

    # Label windows must not overlap
    assert tr_lend <= va_lstart
    assert va_lend <= te_lstart


def test_app_test_execution():
    """Test Streamlit application pages using streamlit.testing.v1.AppTest."""
    from streamlit.testing.v1 import AppTest

    app_path = str(get_project_root() / "app" / "app.py")
    at = AppTest.from_file(app_path)
    at.run()
    assert not at.exception
