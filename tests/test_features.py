"""Unit tests for SQL feature engineering, leakage prevention, and label construction."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from custintel.db import run_sql_file


@pytest.fixture
def tiny_fixture():
    """Tiny hand-computed events dataset."""
    return pd.DataFrame(
        [
            # User 1: 2 views, 1 cart, 1 purchase in lookback ($100 spend, 2 active days)
            {
                "event_time": "2019-10-01 10:00:00",
                "event_type": "view",
                "product_id": 101,
                "category_id": 1,
                "category_code": "electronics",
                "brand": "apple",
                "price": 100.0,
                "user_id": 1,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 10:05:00",
                "event_type": "cart",
                "product_id": 101,
                "category_id": 1,
                "category_code": "electronics",
                "brand": "apple",
                "price": 100.0,
                "user_id": 1,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-02 12:00:00",
                "event_type": "view",
                "product_id": 102,
                "category_id": 2,
                "category_code": "apparel",
                "brand": "nike",
                "price": 50.0,
                "user_id": 1,
                "user_session": "s2",
            },
            {
                "event_time": "2019-10-02 12:10:00",
                "event_type": "purchase",
                "product_id": 101,
                "category_id": 1,
                "category_code": "electronics",
                "brand": "apple",
                "price": 100.0,
                "user_id": 1,
                "user_session": "s2",
            },
            # User 1: Horizon activity (1 purchase)
            {
                "event_time": "2019-10-12 15:00:00",
                "event_type": "purchase",
                "product_id": 103,
                "category_id": 1,
                "category_code": "electronics",
                "brand": "apple",
                "price": 120.0,
                "user_id": 1,
                "user_session": "s3",
            },
            # User 2: 1 view only ($0 spend, 1 active day) -> Churn ineligible (active_days < 2)
            {
                "event_time": "2019-10-05 08:00:00",
                "event_type": "view",
                "product_id": 104,
                "category_id": 1,
                "category_code": "electronics",
                "brand": "samsung",
                "price": 200.0,
                "user_id": 2,
                "user_session": "s4",
            },
            # User 3: 2 active days in lookback, NO horizon events -> Churned!
            {
                "event_time": "2019-10-03 09:00:00",
                "event_type": "view",
                "product_id": 105,
                "category_id": 3,
                "category_code": "home",
                "brand": "ikea",
                "price": 40.0,
                "user_id": 3,
                "user_session": "s5",
            },
            {
                "event_time": "2019-10-04 09:00:00",
                "event_type": "view",
                "product_id": 106,
                "category_id": 3,
                "category_code": "home",
                "brand": "ikea",
                "price": 40.0,
                "user_id": 3,
                "user_session": "s6",
            },
        ]
    )


def test_sql_features_hand_computed(tiny_fixture):
    con = duckdb.connect()
    con.execute("CREATE TABLE events AS SELECT * FROM tiny_fixture")

    sql_path = Path(__file__).resolve().parents[1] / "sql" / "features.sql"
    params = {
        "feature_start": "2019-10-01",
        "feature_end": "2019-10-11",
        "label_start": "2019-10-11",
        "label_end": "2019-10-18",
        "min_events_for_churn_label": "2",
    }
    run_sql_file(con, sql_path, params)
    res = con.execute("SELECT * FROM snapshot_features ORDER BY user_id").fetchdf()

    assert len(res) == 3

    # Check User 1 Hand-Computed Values
    u1 = res[res["user_id"] == 1].iloc[0]
    assert u1["total_events"] == 4
    assert u1["view_count"] == 2
    assert u1["cart_count"] == 1
    assert u1["purchase_count"] == 1
    assert u1["total_spend"] == 100.0
    assert u1["active_days"] == 2
    assert u1["unique_categories"] == 2
    assert u1["label_purchase"] == 1
    assert u1["label_revenue"] == 120.0
    assert u1["label_churn"] == 0

    # Check User 2 (ineligible for churn)
    u2 = res[res["user_id"] == 2].iloc[0]
    assert u2["active_days"] == 1
    assert pd.isna(u2["label_churn"])  # Should be NaN/NULL because active_days < 2
    assert u2["label_purchase"] == 0

    # Check User 3 (churned)
    u3 = res[res["user_id"] == 3].iloc[0]
    assert u3["active_days"] == 2
    assert u3["label_churn"] == 1
    assert u3["label_purchase"] == 0


def test_no_leakage_post_snapshot_deletion(tiny_fixture):
    """Deleting post-snapshot events must leave feature columns strictly unchanged."""
    sql_path = Path(__file__).resolve().parents[1] / "sql" / "features.sql"
    params = {
        "feature_start": "2019-10-01",
        "feature_end": "2019-10-11",
        "label_start": "2019-10-11",
        "label_end": "2019-10-18",
        "min_events_for_churn_label": "2",
    }

    # Run 1: Full data
    con1 = duckdb.connect()
    con1.execute("CREATE TABLE events AS SELECT * FROM tiny_fixture")
    run_sql_file(con1, sql_path, params)
    f1 = con1.execute("SELECT * FROM snapshot_features ORDER BY user_id").fetchdf()

    # Run 2: Post-snapshot events deleted
    con2 = duckdb.connect()
    con2.execute(
        "CREATE TABLE events AS SELECT * FROM tiny_fixture WHERE event_time < '2019-10-11'"
    )
    run_sql_file(con2, sql_path, params)
    f2 = con2.execute("SELECT * FROM snapshot_features ORDER BY user_id").fetchdf()

    feature_cols = [c for c in f1.columns if not c.startswith("label_")]
    for col in feature_cols:
        np.testing.assert_array_equal(
            f1[col].values, f2[col].values, err_msg=f"Leakage detected in feature column {col}!"
        )


def test_edge_case_null_categories_and_brands():
    """Edge cases: null or unknown categories and brands must not crash feature queries."""
    df_edge = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 10:00:00",
                "event_type": "view",
                "product_id": 999,
                "category_id": 1,
                "category_code": "unknown",
                "brand": "unknown",
                "price": 10.0,
                "user_id": 99,
                "user_session": "s_edge",
            },
            {
                "event_time": "2019-10-02 10:00:00",
                "event_type": "view",
                "product_id": 998,
                "category_id": 1,
                "category_code": "unknown",
                "brand": "unknown",
                "price": 10.0,
                "user_id": 99,
                "user_session": "s_edge2",
            },
        ]
    )
    con = duckdb.connect()
    con.register("df_edge", df_edge)
    con.execute("CREATE TABLE events AS SELECT * FROM df_edge")
    sql_path = Path(__file__).resolve().parents[1] / "sql" / "features.sql"
    params = {
        "feature_start": "2019-10-01",
        "feature_end": "2019-10-11",
        "label_start": "2019-10-11",
        "label_end": "2019-10-18",
        "min_events_for_churn_label": "2",
    }
    run_sql_file(con, sql_path, params)
    res = con.execute("SELECT * FROM snapshot_features").fetchdf()
    assert len(res) == 1
    assert res.iloc[0]["unique_categories"] == 1
