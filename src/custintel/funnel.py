"""Funnel analysis and onsite opportunity identification."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from custintel.db import get_connection


def run_funnel(cfg: dict, root: Path):
    """Run funnel analysis on the full dataset."""
    parquet_path = root / cfg["paths"]["parquet"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    con = get_connection(cfg, root)
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')")

    report = {}

    # Overall funnel
    funnel = con.execute("""
        SELECT
            COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) as viewers,
            COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carters,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchasers,
            COUNT(CASE WHEN event_type = 'view' THEN 1 END) as view_events,
            COUNT(CASE WHEN event_type = 'cart' THEN 1 END) as cart_events,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchase_events
        FROM events
    """).fetchone()

    report["overall_funnel"] = {
        "viewers": funnel[0],
        "carters": funnel[1],
        "purchasers": funnel[2],
        "view_events": funnel[3],
        "cart_events": funnel[4],
        "purchase_events": funnel[5],
        "view_to_cart_rate": round(funnel[1] / funnel[0], 4) if funnel[0] else 0,
        "cart_to_purchase_rate": round(funnel[2] / funnel[1], 4) if funnel[1] else 0,
        "overall_conversion": round(funnel[2] / funnel[0], 4) if funnel[0] else 0,
    }
    print(
        f"  Overall funnel: {funnel[0]:,} viewers -> {funnel[1]:,} carters -> {funnel[2]:,} purchasers"
    )

    # Funnel by hour
    hourly = con.execute("""
        SELECT
            EXTRACT(HOUR FROM event_time) as hour,
            COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) as viewers,
            COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carters,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchasers
        FROM events
        GROUP BY hour ORDER BY hour
    """).fetchdf()
    hourly["view_to_cart"] = (hourly["carters"] / hourly["viewers"]).round(4)
    hourly["cart_to_purchase"] = (hourly["purchasers"] / hourly["carters"]).round(4)
    report["hourly_funnel"] = hourly.to_dict(orient="records")

    # Funnel by weekday
    weekday = con.execute("""
        SELECT
            EXTRACT(DOW FROM event_time) as weekday,
            COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) as viewers,
            COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carters,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchasers
        FROM events
        GROUP BY weekday ORDER BY weekday
    """).fetchdf()
    weekday["view_to_cart"] = (weekday["carters"] / weekday["viewers"]).round(4)
    report["weekday_funnel"] = weekday.to_dict(orient="records")

    # Top category funnels
    cat_funnel = con.execute("""
        SELECT
            category_code,
            COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) as viewers,
            COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carters,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchasers,
            COUNT(CASE WHEN event_type = 'view' THEN 1 END) as views
        FROM events
        WHERE category_code != 'unknown'
        GROUP BY category_code
        HAVING views >= 1000
        ORDER BY views DESC
        LIMIT 30
    """).fetchdf()
    cat_funnel["view_to_cart"] = (cat_funnel["carters"] / cat_funnel["viewers"]).round(4)
    cat_funnel["cart_to_purchase"] = (
        cat_funnel["purchasers"] / cat_funnel["carters"].replace(0, np.nan)
    ).round(4)
    report["category_funnel"] = cat_funnel.to_dict(orient="records")

    # Cart abandonment
    cart_abandon = con.execute("""
        WITH cart_users AS (
            SELECT DISTINCT user_id FROM events WHERE event_type = 'cart'
        ),
        purchase_users AS (
            SELECT DISTINCT user_id FROM events WHERE event_type = 'purchase'
        )
        SELECT
            COUNT(*) as total_carters,
            COUNT(*) - COUNT(p.user_id) as abandoners
        FROM cart_users c
        LEFT JOIN purchase_users p ON c.user_id = p.user_id
    """).fetchone()
    report["cart_abandonment"] = {
        "total_carters": cart_abandon[0],
        "abandoners": cart_abandon[1],
        "abandonment_rate": round(cart_abandon[1] / cart_abandon[0], 4) if cart_abandon[0] else 0,
    }

    # Single vs multi-session buyers
    buyer_sessions = con.execute("""
        SELECT
            user_id,
            COUNT(DISTINCT user_session) as n_sessions
        FROM events
        WHERE event_type = 'purchase'
        GROUP BY user_id
    """).fetchdf()
    single = (buyer_sessions["n_sessions"] == 1).sum()
    multi = (buyer_sessions["n_sessions"] > 1).sum()
    report["buyer_sessions"] = {
        "single_session_buyers": int(single),
        "multi_session_buyers": int(multi),
        "multi_session_rate": round(multi / (single + multi), 4) if (single + multi) else 0,
    }

    # Time to purchase (from first view in session to purchase)
    time_to_purchase = con.execute("""
        WITH purchase_sessions AS (
            SELECT DISTINCT user_session FROM events WHERE event_type = 'purchase'
        ),
        session_starts AS (
            SELECT e.user_session, MIN(e.event_time) as session_start
            FROM events e
            JOIN purchase_sessions ps ON e.user_session = ps.user_session
            WHERE e.event_type = 'view'
            GROUP BY e.user_session
        ),
        purchases AS (
            SELECT user_session, MIN(event_time) as purchase_time
            FROM events
            WHERE event_type = 'purchase'
            GROUP BY user_session
        )
        SELECT
            EXTRACT(EPOCH FROM (p.purchase_time - s.session_start)) / 60.0 as minutes_to_purchase
        FROM purchases p
        JOIN session_starts s ON p.user_session = s.user_session
        WHERE p.purchase_time >= s.session_start
    """).fetchdf()

    if len(time_to_purchase) > 0:
        report["time_to_purchase"] = {
            "median_minutes": round(float(time_to_purchase["minutes_to_purchase"].median()), 1),
            "mean_minutes": round(float(time_to_purchase["minutes_to_purchase"].mean()), 1),
            "p25_minutes": round(float(time_to_purchase["minutes_to_purchase"].quantile(0.25)), 1),
            "p75_minutes": round(float(time_to_purchase["minutes_to_purchase"].quantile(0.75)), 1),
        }

    # Opportunity sizing
    if len(cat_funnel) > 0:
        avg_conversion = cat_funnel["cart_to_purchase"].mean()
        cat_funnel["conversion_gap"] = avg_conversion - cat_funnel["cart_to_purchase"].fillna(0)
        # Conservative 10% uplift assumption
        cat_funnel["potential_incremental_orders"] = (
            cat_funnel["carters"] * cat_funnel["conversion_gap"].clip(lower=0) * 0.10
        ).round(0)
        opportunities = cat_funnel[cat_funnel["conversion_gap"] > 0].sort_values(
            "potential_incremental_orders", ascending=False
        )
        report["opportunities"] = (
            opportunities[
                [
                    "category_code",
                    "views",
                    "viewers",
                    "carters",
                    "cart_to_purchase",
                    "conversion_gap",
                    "potential_incremental_orders",
                ]
            ]
            .head(15)
            .to_dict(orient="records")
        )
        report["opportunity_assumptions"] = (
            "Incremental orders estimated assuming 10% of the conversion gap "
            "can be closed through targeted interventions. This is an offline estimate, "
            "not a measured lift."
        )

    # Save report
    with open(reports_dir / "funnel_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Figures
    _plot_funnel(report, figures_dir)
    _plot_hourly_conversion(hourly, figures_dir)
    _plot_opportunities(report, figures_dir)

    con.close()
    print("  Funnel analysis complete.")


def _plot_funnel(report: dict, figures_dir: Path):
    """Overall funnel bar chart."""
    funnel = report["overall_funnel"]
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["Viewers", "Carters", "Purchasers"]
    values = [funnel["viewers"], funnel["carters"], funnel["purchasers"]]
    colors = ["#4a90d9", "#e8744f", "#5cb85c"]
    bars = ax.bar(stages, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Unique Users")
    ax.set_title("Overall Conversion Funnel")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / "funnel_overall.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_hourly_conversion(hourly: pd.DataFrame, figures_dir: Path):
    """Hourly conversion rates."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["view_to_cart"], "o-", label="View→Cart", color="#4a90d9")
    ax.plot(
        hourly["hour"], hourly["cart_to_purchase"], "s-", label="Cart→Purchase", color="#e8744f"
    )
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rates by Hour")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / "hourly_conversion.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_opportunities(report: dict, figures_dir: Path):
    """Opportunity sizing chart."""
    opps = report.get("opportunities", [])
    if not opps:
        return
    df = pd.DataFrame(opps).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Truncate long category names
    labels = [c[:30] for c in df["category_code"]]
    ax.barh(labels, df["potential_incremental_orders"], color="#4a90d9", edgecolor="white")
    ax.set_xlabel("Potential Incremental Orders (10% uplift assumption)")
    ax.set_title("Top Opportunities by Category")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / "opportunities.png", dpi=150, bbox_inches="tight")
    plt.close()
