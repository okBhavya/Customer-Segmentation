"""Ingest raw CSV into cleaned, typed Parquet via DuckDB."""

import json
import time
from pathlib import Path

from custintel.db import get_connection


def run_ingest(cfg: dict, root: Path):
    """CSV -> deduplicated, typed Parquet with data quality report."""
    raw_path = root / cfg["paths"]["raw_csv"]
    out_path = root / cfg["paths"]["parquet"]
    report_path = root / cfg["paths"]["reports_dir"] / "data_profile.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    con = get_connection(cfg, root)

    print(f"Reading CSV: {raw_path}")
    t0 = time.time()

    csv_path = raw_path.as_posix()

    # Step 1: Single-pass profile of raw CSV
    print("  Profiling raw CSV...")
    profile_query = f"""
        SELECT
            COUNT(*) AS total_raw,
            COUNT(CASE WHEN event_time IS NULL THEN 1 END) AS null_event_time,
            COUNT(CASE WHEN event_type IS NULL THEN 1 END) AS null_event_type,
            COUNT(CASE WHEN product_id IS NULL THEN 1 END) AS null_product_id,
            COUNT(CASE WHEN category_id IS NULL THEN 1 END) AS null_category_id,
            COUNT(CASE WHEN category_code IS NULL OR TRIM(category_code) = '' THEN 1 END) AS null_category_code,
            COUNT(CASE WHEN brand IS NULL OR TRIM(brand) = '' THEN 1 END) AS null_brand,
            COUNT(CASE WHEN price IS NULL THEN 1 END) AS null_price,
            COUNT(CASE WHEN price <= 0 THEN 1 END) AS price_invalid,
            COUNT(CASE WHEN user_id IS NULL THEN 1 END) AS null_user_id,
            COUNT(CASE WHEN user_session IS NULL THEN 1 END) AS null_user_session
        FROM read_csv('{csv_path}', auto_detect=true)
    """
    raw_stats = con.execute(profile_query).fetchone()

    total_raw = raw_stats[0]
    invalid_price = raw_stats[8]

    null_rates_raw = {
        "event_time": round(raw_stats[1] / total_raw, 6) if total_raw > 0 else 0,
        "event_type": round(raw_stats[2] / total_raw, 6) if total_raw > 0 else 0,
        "product_id": round(raw_stats[3] / total_raw, 6) if total_raw > 0 else 0,
        "category_id": round(raw_stats[4] / total_raw, 6) if total_raw > 0 else 0,
        "category_code": round(raw_stats[5] / total_raw, 6) if total_raw > 0 else 0,
        "brand": round(raw_stats[6] / total_raw, 6) if total_raw > 0 else 0,
        "price": round(raw_stats[7] / total_raw, 6) if total_raw > 0 else 0,
        "user_id": round(raw_stats[9] / total_raw, 6) if total_raw > 0 else 0,
        "user_session": round(raw_stats[10] / total_raw, 6) if total_raw > 0 else 0,
    }

    # Raw event types
    event_types_raw = con.execute(f"""
        SELECT event_type, COUNT(*) as cnt
        FROM read_csv('{csv_path}', auto_detect=true)
        GROUP BY event_type ORDER BY cnt DESC
    """).fetchall()

    print(f"  Raw rows: {total_raw:,}")
    print(f"  Rows with price <= 0: {invalid_price:,}")

    # Step 2: Ingest cleaned and deduplicated events into Parquet
    # Deduplicates exact rows, drops price <= 0, normalises category_code and brand
    print("  Creating cleaned Parquet with deduplication...")
    con.execute(f"""
        CREATE OR REPLACE TABLE clean_events AS
        SELECT DISTINCT
            CAST(event_time AS TIMESTAMP) AS event_time,
            event_type,
            CAST(product_id AS BIGINT) AS product_id,
            CAST(category_id AS BIGINT) AS category_id,
            COALESCE(NULLIF(TRIM(category_code), ''), 'unknown') AS category_code,
            COALESCE(NULLIF(TRIM(brand), ''), 'unknown') AS brand,
            CAST(price AS DOUBLE) AS price,
            CAST(user_id AS BIGINT) AS user_id,
            user_session
        FROM read_csv('{csv_path}', auto_detect=true)
        WHERE price > 0
    """)

    final_count = con.execute("SELECT COUNT(*) FROM clean_events").fetchone()[0]

    # Calculate exact distinct count of raw to accurately report duplicates
    raw_distinct = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT * FROM read_csv('{csv_path}', auto_detect=true)
        )
    """).fetchone()[0]
    duplicates_removed = total_raw - raw_distinct
    print(f"  Duplicates removed: {duplicates_removed:,}")
    print(f"  Clean rows: {final_count:,}")

    # Step 3: Write Parquet with ZSTD
    print(f"  Writing Parquet to {out_path}...")
    con.execute(f"COPY clean_events TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # Step 4: Final profile metrics
    unique_users = con.execute("SELECT COUNT(DISTINCT user_id) FROM clean_events").fetchone()[0]
    unique_products = con.execute(
        "SELECT COUNT(DISTINCT product_id) FROM clean_events"
    ).fetchone()[0]
    date_range = con.execute(
        "SELECT MIN(event_time)::VARCHAR, MAX(event_time)::VARCHAR FROM clean_events"
    ).fetchone()

    event_types_clean = con.execute("""
        SELECT event_type, COUNT(*) as cnt
        FROM clean_events GROUP BY event_type ORDER BY cnt DESC
    """).fetchall()

    null_rates_clean = {
        "event_time": 0.0,
        "event_type": 0.0,
        "product_id": 0.0,
        "category_id": 0.0,
        "category_code": 0.0,  # normalised to 'unknown'
        "brand": 0.0,  # normalised to 'unknown'
        "price": 0.0,
        "user_id": 0.0,
        "user_session": round(
            con.execute("SELECT COUNT(*) FROM clean_events WHERE user_session IS NULL").fetchone()[
                0
            ]
            / final_count,
            6,
        ),
    }

    elapsed = time.time() - t0

    profile = {
        "total_raw_events": total_raw,
        "total_clean_events": final_count,
        "duplicates_removed": duplicates_removed,
        "price_invalid_removed": invalid_price,
        "unique_users": unique_users,
        "unique_products": unique_products,
        "date_range": {"min": date_range[0], "max": date_range[1]},
        "event_type_counts_raw": {r[0]: r[1] for r in event_types_raw},
        "event_type_counts": {r[0]: r[1] for r in event_types_clean},
        "null_rates_raw": null_rates_raw,
        "null_rates_clean": null_rates_clean,
        "ingest_time_seconds": round(elapsed, 1),
    }

    with open(report_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"  Profile saved: {report_path}")
    print(f"  Ingest completed in {elapsed:.1f}s")
    con.close()
