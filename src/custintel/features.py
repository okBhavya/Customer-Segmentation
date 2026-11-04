"""Build user-level feature snapshots from cleaned events via DuckDB."""

import json
import time
from pathlib import Path

from custintel.db import get_connection, run_sql_file


def run_features(cfg: dict, root: Path):
    """Build train/validation/test feature sets from the events parquet."""
    parquet_path = root / cfg["paths"]["parquet"]
    features_dir = root / cfg["paths"]["features_dir"]
    sql_path = root / "sql" / "features.sql"
    report_path = root / cfg["paths"]["reports_dir"] / "feature_summary.json"

    features_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    min_churn = cfg.get("features", {}).get("min_events_for_churn_label", 2)
    summary: dict = {"snapshots": {}}
    t0 = time.time()

    for snap_name, dates in cfg["snapshots"].items():
        print(f"\n  Building features: {snap_name}")
        print(f"    Lookback: {dates['feature_start']} -> {dates['feature_end']}")
        print(f"    Horizon:  {dates['label_start']} -> {dates['label_end']}")

        con = get_connection(cfg, root)

        # Load parquet as the 'events' table the SQL expects
        con.execute(f"""
            CREATE VIEW events AS
            SELECT * FROM read_parquet('{parquet_path.as_posix()}')
        """)

        params = {
            "feature_start": dates["feature_start"],
            "feature_end": dates["feature_end"],
            "label_start": dates["label_start"],
            "label_end": dates["label_end"],
            "min_events_for_churn_label": str(min_churn),
        }
        run_sql_file(con, sql_path, params)

        # Read results
        out_path = features_dir / f"{snap_name}_features.parquet"
        row_count = con.execute("SELECT COUNT(*) FROM snapshot_features").fetchone()[0]

        # Feature column metadata
        col_info = con.execute("DESCRIBE snapshot_features").fetchall()
        feature_cols = [
            r[0]
            for r in col_info
            if r[0]
            not in (
                "user_id",
                "top_category",
                "top_brand",
                "label_purchase",
                "label_churn",
                "label_revenue",
            )
        ]

        # Class balance
        stats = con.execute("""
            SELECT
                AVG(label_purchase)  AS purchase_rate,
                AVG(label_churn)     AS churn_rate,
                AVG(label_revenue)   AS avg_revenue,
                SUM(label_purchase)  AS n_purchasers,
                SUM(label_churn)     AS n_churners
            FROM snapshot_features
        """).fetchone()

        snap_summary = {
            "rows": row_count,
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "purchase_rate": round(stats[0], 4) if stats[0] is not None else 0.0,
            "churn_rate": round(stats[1], 4) if stats[1] is not None else 0.0,
            "avg_revenue": round(stats[2], 4) if stats[2] is not None else 0.0,
            "n_purchasers": int(stats[3]) if stats[3] is not None else 0,
            "n_churners": int(stats[4]) if stats[4] is not None else 0,
        }
        summary["snapshots"][snap_name] = snap_summary

        print(f"    Users: {row_count:,}")
        print(f"    Features: {len(feature_cols)}")
        print(f"    Purchase rate: {snap_summary['purchase_rate']:.2%}")
        print(f"    Churn rate:    {snap_summary['churn_rate']:.2%}")

        # Save parquet
        con.execute(f"""
            COPY snapshot_features TO '{out_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print(f"    Saved: {out_path}")

        con.close()

    elapsed = time.time() - t0
    summary["total_time_seconds"] = round(elapsed, 1)

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Feature summary saved: {report_path}")
    print(f"  Feature engineering completed in {elapsed:.1f}s")
