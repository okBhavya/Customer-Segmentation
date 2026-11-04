"""Export compact, standalone data files for Streamlit demo app (<= 15 MB)."""

import json
import shutil
from pathlib import Path

import pandas as pd


def run_app_data(cfg: dict, root: Path):
    """Sample test users, copy reports and metrics into app_data/."""
    features_dir = root / cfg["paths"]["features_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    app_data_dir = root / "app_data"
    app_data_dir.mkdir(parents=True, exist_ok=True)

    print("  Preparing compact app_data/...")

    # 1. Sample ~20k test users with segments, propensities, and NBA allocations
    nba_path = features_dir / "test_nba_allocated.parquet"
    if nba_path.exists():
        test_df = pd.read_parquet(nba_path)
    else:
        test_df = pd.read_parquet(features_dir / "test_features.parquet")

    sample_n = min(20000, len(test_df))
    sample_users = test_df.sample(n=sample_n, random_state=cfg.get("seed", 42))

    compact_cols = [
        "user_id",
        "active_days",
        "total_events",
        "total_spend",
        "avg_order_value",
        "days_since_last_event",
        "has_open_cart",
        "unique_categories",
        "segment_kmeans",
        "rfm_segment",
        "proba_purchase",
        "proba_churn",
        "nba_action",
        "nba_reason",
        "nba_ev",
        "label_purchase",
        "label_revenue",
    ]
    avail_cols = [c for c in compact_cols if c in sample_users.columns]
    sample_users[avail_cols].to_parquet(app_data_dir / "sampled_users.parquet", index=False)
    print(f"    Exported {len(sample_users):,} sampled users to app_data/sampled_users.parquet")

    # 2. Copy JSON summary reports into app_data/
    for rep in [
        "data_profile.json",
        "funnel_report.json",
        "segment_metrics.json",
        "segment_profiles.json",
        "propensity_metrics.json",
        "nba_report.json",
        "ranking_metrics.json",
        "experiment_report.json",
    ]:
        src = reports_dir / rep
        if src.exists():
            shutil.copy2(src, app_data_dir / rep)
            print(f"    Copied {rep} to app_data/")

    # 3. Create master metrics.json summarizing all key findings
    metrics_summary = {}
    for rep in [
        "data_profile.json",
        "segment_metrics.json",
        "propensity_metrics.json",
        "nba_report.json",
        "ranking_metrics.json",
        "experiment_report.json",
    ]:
        src = reports_dir / rep
        if src.exists():
            with open(src) as f:
                metrics_summary[rep.replace(".json", "")] = json.load(f)

    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    shutil.copy2(reports_dir / "metrics.json", app_data_dir / "metrics.json")
    print("    Consolidated master reports/metrics.json created and copied to app_data/")

    # 4. Copy key figures to app_data/figures/
    app_fig_dir = app_data_dir / "figures"
    app_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_file in figures_dir.glob("*.png"):
        shutil.copy2(fig_file, app_fig_dir / fig_file.name)
    print("    Copied report figures to app_data/figures/")

    # Verify total size of app_data/ <= 15 MB
    total_bytes = sum(f.stat().st_size for f in app_data_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    print(f"\n  Total app_data/ footprint: {total_mb:.2f} MB (Budget <= 15 MB)")
