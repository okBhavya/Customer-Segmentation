"""Format markdown metrics summary tables from reports/metrics.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"


def main():
    if not METRICS_PATH.exists():
        print("metrics.json not found.")
        return

    with open(METRICS_PATH) as f:
        m = json.load(f)

    # 1. Dataset Profile Table
    dp = m.get("data_profile", {})
    print("### Data Ingestion & Quality Profile")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Raw Events Processed | {dp.get('total_raw_events', 0):,} |")
    print(f"| Clean Events | {dp.get('total_clean_events', 0):,} |")
    print(f"| Duplicate Rows Removed | {dp.get('duplicates_removed', 0):,} |")
    print(f"| Invalid Prices Removed | {dp.get('price_invalid_removed', 0):,} |")
    print(f"| Unique Users | {dp.get('unique_users', 0):,} |")
    print(f"| Unique Products | {dp.get('unique_products', 0):,} |")
    print(f"| Ingestion Runtime | {dp.get('ingest_time_seconds', 0.0):.1f}s |")
    print()

    # 2. Propensity Scorecard Table
    prop = m.get("propensity_metrics", {})
    purch = prop.get("purchase", {})
    churn = prop.get("churn", {})

    print("### Propensity Models Performance (Out-of-Time Test Snapshot)")
    print("| Target | Model | ROC-AUC | PR-AUC | Log-Loss | Brier Score |")
    print("|---|---|---|---|---|---|")
    for target_name, target_dict in [("Purchase", purch), ("Churn", churn)]:
        for model_key in [
            "rfm_heuristic",
            "logistic_regression",
            "lightgbm_raw",
            "lightgbm_calibrated",
        ]:
            if model_key in target_dict:
                row = target_dict[model_key]
                print(
                    f"| {target_name} | `{model_key}` | "
                    f"{row.get('roc_auc', 0.0):.4f} | "
                    f"{row.get('pr_auc', 0.0):.4f} | "
                    f"{row.get('log_loss', 0.0):.4f} | "
                    f"{row.get('brier', 0.0):.4f} |"
                )
    print()

    # 3. Banner Ranking Table
    rk = m.get("ranking_metrics", {})
    print("### Category Banner Ranking (Held-Out Test Sample)")
    print(
        "| Model Policy | NDCG@10 | Recall@10 | HitRate@10 | Catalog Coverage | Cold-Start NDCG |"
    )
    print("|---|---|---|---|---|---|")
    for model_key in [
        "global_popularity",
        "segment_popularity",
        "user_most_viewed",
        "lightgbm_ltr",
    ]:
        if model_key in rk:
            r = rk[model_key]
            print(
                f"| `{model_key}` | "
                f"{r.get('ndcg_at_10', 0.0):.4f} | "
                f"{r.get('recall_at_10', 0.0):.4f} | "
                f"{r.get('hit_rate_at_10', 0.0):.4f} | "
                f"{r.get('catalog_coverage', 0.0):.1%} | "
                f"{r.get('cold_start_ndcg', 0.0):.4f} |"
            )
    print()

    # 4. Next-Best-Action Policy Simulation Table
    nba = m.get("nba_report", {})
    sims = nba.get("simulations", {})
    print("### Next-Best-Action Policy Simulation (Offline Expected Value)")
    print(
        "| Policy | Total Simulated Cost | Total Simulated EV | Avg EV / User | Interventions Sent |"
    )
    print("|---|---|---|---|---|")
    for pol_key, pol in sims.items():
        print(
            f"| `{pol_key}` | "
            f"${pol.get('total_cost', 0.0):,.2f} | "
            f"${pol.get('total_simulated_ev', 0.0):,.2f} | "
            f"${pol.get('avg_ev_per_user', 0.0):.4f} | "
            f"{pol.get('actions_taken', 0):,} |"
        )
    print()

    # 5. Experimentation & Power Table
    exp = m.get("experiment_report", {})
    print("### Experiment Design & MDE Table (Baseline Conversion = 3.47%)")
    print("| Relative MDE | Absolute MDE | Required Sample / Arm | Total Sample (2 Arms) |")
    print("|---|---|---|---|")
    for s in exp.get("sample_size_table", []):
        print(
            f"| {s.get('relative_mde', 0.0):.1%} | "
            f"{s.get('absolute_mde', 0.0):.5f} | "
            f"{s.get('sample_size_per_arm', 0):,} | "
            f"{s.get('total_sample_size', 0):,} |"
        )


if __name__ == "__main__":
    main()
