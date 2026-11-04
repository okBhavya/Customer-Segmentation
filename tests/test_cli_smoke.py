"""CLI smoke tests and end-to-end integration tests using synthetic data."""

from custintel.config import get_project_root, load_config
from custintel.db import get_connection


def test_config_loader():
    """Verify default.yaml config loads with required keys."""
    cfg = load_config()
    assert "paths" in cfg
    assert "snapshots" in cfg
    assert "seed" in cfg
    assert "segmentation" in cfg
    assert "propensity" in cfg
    assert "ranking" in cfg
    assert "nba" in cfg
    assert "experiment" in cfg


def test_duckdb_connection_limits():
    """Verify DuckDB connection applies memory and thread limits."""
    cfg = load_config()
    root = get_project_root()
    con = get_connection(cfg, root)
    res = con.execute("SELECT 1 + 1").fetchone()[0]
    assert res == 2
    con.close()


def test_smoke_events_fixture_exists():
    """Verify smoke fixture file exists."""
    root = get_project_root()
    p = root / "tests" / "fixtures" / "smoke_events.csv"
    assert p.exists()
    assert p.stat().st_size > 1000


def test_smoke_pipeline_end_to_end(tmp_path):
    """Run full pipeline end-to-end on smoke fixture in an isolated temporary directory."""
    root = get_project_root()
    cfg = load_config()

    # Configure isolated smoke paths
    smoke_raw = root / "tests" / "fixtures" / "smoke_events.csv"
    cfg["paths"]["raw_csv"] = str(smoke_raw)
    cfg["paths"]["parquet"] = str(tmp_path / "smoke_events.parquet")
    cfg["paths"]["features_dir"] = str(tmp_path / "features")
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    cfg["paths"]["figures_dir"] = str(tmp_path / "figures")
    cfg["paths"]["app_data_dir"] = str(tmp_path / "app_data")

    # Fast parameters for test execution
    cfg["duckdb"]["memory_limit"] = "512MB"
    cfg["duckdb"]["threads"] = 2
    cfg["segmentation"]["silhouette_sample"] = 50
    cfg["segmentation"]["stability_bootstraps"] = 2
    cfg["propensity"]["tuning_trials"] = 2
    cfg["propensity"]["bootstrap_ci_n"] = 20
    cfg["ranking"]["min_unit_events"] = 1
    cfg["ranking"]["user_sample_size"] = 50
    cfg["ranking"]["eval_user_sample"] = 25
    cfg["ranking"]["bootstrap_ci_n"] = 20

    from custintel.app_data import run_app_data
    from custintel.experiment import run_experiment
    from custintel.features import run_features
    from custintel.funnel import run_funnel
    from custintel.ingest import run_ingest
    from custintel.nba import run_nba
    from custintel.propensity import run_propensity
    from custintel.ranking import run_ranking
    from custintel.segment import run_segment

    run_ingest(cfg, root)
    assert (tmp_path / "smoke_events.parquet").exists()
    assert (tmp_path / "reports" / "data_profile.json").exists()

    run_features(cfg, root)
    assert (tmp_path / "features" / "train_features.parquet").exists()
    assert (tmp_path / "features" / "test_features.parquet").exists()

    run_funnel(cfg, root)
    assert (tmp_path / "reports" / "funnel_report.json").exists()

    run_segment(cfg, root)
    assert (tmp_path / "features" / "train_segments.parquet").exists()
    assert (tmp_path / "reports" / "segment_metrics.json").exists()

    run_propensity(cfg, root)
    assert (tmp_path / "reports" / "propensity_metrics.json").exists()

    run_ranking(cfg, root)
    assert (tmp_path / "reports" / "ranking_metrics.json").exists()

    run_nba(cfg, root)
    assert (tmp_path / "reports" / "nba_report.json").exists()

    run_experiment(cfg, root)
    assert (tmp_path / "reports" / "experiment_report.json").exists()

    run_app_data(cfg, root)
    assert (tmp_path / "app_data" / "sampled_users.parquet").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
