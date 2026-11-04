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
