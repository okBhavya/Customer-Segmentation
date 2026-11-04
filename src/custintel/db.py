"""DuckDB helper: connect with resource limits and temp directory."""

from pathlib import Path

import duckdb


def get_connection(cfg: dict, root: Path) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB in-memory connection with configured limits."""
    con = duckdb.connect(":memory:")
    mem = cfg.get("duckdb", {}).get("memory_limit", "8GB")
    threads = cfg.get("duckdb", {}).get("threads", 4)
    temp_dir = root / cfg["paths"].get("duckdb_temp", "data/processed/duckdb_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET memory_limit = '{mem}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    return con


def run_sql_file(con: duckdb.DuckDBPyConnection, sql_path: Path, params: dict | None = None):
    """Read and execute a SQL file, substituting {{param}} placeholders."""
    sql = sql_path.read_text()
    if params:
        for key, val in params.items():
            sql = sql.replace(f"{{{{{key}}}}}", str(val))
    con.execute(sql)
