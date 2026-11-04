"""Load and validate configuration."""

from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load YAML config from the given path or the default location."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]
