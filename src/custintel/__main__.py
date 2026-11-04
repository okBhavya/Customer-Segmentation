"""CLI entry point: python -m custintel <stage>."""

import argparse
import json
import sys
import time
from pathlib import Path

from custintel.config import get_project_root, load_config

STAGES = [
    "ingest",
    "features",
    "funnel",
    "segment",
    "propensity",
    "rank",
    "nba",
    "experiment",
    "app-data",
    "app",
]


def main():
    parser = argparse.ArgumentParser(
        prog="custintel",
        description="Customer Segmentation, NBA and Banner Ranking pipeline.",
    )
    parser.add_argument(
        "stage",
        choices=STAGES + ["all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on synthetic test data instead of real data.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = get_project_root()

    if args.smoke:
        cfg["_smoke"] = True
        cfg["paths"]["raw_csv"] = str(root / "tests" / "fixtures" / "smoke_events.csv")

    runtimes = {}

    stages_to_run = [s for s in STAGES if s != "app"] if args.stage == "all" else [args.stage]
    for stage in stages_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Stage: {stage}")
        print(f"{'=' * 60}\n")
        t0 = time.time()
        _run_stage(stage, cfg, root)
        elapsed = time.time() - t0
        runtimes[stage] = round(elapsed, 1)
        print(f"\n  [{stage}] completed in {elapsed:.1f}s\n")

    # Save runtimes
    rt_path = root / "reports" / "runtimes.json"
    rt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rt_path, "w") as f:
        json.dump(runtimes, f, indent=2)
    print(f"\nRuntimes saved to {rt_path}")


def _run_stage(stage: str, cfg: dict, root: Path):
    if stage == "ingest":
        from custintel.ingest import run_ingest

        run_ingest(cfg, root)
    elif stage == "features":
        from custintel.features import run_features

        run_features(cfg, root)
    elif stage == "funnel":
        from custintel.funnel import run_funnel

        run_funnel(cfg, root)
    elif stage == "segment":
        from custintel.segment import run_segment

        run_segment(cfg, root)
    elif stage == "propensity":
        from custintel.propensity import run_propensity

        run_propensity(cfg, root)
    elif stage == "rank":
        from custintel.ranking import run_ranking

        run_ranking(cfg, root)
    elif stage == "nba":
        from custintel.nba import run_nba

        run_nba(cfg, root)
    elif stage == "experiment":
        from custintel.experiment import run_experiment

        run_experiment(cfg, root)
    elif stage == "app-data":
        from custintel.app_data import run_app_data

        run_app_data(cfg, root)
    elif stage == "app":
        import subprocess

        app_path = root / "app" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
