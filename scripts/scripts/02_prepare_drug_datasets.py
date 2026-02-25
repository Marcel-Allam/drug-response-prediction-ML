#!/usr/bin/env python3
"""
02_prepare_drug_datasets.py

Purpose
-------
Build model-ready per-drug datasets:
- align cell line identifiers
- select drugs with sufficient sample size
- create regression targets and classification labels (extreme responders)

Inputs
------
- data/interim/harmonised/* (gene-harmonised expression matrices)
- raw response tables (GDSC + DepMap PRISM; configured)
- config/default_config.yaml

Outputs
-------
- data/processed/per_drug/ (per-drug X_train/X_test/y_train/y_test files)
- reports/tables/drug_sample_counts.csv

Usage
-----
python scripts/02_prepare_drug_datasets.py --config config/default_config.yaml

Notes
-----
- Implemented after Phase 1 harmonisation is complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


# ----------------------------- CONFIGURATION -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"


# ------------------------------- LOGGING ------------------------------------ #

def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


# ------------------------------- UTILITIES ---------------------------------- #

def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        log_error(f"Missing config file: {path}")
        raise SystemExit(1)

    try:
        import yaml  # type: ignore
    except Exception as e:
        log_error("PyYAML is required for config-driven scripts.")
        log_error(f"Install with: pip install pyyaml (or add to environment.yml). Details: {e}")
        raise SystemExit(1)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------- MAIN ------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-drug datasets for modelling (GDSC train, DepMap test).")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    _config = load_yaml_config(config_path)

    log_info("This is a scaffold script.")
    log_info("Next: implement after Phase 1 confirms harmonised expression outputs and response table schemas.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()