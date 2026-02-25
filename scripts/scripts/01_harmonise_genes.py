#!/usr/bin/env python3
"""
01_harmonise_genes.py

Purpose
-------
Harmonise gene identifiers between GDSC and DepMap expression matrices by mapping
to HGNC symbols and taking the intersection of shared genes.

Inputs
------
- data/raw/gdsc/   (GDSC expression file; release-dependent filename)
- data/raw/depmap/ (DepMap/CCLE expression file; release-dependent filename)
- config/default_config.yaml (paths + settings)

Outputs
-------
- data/interim/harmonised/gdsc_expression_hgnc.parquet
- data/interim/harmonised/depmap_expression_hgnc.parquet
- data/interim/harmonised/shared_genes.txt

Usage
-----
python scripts/01_harmonise_genes.py --config config/default_config.yaml

Notes
-----
- This script is implemented after Phase 0 confirms exact file formats/columns.
- Do not guess filenames. Run scripts/00_inspect_datasets.py first.
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
    parser = argparse.ArgumentParser(description="HGNC harmonisation for GDSC and DepMap expression matrices.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml_config(config_path)

    log_info("This is a scaffold script.")
    log_info("Next: we will implement this after we see Phase 0 inspection output,")
    log_info("so we correctly load your specific GDSC/DepMap expression formats (CSV/TSV/Parquet, row/col orientation).")

    raise SystemExit(0)


if __name__ == "__main__":
    main()