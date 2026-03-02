#!/usr/bin/env python3
"""
00_validate_config.py

Purpose
-------
Validate that config/default_config.yaml exists, loads correctly, and that
required directory paths resolve relative to REPO_ROOT.

This script performs strict configuration validation only.
No data inspection and no downloading logic are included.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml


# ----------------------------- PATH SETUP ---------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"


# ----------------------------- LOGGING ------------------------------------- #

LOGGER = logging.getLogger("config_validator")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_info(message: str) -> None:
    LOGGER.info("[INFO] %s", message)


def log_warn(message: str) -> None:
    LOGGER.warning("[WARN] %s", message)


def log_error(message: str) -> None:
    LOGGER.error("[ERROR] %s", message)


# ----------------------------- VALIDATION ---------------------------------- #

def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        log_error(f"Expected mapping at '{context}', got {type(value).__name__}.")
        raise SystemExit(1)
    return value


def require_key(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        log_error(f"Missing required key '{key}' in '{context}'.")
        raise SystemExit(1)
    return mapping[key]


def require_relative_dir_path(value: Any, key_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Path value for '{key_name}' must be a non-empty string.")
        raise SystemExit(1)

    raw_path = Path(value)
    if raw_path.is_absolute():
        log_error(f"Path for '{key_name}' must be relative to REPO_ROOT, got absolute path: {raw_path}")
        raise SystemExit(1)

    return (REPO_ROOT / raw_path).resolve()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists() or not config_path.is_file():
        log_error(f"Config file does not exist: {config_path}")
        raise SystemExit(1)

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception as exc:
        log_error(f"Failed to parse YAML config '{config_path}': {exc}")
        raise SystemExit(1) from exc

    config = require_mapping(parsed, "root")

    for top_key in ("project", "paths", "files", "schema"):
        require_key(config, top_key, "root")

    return config


def resolve_required_directories(config: dict[str, Any]) -> dict[str, Path]:
    paths = require_mapping(require_key(config, "paths", "root"), "paths")

    raw = require_mapping(require_key(paths, "raw", "paths"), "paths.raw")
    interim = require_mapping(require_key(paths, "interim", "paths"), "paths.interim")
    processed = require_mapping(require_key(paths, "processed", "paths"), "paths.processed")
    reports = require_mapping(require_key(paths, "reports", "paths"), "paths.reports")
    results = require_mapping(require_key(paths, "results", "paths"), "paths.results")

    resolved: dict[str, Path] = {
        "raw_gdsc_dir": require_relative_dir_path(
            require_key(raw, "gdsc_dir", "paths.raw"), "paths.raw.gdsc_dir"
        ),
        "raw_depmap_dir": require_relative_dir_path(
            require_key(raw, "depmap_dir", "paths.raw"), "paths.raw.depmap_dir"
        ),
        "harmonised_dir": require_relative_dir_path(
            require_key(interim, "harmonised_dir", "paths.interim"), "paths.interim.harmonised_dir"
        ),
        "per_drug_dir": require_relative_dir_path(
            require_key(processed, "per_drug_dir", "paths.processed"), "paths.processed.per_drug_dir"
        ),
        "reports_tables_dir": require_relative_dir_path(
            require_key(reports, "tables_dir", "paths.reports"), "paths.reports.tables_dir"
        ),
        "results_metrics_dir": require_relative_dir_path(
            require_key(results, "metrics_dir", "paths.results"), "paths.results.metrics_dir"
        ),
    }

    return resolved


# ----------------------------- ENTRYPOINT ---------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict validation for default project config.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default: config/default_config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()

    log_info("==================== CONFIG VALIDATION ====================")
    log_info(f"REPO_ROOT: {REPO_ROOT}")
    log_info(f"Config path: {config_path}")

    config = load_yaml_config(config_path)
    log_info("Top-level keys validated: project, paths, files, schema")

    log_info("==================== RESOLVED DIRECTORIES ====================")
    resolved_dirs = resolve_required_directories(config)
    for key in (
        "raw_gdsc_dir",
        "raw_depmap_dir",
        "harmonised_dir",
        "per_drug_dir",
        "reports_tables_dir",
        "results_metrics_dir",
    ):
        log_info(f"{key}: {resolved_dirs[key]}")

    log_info("==================== VALIDATION STATUS ====================")
    log_info("Config validation completed successfully.")


if __name__ == "__main__":
    main()
