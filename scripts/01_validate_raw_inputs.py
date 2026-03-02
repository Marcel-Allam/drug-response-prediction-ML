#!/usr/bin/env python3
"""
01_validate_raw_inputs.py

Purpose
-------
Validate that raw data files declared in config/default_config.yaml exist
in expected directories before downstream processing starts.
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

LOGGER = logging.getLogger("raw_input_validator")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_info(message: str) -> None:
    LOGGER.info("[INFO] %s", message)


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


def require_non_empty_filename(value: Any, key_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Filename at '{key_path}' must be a non-empty string.")
        raise SystemExit(1)
    return value.strip()


def require_relative_dir(value: Any, key_path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Directory path at '{key_path}' must be a non-empty string.")
        raise SystemExit(1)

    rel_path = Path(value)
    if rel_path.is_absolute():
        log_error(f"Directory path at '{key_path}' must be relative to REPO_ROOT: {rel_path}")
        raise SystemExit(1)

    return (REPO_ROOT / rel_path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        log_error(f"Config file does not exist: {path}")
        raise SystemExit(1)

    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception as exc:
        log_error(f"Failed to parse YAML config '{path}': {exc}")
        raise SystemExit(1) from exc

    config = require_mapping(parsed, "root")
    for top_key in ("project", "paths", "files", "schema"):
        require_key(config, top_key, "root")
    return config


def validate_raw_file_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = require_mapping(require_key(config, "paths", "root"), "paths")
    files = require_mapping(require_key(config, "files", "root"), "files")

    raw_paths = require_mapping(require_key(paths, "raw", "paths"), "paths.raw")
    gdsc_dir = require_relative_dir(require_key(raw_paths, "gdsc_dir", "paths.raw"), "paths.raw.gdsc_dir")
    depmap_dir = require_relative_dir(require_key(raw_paths, "depmap_dir", "paths.raw"), "paths.raw.depmap_dir")

    gdsc_files = require_mapping(require_key(files, "gdsc", "files"), "files.gdsc")
    depmap_files = require_mapping(require_key(files, "depmap", "files"), "files.depmap")

    gdsc_response_name = require_non_empty_filename(
        require_key(gdsc_files, "response", "files.gdsc"), "files.gdsc.response"
    )
    depmap_expression_name = require_non_empty_filename(
        require_key(depmap_files, "expression", "files.depmap"), "files.depmap.expression"
    )
    depmap_prism_response_name = require_non_empty_filename(
        require_key(depmap_files, "prism_response", "files.depmap"), "files.depmap.prism_response"
    )

    checked_files = {
        "gdsc_response": gdsc_dir / gdsc_response_name,
        "depmap_expression": depmap_dir / depmap_expression_name,
        "depmap_prism_response": depmap_dir / depmap_prism_response_name,
    }

    for label, file_path in checked_files.items():
        if not file_path.exists() or not file_path.is_file():
            log_error(f"Missing required file '{label}': {file_path}")
            raise SystemExit(1)

    return checked_files


# ----------------------------- ENTRYPOINT ---------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate required raw input file paths from config.")
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
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()

    log_info("==================== RAW INPUT VALIDATION ====================")
    log_info(f"REPO_ROOT: {REPO_ROOT}")
    log_info(f"Config path: {config_path}")

    config = load_config(config_path)
    resolved_files = validate_raw_file_paths(config)

    log_info("==================== VALIDATED FILE PATHS ====================")
    log_info(f"gdsc_response: {resolved_files['gdsc_response'].resolve()}")
    log_info(f"depmap_expression: {resolved_files['depmap_expression'].resolve()}")
    log_info(f"depmap_prism_response: {resolved_files['depmap_prism_response'].resolve()}")

    log_info("==================== VALIDATION STATUS ====================")
    log_info("Raw input file validation completed successfully.")


if __name__ == "__main__":
    main()
