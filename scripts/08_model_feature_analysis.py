#!/usr/bin/env python3
"""
08_model_feature_analysis.py

Purpose
-------
Add a final model-interpretation layer for a single-drug ElasticNet model by
analysing coefficient structure and generating publication-style feature plots.

Inputs
------
- config/default_config.yaml
- results/metrics/<drug_slug>_elasticnet_metrics.json
- results/metrics/<drug_slug>_elasticnet_all_nonzero_features.csv
- data/processed/per_drug/<drug_slug>.parquet

Outputs
-------
- results/metrics/<drug_slug>_elasticnet_coefficient_summary.json
- results/metrics/<drug_slug>_elasticnet_top_positive_features.csv
- results/metrics/<drug_slug>_elasticnet_top_negative_features.csv
- results/plots/<drug_slug>_coefficient_distribution.png
- results/plots/<drug_slug>_top_positive_coefficients.png
- results/plots/<drug_slug>_top_negative_coefficients.png

Usage
-----
python scripts/08_model_feature_analysis.py --drug "Selumetinib"
python scripts/08_model_feature_analysis.py --config config/default_config.yaml --drug "Selumetinib"

Notes
-----
- Paths are resolved relative to REPO_ROOT.
- Missing required files or columns cause immediate exit.
- Plotting uses matplotlib only (no seaborn).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ----------------------------- CONFIGURATION -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"
LOGGER = logging.getLogger("model_feature_analysis")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_info(message: str) -> None:
    LOGGER.info("[INFO] %s", message)


def log_warn(message: str) -> None:
    LOGGER.warning("[WARN] %s", message)


def log_error(message: str) -> None:
    LOGGER.error("[ERROR] %s", message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse ElasticNet coefficients and generate feature-importance plots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--drug",
        type=str,
        required=True,
        help="Drug name (e.g. 'Selumetinib').",
    )
    return parser.parse_args()


def slugify_drug_name(drug_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", drug_name.strip().lower())
    return slug.strip("_")


def resolve_path_from_repo(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    return (REPO_ROOT / path_value).resolve()


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        log_error(f"Expected mapping at '{context}', got {type(value).__name__}.")
        raise SystemExit(1)
    return value


def require_non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Expected non-empty string at '{context}'.")
        raise SystemExit(1)
    return value.strip()


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception as exc:
        log_error(f"Failed to parse YAML config '{path}': {exc}")
        raise SystemExit(1) from exc

    return require_mapping(parsed, "root")


# ----------------------------- VALIDATION ----------------------------------- #

def require_file_exists(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        log_error(f"Missing required {label}: {path}")
        raise SystemExit(1)


def validate_feature_columns(df: pd.DataFrame) -> None:
    required_columns = {"feature", "coefficient", "abs_coefficient"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        log_error(f"Missing required columns in feature table: {missing}")
        raise SystemExit(1)


# ----------------------------- DATA LOADING --------------------------------- #

def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
    except Exception as exc:
        log_error(f"Failed to read {label} JSON '{path}': {exc}")
        raise SystemExit(1) from exc

    if not isinstance(parsed, dict):
        log_error(f"Expected JSON object in {label}, got {type(parsed).__name__}.")
        raise SystemExit(1)
    return parsed


def load_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log_error(f"Failed to read {label} CSV '{path}': {exc}")
        raise SystemExit(1) from exc


# ----------------------------- ANALYSIS ------------------------------------- #

def compute_summary_stats(coef_df: pd.DataFrame) -> dict[str, Any]:
    coefficients = pd.to_numeric(coef_df["coefficient"], errors="coerce")
    abs_coefficients = pd.to_numeric(coef_df["abs_coefficient"], errors="coerce")

    if coefficients.isna().any() or abs_coefficients.isna().any():
        log_error("Non-numeric coefficient values detected in feature table.")
        raise SystemExit(1)

    positive = coefficients[coefficients > 0]
    negative = coefficients[coefficients < 0]

    summary: dict[str, Any] = {
        "n_nonzero_features": int(len(coef_df)),
        "max_positive_coefficient": float(positive.max()) if not positive.empty else None,
        "min_negative_coefficient": float(negative.min()) if not negative.empty else None,
        "mean_abs_coefficient": float(abs_coefficients.mean()) if not abs_coefficients.empty else None,
        "median_abs_coefficient": float(abs_coefficients.median()) if not abs_coefficients.empty else None,
        "n_positive_coefficients": int((coefficients > 0).sum()),
        "n_negative_coefficients": int((coefficients < 0).sum()),
    }
    return summary


def build_ranked_tables(coef_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_numeric = coef_df.copy()
    coef_numeric["coefficient"] = pd.to_numeric(coef_numeric["coefficient"], errors="coerce")
    coef_numeric["abs_coefficient"] = pd.to_numeric(coef_numeric["abs_coefficient"], errors="coerce")

    positive_df = coef_numeric[coef_numeric["coefficient"] > 0].sort_values(
        "coefficient",
        ascending=False,
    ).head(25)
    negative_df = coef_numeric[coef_numeric["coefficient"] < 0].sort_values(
        "coefficient",
        ascending=True,
    ).head(25)

    return positive_df, negative_df


# ----------------------------- PLOTTING ------------------------------------- #

def plot_coefficient_distribution(coefficients: pd.Series, drug_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(coefficients, bins=30)
    ax.set_xlabel("coefficient")
    ax.set_ylabel("count")
    ax.set_title(f"{drug_name} ElasticNet coefficient distribution")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_top_coefficients(
    df: pd.DataFrame,
    drug_name: str,
    output_path: Path,
    title_suffix: str,
) -> None:
    plot_df = df.head(15).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_df.empty:
        log_warn(f"No rows available for plot: {output_path}")
        ax.text(0.5, 0.5, "No features available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.barh(plot_df["feature"], plot_df["coefficient"])
        ax.set_xlabel("coefficient")
        ax.set_ylabel("feature")
        ax.invert_yaxis()

    ax.set_title(f"{drug_name} {title_suffix}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ----------------------------- OUTPUT --------------------------------------- #

def main() -> None:
    configure_logging()
    args = parse_args()

    config_path = resolve_path_from_repo(args.config)
    require_file_exists(config_path, "config file")
    config = load_config(config_path)

    drug_slug = slugify_drug_name(args.drug)
    if not drug_slug:
        log_error("Drug name produced an empty slug. Provide a valid --drug value.")
        raise SystemExit(1)

    paths = require_mapping(config.get("paths"), "paths")
    results_paths = require_mapping(paths.get("results"), "paths.results")
    processed_paths = require_mapping(paths.get("processed"), "paths.processed")

    metrics_dir = resolve_path_from_repo(
        Path(require_non_empty_string(results_paths.get("metrics_dir"), "paths.results.metrics_dir"))
    )
    plots_dir_raw = results_paths.get("plots_dir", "results/plots")
    plots_dir = resolve_path_from_repo(
        Path(require_non_empty_string(plots_dir_raw, "paths.results.plots_dir"))
    )
    per_drug_dir = resolve_path_from_repo(
        Path(require_non_empty_string(processed_paths.get("per_drug_dir"), "paths.processed.per_drug_dir"))
    )

    metrics_json_path = metrics_dir / f"{drug_slug}_elasticnet_metrics.json"
    all_features_path = metrics_dir / f"{drug_slug}_elasticnet_all_nonzero_features.csv"

    require_file_exists(metrics_json_path, "metrics JSON")
    require_file_exists(all_features_path, "all non-zero features CSV")
    if not per_drug_dir.exists() or not per_drug_dir.is_dir():
        log_error(f"Missing required per-drug directory: {per_drug_dir}")
        raise SystemExit(1)

    _metrics = load_json(metrics_json_path, "metrics")
    coef_df = load_csv(all_features_path, "all non-zero features")

    validate_feature_columns(coef_df)

    summary = compute_summary_stats(coef_df)
    top_positive_df, top_negative_df = build_ranked_tables(coef_df)

    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / f"{drug_slug}_elasticnet_coefficient_summary.json"
    top_positive_path = metrics_dir / f"{drug_slug}_elasticnet_top_positive_features.csv"
    top_negative_path = metrics_dir / f"{drug_slug}_elasticnet_top_negative_features.csv"
    distribution_plot_path = plots_dir / f"{drug_slug}_coefficient_distribution.png"
    positive_plot_path = plots_dir / f"{drug_slug}_top_positive_coefficients.png"
    negative_plot_path = plots_dir / f"{drug_slug}_top_negative_coefficients.png"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    top_positive_df.to_csv(top_positive_path, index=False)
    top_negative_df.to_csv(top_negative_path, index=False)

    coefficient_series = pd.to_numeric(coef_df["coefficient"], errors="coerce")
    if coefficient_series.isna().any():
        log_error("Non-numeric coefficient values detected; cannot plot distribution.")
        raise SystemExit(1)

    plot_coefficient_distribution(
        coefficients=coefficient_series,
        drug_name=args.drug,
        output_path=distribution_plot_path,
    )
    plot_top_coefficients(
        df=top_positive_df,
        drug_name=args.drug,
        output_path=positive_plot_path,
        title_suffix="top positive ElasticNet coefficients",
    )
    plot_top_coefficients(
        df=top_negative_df,
        drug_name=args.drug,
        output_path=negative_plot_path,
        title_suffix="top negative ElasticNet coefficients",
    )

    log_info(f"Number of non-zero features: {summary['n_nonzero_features']}")
    log_info(f"Number of positive features: {summary['n_positive_coefficients']}")
    log_info(f"Number of negative features: {summary['n_negative_coefficients']}")

    written_paths = [
        summary_path,
        top_positive_path,
        top_negative_path,
        distribution_plot_path,
        positive_plot_path,
        negative_plot_path,
    ]
    for path in written_paths:
        log_info(f"Wrote: {path}")


if __name__ == "__main__":
    main()
