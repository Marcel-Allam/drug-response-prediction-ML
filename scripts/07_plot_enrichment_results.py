#!/usr/bin/env python3
"""
07_plot_enrichment_results.py

Purpose
-------
Create publication-style pathway enrichment barplots for a single drug.

Inputs
------
- config/default_config.yaml
- results/enrichment/<drug_slug>_kegg_enrichment.csv
- results/enrichment/<drug_slug>_reactome_enrichment.csv
- results/enrichment/<drug_slug>_gobp_enrichment.csv

Outputs
-------
- results/plots/<drug_slug>_kegg_barplot.png
- results/plots/<drug_slug>_reactome_barplot.png
- results/plots/<drug_slug>_gobp_barplot.png

Usage
-----
python scripts/07_plot_enrichment_results.py --drug "Selumetinib"
python scripts/07_plot_enrichment_results.py --config config/default_config.yaml --drug "Selumetinib"

Notes
-----
- Plots use matplotlib only (no seaborn).
- Top pathways are selected by ascending adjusted p-value.
- If adjusted p-values include zeros, zeros are replaced with a small positive value.
"""

from __future__ import annotations

import argparse
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
LOGGER = logging.getLogger("enrichment_plotter")


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
        description="Create pathway enrichment barplots for a single drug."
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


def resolve_path_from_repo(path_value: Any, context: str) -> Path:
    raw = Path(require_non_empty_string(path_value, context))
    if raw.is_absolute():
        return raw.resolve()
    return (REPO_ROOT / raw).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists() or not config_path.is_file():
        log_error(f"Config file does not exist: {config_path}")
        raise SystemExit(1)

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception as exc:
        log_error(f"Failed to parse YAML config '{config_path}': {exc}")
        raise SystemExit(1) from exc

    return require_mapping(parsed, "root")


# ----------------------------- VALIDATION ----------------------------------- #

def validate_input_files(enrichment_files: dict[str, Path]) -> None:
    missing = [path for path in enrichment_files.values() if not path.exists() or not path.is_file()]
    if missing:
        for path in missing:
            log_error(f"Missing required enrichment file: {path}")
        raise SystemExit(1)


# ----------------------------- DATA LOADING --------------------------------- #

def load_enrichment_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log_error(f"Failed to read enrichment file '{path}': {exc}")
        raise SystemExit(1) from exc


# ----------------------------- PLOTTING ------------------------------------- #

def create_barplot(
    df: pd.DataFrame,
    drug_name: str,
    database_label: str,
    output_path: Path,
) -> None:
    if df.empty:
        log_warn(f"Enrichment table is empty, skipping plot: {output_path}")
        return

    required_cols = {"Term", "Adjusted P-value"}
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        log_error(f"Missing required columns {missing} in enrichment table for '{database_label}'.")
        raise SystemExit(1)

    plot_df = df.sort_values("Adjusted P-value", ascending=True).head(10).copy()

    if plot_df.empty:
        log_warn(f"No rows available for plotting after filtering: {output_path}")
        return

    adjusted_p = pd.to_numeric(plot_df["Adjusted P-value"], errors="coerce")
    if adjusted_p.isna().any():
        log_error(f"Non-numeric adjusted p-values found for '{database_label}'.")
        raise SystemExit(1)

    adjusted_p = adjusted_p.replace(0, np.finfo(float).tiny)
    scores = -np.log10(adjusted_p)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["Term"], scores)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(f"{drug_name} - {database_label} pathway enrichment")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ----------------------------- OUTPUT --------------------------------------- #

def main() -> None:
    configure_logging()
    args = parse_args()

    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config).resolve()
    config = load_config(config_path)

    drug_slug = slugify_drug_name(args.drug)
    if not drug_slug:
        log_error("Drug name produced an empty slug. Provide a valid --drug value.")
        raise SystemExit(1)

    paths = require_mapping(config.get("paths"), "paths")
    results_paths = require_mapping(paths.get("results"), "paths.results")

    enrichment_dir = resolve_path_from_repo(
        results_paths.get("enrichment_dir", "results/enrichment"),
        "paths.results.enrichment_dir",
    )
    plots_dir = resolve_path_from_repo(
        results_paths.get("plots_dir", "results/plots"),
        "paths.results.plots_dir",
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    enrichment_files = {
        "KEGG": enrichment_dir / f"{drug_slug}_kegg_enrichment.csv",
        "Reactome": enrichment_dir / f"{drug_slug}_reactome_enrichment.csv",
        "GO_BP": enrichment_dir / f"{drug_slug}_gobp_enrichment.csv",
    }
    output_files = {
        "KEGG": plots_dir / f"{drug_slug}_kegg_barplot.png",
        "Reactome": plots_dir / f"{drug_slug}_reactome_barplot.png",
        "GO_BP": plots_dir / f"{drug_slug}_gobp_barplot.png",
    }

    validate_input_files(enrichment_files)

    for db_key, input_path in enrichment_files.items():
        suffix = "gobp" if db_key == "GO_BP" else db_key.lower()
        sig_path = enrichment_dir / f"{drug_slug}_{suffix}_enrichment_sig.csv"

        if sig_path.exists() and sig_path.is_file():
            sig_df = load_enrichment_table(sig_path)
            if not sig_df.empty:
                log_info(f"Processing enrichment file: {sig_path}")
                df = sig_df
            else:
                log_warn(
                    f"Significant enrichment table empty for {db_key}; "
                    f"falling back to full table: {input_path}"
                )
                log_info(f"Processing enrichment file: {input_path}")
                df = load_enrichment_table(input_path)
        else:
            log_warn(
                f"Significant enrichment table unavailable for {db_key}; "
                f"falling back to full table: {input_path}"
            )
            log_info(f"Processing enrichment file: {input_path}")
            df = load_enrichment_table(input_path)

        create_barplot(
            df=df,
            drug_name=args.drug,
            database_label=db_key if db_key != "GO_BP" else "GO-BP",
            output_path=output_files[db_key],
        )
        if output_files[db_key].exists():
            log_info(f"Saved plot: {output_files[db_key]}")


if __name__ == "__main__":
    main()
