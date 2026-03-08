#!/usr/bin/env python3
"""
06_run_pathway_enrichment.py

Purpose
-------
Run pathway enrichment analysis for a single drug using an ElasticNet-derived
plain-text gene list.

Inputs
------
- config/default_config.yaml
- results/metrics/<drug_slug>_elasticnet_top_genes.csv
- results/metrics/<drug_slug>_elasticnet_gene_list.txt

Outputs
-------
- results/enrichment/<drug_slug>_kegg_enrichment.csv
- results/enrichment/<drug_slug>_reactome_enrichment.csv
- results/enrichment/<drug_slug>_gobp_enrichment.csv
- results/enrichment/<drug_slug>_kegg_enrichment_sig.csv
- results/enrichment/<drug_slug>_reactome_enrichment_sig.csv
- results/enrichment/<drug_slug>_gobp_enrichment_sig.csv

Usage
-----
python scripts/06_run_pathway_enrichment.py --drug "Selumetinib"
python scripts/06_run_pathway_enrichment.py --config config/default_config.yaml --drug "Selumetinib"

Notes
-----
- Gene list input is read from the plain text file (one gene per line).
- Blank lines are ignored.
- Enrichment is run with Enrichr via gseapy.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ----------------------------- CONFIGURATION -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"
LOGGER = logging.getLogger("pathway_enrichment_runner")


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
        description="Run pathway enrichment on ElasticNet top genes for a single drug."
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
    parser.add_argument(
        "--source",
        type=str,
        choices=["top_features", "all_nonzero_features"],
        default="all_nonzero_features",
        help="Gene list source to use for enrichment.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Top-N ranked genes to use when --source=all_nonzero_features.",
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

def validate_input_files(gene_list_path: Path) -> None:
    if not gene_list_path.exists() or not gene_list_path.is_file():
        log_error(f"Missing required file: {gene_list_path}")
        raise SystemExit(1)


# ----------------------------- DATA LOADING --------------------------------- #

def load_gene_list(gene_list_path: Path) -> list[str]:
    try:
        with gene_list_path.open("r", encoding="utf-8") as handle:
            genes = [line.strip() for line in handle if line.strip()]
    except Exception as exc:
        log_error(f"Failed to read gene list file '{gene_list_path}': {exc}")
        raise SystemExit(1) from exc

    if not genes:
        log_error(f"Gene list is empty after stripping blank lines: {gene_list_path}")
        raise SystemExit(1)

    return genes


# ----------------------------- ENRICHMENT ----------------------------------- #

def run_single_enrichment(gene_list: list[str], gene_set: str) -> pd.DataFrame:
    try:
        import gseapy as gp
    except ImportError as exc:
        log_error("gseapy is required but not installed. Install it and re-run.")
        raise SystemExit(1) from exc

    try:
        enrichment = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_set,
            organism="human",
            outdir=None,
            no_plot=True,
        )
    except Exception as exc:
        log_error(f"Enrichment failed for database '{gene_set}': {exc}")
        raise SystemExit(1) from exc

    if not hasattr(enrichment, "results") or not isinstance(enrichment.results, pd.DataFrame):
        log_error(f"No valid results table returned for database '{gene_set}'.")
        raise SystemExit(1)

    return enrichment.results.copy()


# ----------------------------- OUTPUT --------------------------------------- #

def save_enrichment_outputs(
    enrichment_dir: Path,
    drug_slug: str,
    label: str,
    results_df: pd.DataFrame,
) -> int:
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    full_path = enrichment_dir / f"{drug_slug}_{label}_enrichment.csv"
    sig_path = enrichment_dir / f"{drug_slug}_{label}_enrichment_sig.csv"

    results_df.to_csv(full_path, index=False)

    if "Adjusted P-value" not in results_df.columns:
        log_error(f"Results table for '{label}' is missing 'Adjusted P-value' column.")
        raise SystemExit(1)

    sig_df = results_df[results_df["Adjusted P-value"] < 0.05].copy()
    sig_df.to_csv(sig_path, index=False)

    if sig_df.empty:
        log_warn(f"No significant pathways found for '{label}' at Adjusted P-value < 0.05.")
    else:
        preview = sig_df["Term"].head(5).tolist()
        log_info(f"Top pathways: {', '.join(preview)}")

    log_info(f"Saved full enrichment table: {full_path}")
    log_info(f"Saved significant enrichment table: {sig_path}")

    return int(sig_df.shape[0])


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

    metrics_dir = resolve_path_from_repo(results_paths.get("metrics_dir"), "paths.results.metrics_dir")
    enrichment_dir = resolve_path_from_repo(
        results_paths.get("enrichment_dir", "results/enrichment"),
        "paths.results.enrichment_dir",
    )

    if args.source == "top_features":
        gene_list_path = metrics_dir / f"{drug_slug}_elasticnet_gene_list.txt"
    elif args.top_n is None:
        gene_list_path = metrics_dir / f"{drug_slug}_elasticnet_all_nonzero_gene_list.txt"
    else:
        gene_list_path = (
            metrics_dir / f"{drug_slug}_elasticnet_all_nonzero_top_{args.top_n}_gene_list.txt"
        )

    top_n_used = args.top_n is not None and args.source == "all_nonzero_features"
    log_info(
        f"Gene list selection -> source: {args.source}; "
        f"top_n_used: {top_n_used}; "
        f"gene_list_file: {gene_list_path}"
    )

    validate_input_files(gene_list_path=gene_list_path)
    gene_list = load_gene_list(gene_list_path)
    gene_list = sorted(set(gene_list))

    log_info(f"Number of input genes: {len(gene_list)}")

    databases = {
        "KEGG_2021_Human": "kegg",
        "Reactome_2022": "reactome",
        "GO_Biological_Process_2023": "gobp",
    }
    log_info(f"Databases queried: {', '.join(databases.keys())}")

    for db_name, label in databases.items():
        log_info(f"Running enrichment for database: {db_name}")
        results_df = run_single_enrichment(gene_list=gene_list, gene_set=db_name)
        n_sig = save_enrichment_outputs(
            enrichment_dir=enrichment_dir,
            drug_slug=drug_slug,
            label=label,
            results_df=results_df,
        )
        log_info(f"Significant pathways in {db_name}: {n_sig}")


if __name__ == "__main__":
    main()
