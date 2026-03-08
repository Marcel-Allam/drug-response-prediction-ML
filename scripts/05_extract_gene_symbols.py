#!/usr/bin/env python3
"""
05_extract_gene_symbols.py

Extract gene symbols from ElasticNet top feature file.

Input:
    results/metrics/<drug>_elasticnet_top_features.csv

Output:
    results/metrics/<drug>_elasticnet_top_genes.csv
    results/metrics/<drug>_elasticnet_gene_list.txt
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd


# ----------------------------- CONFIG -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("extract_genes")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_info(msg: str) -> None:
    LOGGER.info("[INFO] %s", msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drug",
        type=str,
        required=True,
        help="Drug name (e.g. Selumetinib)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["top_features", "all_nonzero_features"],
        default="all_nonzero_features",
        help="Feature source file to extract genes from.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Keep only top N ranked rows from the selected feature file before extraction.",
    )
    return parser.parse_args()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ----------------------------- MAIN -------------------------------- #

def main() -> None:
    configure_logging()
    args = parse_args()

    drug_slug = slugify(args.drug)
    top_suffix = f"_top_{args.top_n}" if args.top_n is not None else ""

    if args.source == "top_features":
        input_path = REPO_ROOT / "results" / "metrics" / f"{drug_slug}_elasticnet_top_features.csv"
        output_csv = REPO_ROOT / "results" / "metrics" / f"{drug_slug}_elasticnet_top{top_suffix}_genes.csv"
        output_txt = REPO_ROOT / "results" / "metrics" / f"{drug_slug}_elasticnet{top_suffix}_gene_list.txt"
    else:
        input_path = REPO_ROOT / "results" / "metrics" / f"{drug_slug}_elasticnet_all_nonzero_features.csv"
        output_csv = REPO_ROOT / "results" / "metrics" / f"{drug_slug}_elasticnet_all_nonzero{top_suffix}_genes.csv"
        output_txt = REPO_ROOT / "results" / "metrics" / f"{drug_slug}_elasticnet_all_nonzero{top_suffix}_gene_list.txt"

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    log_info(f"Reading features: {input_path}")
    df = pd.read_csv(input_path)
    if args.top_n is not None:
        df = df.head(args.top_n)

    # Extract gene symbol before "(ENTREZ)"
    df["gene_symbol"] = df["feature"].str.extract(r"^([A-Za-z0-9\-]+)")

    df_clean = df[["gene_symbol", "coefficient", "abs_coefficient"]]

    log_info(f"Writing cleaned gene table: {output_csv}")
    df_clean.to_csv(output_csv, index=False)

    log_info(f"Writing gene list for enrichment: {output_txt}")
    df_clean["gene_symbol"].to_csv(output_txt, index=False, header=False)

    log_info("Gene extraction complete")


if __name__ == "__main__":
    main()
