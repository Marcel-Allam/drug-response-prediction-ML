#!/usr/bin/env python3
"""
00_inspect_datasets.py

Purpose
-------
Inspect raw GDSC and DepMap files to identify likely expression/response tables,
infer schema/orientation, and summarise dataset readiness for harmonisation.

Inputs
------
- data/raw/gdsc/   (raw GDSC files; not committed)
- data/raw/depmap/ (raw DepMap files; not committed)
- config/default_config.yaml (optional; used if present)

Outputs
-------
- reports/tables/dataset_inspection_summary.csv
- reports/tables/dataset_inspection_summary.md

Usage
-----
python scripts/00_inspect_datasets.py --config config/default_config.yaml

Notes
-----
- This script does NOT download data.
- It reads only small file heads for safety (unless a file is Parquet).
- Filenames vary by release; this script uses heuristics to find candidates.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# ----------------------------- CONFIGURATION -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"

DEFAULT_RAW_GDSC_DIR = REPO_ROOT / "data" / "raw" / "gdsc"
DEFAULT_RAW_DEPMAP_DIR = REPO_ROOT / "data" / "raw" / "depmap"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "tables"

CANDIDATE_SAMPLE_ID_COLS = [
    "DepMap_ID",
    "depmap_id",
    "CCLE_Name",
    "ccle_name",
    "cell_line",
    "cellline",
    "model_id",
    "ModelID",
    "COSMIC_ID",
    "cosmic_id",
    "Sample",
    "sample",
    "sample_id",
]

CANDIDATE_DRUG_COLS = ["drug", "compound", "treatment", "perturbation", "name"]
CANDIDATE_RESPONSE_COLS = ["ic50", "auc", "viability", "response", "lfc", "logfc"]

ENSEMBL_REGEX = re.compile(r"^ENSG\d+")
GENE_SYMBOL_REGEX = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,}$")


# ------------------------------- LOGGING ------------------------------------ #

def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


# ------------------------------- UTILITIES ---------------------------------- #

@dataclass(frozen=True)
class FileCandidate:
    path: Path
    inferred_kind: str  # "expression" | "response" | "unknown"
    inferred_format: str


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_nonempty_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files: List[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.name != ".gitkeep":
            files.append(p)
    return files


def infer_kind_from_name(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ["expression", "rnaseq", "rna_seq", "transcript", "tpm", "counts"]):
        return "expression"
    if any(k in name for k in ["ic50", "auc", "dose", "response", "prism", "drug", "viability", "lfc", "repurposing"]):
        return "response"
    return "unknown"


def read_table_head(path: Path, nrows: int = 50) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).head(nrows)
    if suffix in [".csv", ".tsv", ".txt"]:
        if suffix in [".tsv", ".txt"]:
            try:
                return pd.read_csv(path, sep="\t", nrows=nrows)
            except Exception:
                return pd.read_csv(path, nrows=nrows)
        return pd.read_csv(path, nrows=nrows)
    raise ValueError(f"Unsupported file format: {path.name}")


def guess_expression_orientation(df_head: pd.DataFrame) -> str:
    cols_lower = [str(c).lower() for c in df_head.columns]
    if any("gene" in c for c in cols_lower) and any(k in " ".join(cols_lower) for k in ["value", "tpm", "count", "expr", "expression"]):
        return "long_format_likely"

    gene_like_cols = 0
    for c in df_head.columns[: min(200, len(df_head.columns))]:
        cs = str(c)
        if ENSEMBL_REGEX.match(cs) or GENE_SYMBOL_REGEX.match(cs):
            gene_like_cols += 1

    if gene_like_cols >= 20:
        return "samples_rows_genes_cols_likely"

    first_col = str(df_head.columns[0])
    try:
        frac_ensembl = df_head[first_col].astype(str).str.match(ENSEMBL_REGEX).mean()
        if frac_ensembl > 0.5:
            return "genes_rows_samples_cols_likely"
    except Exception:
        pass

    return "unknown"


def find_candidate_id_cols(df_head: pd.DataFrame) -> List[str]:
    found: List[str] = []
    for col in df_head.columns:
        if str(col) in CANDIDATE_SAMPLE_ID_COLS:
            found.append(str(col))
    for col in df_head.columns:
        c = str(col).lower()
        if ("id" in c or "cell" in c or "sample" in c) and str(col) not in found:
            found.append(str(col))
    return found[:10]


def find_candidate_response_cols(df_head: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [str(c) for c in df_head.columns]
    drug_cols = [c for c in cols if any(k in c.lower() for k in CANDIDATE_DRUG_COLS)]
    response_cols = [c for c in cols if any(k in c.lower() for k in CANDIDATE_RESPONSE_COLS)]
    return drug_cols[:10], response_cols[:10]


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        log_warn("PyYAML not installed; continuing without config parsing.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_paths_from_config(config: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    raw_gdsc_dir = DEFAULT_RAW_GDSC_DIR
    raw_depmap_dir = DEFAULT_RAW_DEPMAP_DIR
    reports_dir = DEFAULT_REPORTS_DIR

    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    reports_cfg = config.get("reports", {}) if isinstance(config.get("reports", {}), dict) else {}

    if isinstance(data_cfg.get("raw_gdsc_dir"), str):
        raw_gdsc_dir = (REPO_ROOT / data_cfg["raw_gdsc_dir"]).resolve()
    if isinstance(data_cfg.get("raw_depmap_dir"), str):
        raw_depmap_dir = (REPO_ROOT / data_cfg["raw_depmap_dir"]).resolve()
    if isinstance(reports_cfg.get("tables_dir"), str):
        reports_dir = (REPO_ROOT / reports_cfg["tables_dir"]).resolve()

    return raw_gdsc_dir, raw_depmap_dir, reports_dir


def build_candidates(directory: Path) -> List[FileCandidate]:
    candidates: List[FileCandidate] = []
    for p in list_nonempty_files(directory):
        candidates.append(
            FileCandidate(
                path=p,
                inferred_kind=infer_kind_from_name(p.name),
                inferred_format=p.suffix.lower().lstrip("."),
            )
        )
    return candidates


def summarise_candidate(candidate: FileCandidate) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "file": str(candidate.path.relative_to(REPO_ROOT)),
        "format": candidate.inferred_format,
        "inferred_kind": candidate.inferred_kind,
        "head_rows_read": None,
        "head_cols": None,
        "orientation_guess": None,
        "candidate_id_cols": None,
        "candidate_drug_cols": None,
        "candidate_response_cols": None,
        "read_status": "OK",
        "notes": "",
    }

    try:
        df_head = read_table_head(candidate.path, nrows=50)
        summary["head_rows_read"] = int(df_head.shape[0])
        summary["head_cols"] = int(df_head.shape[1])
        summary["candidate_id_cols"] = ", ".join(find_candidate_id_cols(df_head))

        if candidate.inferred_kind == "expression":
            summary["orientation_guess"] = guess_expression_orientation(df_head)

        if candidate.inferred_kind == "response":
            drug_cols, resp_cols = find_candidate_response_cols(df_head)
            summary["candidate_drug_cols"] = ", ".join(drug_cols)
            summary["candidate_response_cols"] = ", ".join(resp_cols)

    except Exception as e:
        summary["read_status"] = "FAILED"
        summary["notes"] = f"{type(e).__name__}: {e}"

    return summary


def write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Dataset Inspection Summary")
    lines.append("")
    lines.append("Generated by `scripts/00_inspect_datasets.py`.")
    lines.append("")
    if df.empty:
        lines.append("_No files detected in the raw data directories._")
        lines.append("")
    else:
        lines.append(df.to_markdown(index=False))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------- MAIN ------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw GDSC/DepMap files and summarise candidates.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file (optional). Defaults to config/default_config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    log_info(f"Repo root: {REPO_ROOT}")
    log_info(f"Config path: {config_path}")

    config = load_yaml_config(config_path)
    raw_gdsc_dir, raw_depmap_dir, reports_dir = resolve_paths_from_config(config)

    safe_mkdir(reports_dir)

    log_info(f"Raw GDSC dir: {raw_gdsc_dir}")
    log_info(f"Raw DepMap dir: {raw_depmap_dir}")
    log_info(f"Reports dir: {reports_dir}")

    gdsc_candidates = build_candidates(raw_gdsc_dir)
    depmap_candidates = build_candidates(raw_depmap_dir)

    if not raw_gdsc_dir.exists():
        log_error(f"Missing directory: {raw_gdsc_dir}")
    if not raw_depmap_dir.exists():
        log_error(f"Missing directory: {raw_depmap_dir}")

    if len(gdsc_candidates) == 0:
        log_warn(f"No files found in {raw_gdsc_dir}")
    if len(depmap_candidates) == 0:
        log_warn(f"No files found in {raw_depmap_dir}")

    rows: List[Dict[str, Any]] = []

    for c in gdsc_candidates:
        log_info(f"GDSC candidate: {c.path.name} [{c.inferred_kind}]")
        row = summarise_candidate(c)
        row["dataset"] = "GDSC"
        rows.append(row)

    for c in depmap_candidates:
        log_info(f"DepMap candidate: {c.path.name} [{c.inferred_kind}]")
        row = summarise_candidate(c)
        row["dataset"] = "DepMap"
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    csv_out = reports_dir / "dataset_inspection_summary.csv"
    md_out = reports_dir / "dataset_inspection_summary.md"

    summary_df.to_csv(csv_out, index=False)
    write_markdown_table(summary_df, md_out)

    log_info(f"Wrote CSV summary: {csv_out.relative_to(REPO_ROOT)}")
    log_info(f"Wrote Markdown summary: {md_out.relative_to(REPO_ROOT)}")

    log_info("Next steps:")
    log_info("1) Confirm which files are expression matrices vs response tables.")
    log_info("2) Confirm gene identifiers (HGNC symbols vs Ensembl IDs).")
    log_info("3) Proceed to scripts/01_harmonise_genes.py once files are identified.")


if __name__ == "__main__":
    main()