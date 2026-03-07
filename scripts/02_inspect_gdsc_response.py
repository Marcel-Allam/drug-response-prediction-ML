#!/usr/bin/env python3
"""
02_inspect_gdsc_response.py

Purpose
-------
Inspect the raw GDSC response Excel file declared in config/default_config.yaml,
summarise dataset schema, and write inspection reports for downstream drug selection.

Inputs
------
- config/default_config.yaml
- paths.raw.gdsc_dir
- files.gdsc.response (Excel .xlsx file)

Outputs
-------
- reports/tables/gdsc_response_schema_summary.csv
- reports/tables/gdsc_per_drug_summary.csv
- reports/tables/gdsc_response_schema_summary.md

Usage
-----
python scripts/02_inspect_gdsc_response.py --config config/default_config.yaml

Notes
-----
- Inspection only; no modelling or train/test splitting is performed.
- Paths are resolved relative to REPO_ROOT.
- Column detection is conservative: ambiguous matches are reported and not forced.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ----------------------------- CONFIGURATION -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"

SCHEMA_SUMMARY_CSV_NAME = "gdsc_response_schema_summary.csv"
PER_DRUG_SUMMARY_CSV_NAME = "gdsc_per_drug_summary.csv"
SCHEMA_SUMMARY_MD_NAME = "gdsc_response_schema_summary.md"


# ----------------------------- LOGGING -------------------------------------- #

LOGGER = logging.getLogger("gdsc_response_inspector")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_info(message: str) -> None:
    LOGGER.info("[INFO] %s", message)


def log_warn(message: str) -> None:
    LOGGER.warning("[WARN] %s", message)


def log_error(message: str) -> None:
    LOGGER.error("[ERROR] %s", message)


# ----------------------------- VALIDATION ----------------------------------- #

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


def require_relative_dir(value: Any, key_path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Directory path at '{key_path}' must be a non-empty string.")
        raise SystemExit(1)

    rel_path = Path(value.strip())
    if rel_path.is_absolute():
        log_error(f"Directory path at '{key_path}' must be relative to REPO_ROOT: {rel_path}")
        raise SystemExit(1)

    resolved = (REPO_ROOT / rel_path).resolve()
    if not resolved.exists() or not resolved.is_dir():
        log_error(f"Resolved directory does not exist for '{key_path}': {resolved}")
        raise SystemExit(1)
    return resolved


def require_relative_filename(value: Any, key_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Filename at '{key_path}' must be a non-empty string.")
        raise SystemExit(1)

    filename = value.strip()
    path = Path(filename)
    if path.is_absolute():
        log_error(f"Filename at '{key_path}' must be relative, got absolute path: {filename}")
        raise SystemExit(1)

    if path.suffix.lower() not in {".xlsx", ".csv"}:
        log_error(
            f"File at '{key_path}' must be .xlsx or .csv, got: {filename}"
        )
        raise SystemExit(1)

    return filename


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

    config = require_mapping(parsed, "root")
    for top_key in ("project", "paths", "files", "schema"):
        require_key(config, top_key, "root")
    return config


def resolve_inputs_and_outputs(config: dict[str, Any]) -> tuple[Path, Path]:
    paths = require_mapping(require_key(config, "paths", "root"), "paths")
    files = require_mapping(require_key(config, "files", "root"), "files")

    raw = require_mapping(require_key(paths, "raw", "paths"), "paths.raw")
    reports = require_mapping(require_key(paths, "reports", "paths"), "paths.reports")
    gdsc_files = require_mapping(require_key(files, "gdsc", "files"), "files.gdsc")

    gdsc_dir = require_relative_dir(require_key(raw, "gdsc_dir", "paths.raw"), "paths.raw.gdsc_dir")
    response_filename = require_relative_filename(
        require_key(gdsc_files, "response", "files.gdsc"), "files.gdsc.response"
    )

    reports_tables_dir = require_relative_dir(
        require_key(reports, "tables_dir", "paths.reports"), "paths.reports.tables_dir"
    )

    response_path = (gdsc_dir / response_filename).resolve()
    if not response_path.exists() or not response_path.is_file():
        log_error(f"GDSC response file does not exist: {response_path}")
        raise SystemExit(1)

    return response_path, reports_tables_dir


# ----------------------------- INSPECTION ----------------------------------- #

def normalise_column_name(column_name: str) -> str:
    chars = []
    for char in column_name.lower():
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def detect_column_candidates(columns: list[str], aliases: list[str]) -> list[str]:
    normalised_aliases = [normalise_column_name(alias) for alias in aliases]
    matched: list[str] = []
    for column in columns:
        norm_col = normalise_column_name(column)
        if any(alias == norm_col or alias in norm_col or norm_col in alias for alias in normalised_aliases):
            matched.append(column)
    return matched


def resolve_single_column_or_none(label: str, candidates: list[str]) -> str | None:
    if not candidates:
        log_warn(f"No likely '{label}' column detected.")
        return None
    if len(candidates) > 1:
        joined = ", ".join(candidates)
        log_warn(f"Ambiguous '{label}' column candidates: {joined}")
        return None
    log_info(f"Detected '{label}' column: {candidates[0]}")
    return candidates[0]


def inspect_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | None], pd.Series]:
    if df.shape[1] == 0:
        log_error("Loaded DataFrame has zero columns; cannot inspect schema.")
        raise SystemExit(1)

    column_names = [str(col) for col in df.columns]
    missing_counts = df.isna().sum().sort_values(ascending=False)

    log_info(f"Rows: {df.shape[0]}")
    log_info(f"Columns: {df.shape[1]}")
    log_info("Full column names:")
    for name in column_names:
        log_info(f"  - {name}")

    log_info("Missing values per column (top 20):")
    for column_name, count in missing_counts.head(20).items():
        log_info(f"  - {column_name}: {int(count)}")

    schema_summary = pd.DataFrame(
        {
            "column_name": column_names,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": [int(df[col].notna().sum()) for col in df.columns],
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_proportion": [
                float(df[col].isna().mean()) if len(df) > 0 else 0.0 for col in df.columns
            ],
        }
    )

    if "DRUG_NAME" in column_names:
        drug_name_col = "DRUG_NAME"
    else:
        drug_name_col = resolve_single_column_or_none(
            "drug name",
            detect_column_candidates(
                column_names,
                ["DRUG_NAME", "Drug Name", "DRUG", "COMPOUND_NAME", "Compound", "drug"],
            ),
        )

    keys = {
        "drug_name": drug_name_col,
        "cell_line_name": resolve_single_column_or_none(
            "cell line name",
            detect_column_candidates(
                column_names,
                ["CELL_LINE_NAME", "Cell line name", "CELL_LINE", "CCLE_NAME", "cell line"],
            ),
        ),
        "cosmic_id": resolve_single_column_or_none(
            "COSMIC ID",
            detect_column_candidates(
                column_names,
                ["COSMIC_ID", "COSMIC ID", "COSMIC_IDENTIFIER", "COSMIC"],
            ),
        ),
        "ln_ic50": resolve_single_column_or_none(
            "LN_IC50",
            detect_column_candidates(
                column_names,
                ["LN_IC50", "ln_ic50", "LN IC50", "LNIC50", "LOG_IC50", "log ic50"],
            ),
        ),
        "auc": resolve_single_column_or_none(
            "AUC",
            detect_column_candidates(
                column_names,
                ["AUC", "area under curve", "AUC_VALUE"],
            ),
        ),
    }

    per_drug_summary = build_per_drug_summary(df, keys["drug_name"], keys["cell_line_name"], keys["ln_ic50"])
    return schema_summary, per_drug_summary, keys, missing_counts


def build_per_drug_summary(
    df: pd.DataFrame,
    drug_col: str | None,
    cell_line_col: str | None,
    ln_ic50_col: str | None,
) -> pd.DataFrame:
    output_columns = [
        "drug_name",
        "n_rows",
        "n_unique_cell_lines",
        "prop_missing_ln_ic50",
    ]

    if drug_col is None:
        log_warn("Per-drug summary cannot be grouped because drug name column is missing or ambiguous.")
        return pd.DataFrame(columns=output_columns)

    grouped = df.groupby(drug_col, dropna=False)
    summary = grouped.size().rename("n_rows").reset_index()
    summary = summary.rename(columns={drug_col: "drug_name"})

    if cell_line_col is not None:
        unique_cell_lines = (
            grouped[cell_line_col]
            .nunique(dropna=True)
            .rename("n_unique_cell_lines")
            .reset_index()
            .rename(columns={drug_col: "drug_name"})
        )
        summary = summary.merge(unique_cell_lines, on="drug_name", how="left")
    else:
        log_warn("Cell line column missing or ambiguous; n_unique_cell_lines will be NA.")
        summary["n_unique_cell_lines"] = pd.NA

    if ln_ic50_col is not None:
        missing_ln_ic50 = (
            grouped[ln_ic50_col]
            .apply(lambda series: float(series.isna().mean()))
            .rename("prop_missing_ln_ic50")
            .reset_index()
            .rename(columns={drug_col: "drug_name"})
        )
        summary = summary.merge(missing_ln_ic50, on="drug_name", how="left")
    else:
        log_warn("LN_IC50 column missing or ambiguous; prop_missing_ln_ic50 will be NA.")
        summary["prop_missing_ln_ic50"] = pd.NA

    return summary[output_columns].sort_values(by="n_rows", ascending=False).reset_index(drop=True)


# ----------------------------- REPORTING ------------------------------------ #

def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows to display._"
    return df.to_markdown(index=False)


def write_reports(
    reports_dir: Path,
    schema_summary: pd.DataFrame,
    per_drug_summary: pd.DataFrame,
    keys: dict[str, str | None],
    n_rows: int,
    n_columns: int,
    missing_counts: pd.Series,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    schema_csv_path = reports_dir / SCHEMA_SUMMARY_CSV_NAME
    per_drug_csv_path = reports_dir / PER_DRUG_SUMMARY_CSV_NAME
    schema_md_path = reports_dir / SCHEMA_SUMMARY_MD_NAME

    schema_summary.to_csv(schema_csv_path, index=False)
    per_drug_summary.to_csv(per_drug_csv_path, index=False)

    key_detection_df = pd.DataFrame(
        [
            {"field": "drug_name_column", "detected_column": keys["drug_name"]},
            {"field": "cell_line_name_column", "detected_column": keys["cell_line_name"]},
            {"field": "cosmic_id_column", "detected_column": keys["cosmic_id"]},
            {"field": "ln_ic50_column", "detected_column": keys["ln_ic50"]},
            {"field": "auc_column", "detected_column": keys["auc"]},
        ]
    )

    missing_top20_df = (
        missing_counts.head(20)
        .rename_axis("column_name")
        .reset_index(name="missing_count")
    )

    md_lines = [
        "# GDSC Response Schema Summary",
        "",
        "## Dataset Overview",
        f"- rows: {n_rows}",
        f"- columns: {n_columns}",
        "",
        "## Detected Key Columns",
        to_markdown_table(key_detection_df.fillna("NOT_DETECTED_OR_AMBIGUOUS")),
        "",
        "## Missing Values (Top 20 Columns)",
        to_markdown_table(missing_top20_df),
        "",
        "## Full Schema Table",
        to_markdown_table(schema_summary),
    ]
    schema_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    log_info(f"Wrote schema summary CSV: {schema_csv_path}")
    log_info(f"Wrote per-drug summary CSV: {per_drug_csv_path}")
    log_info(f"Wrote schema summary Markdown: {schema_md_path}")


# ----------------------------- ENTRYPOINT ----------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GDSC response Excel and write schema reports.")
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

    log_info("==================== GDSC RESPONSE INSPECTION ====================")
    log_info(f"REPO_ROOT: {REPO_ROOT}")
    log_info(f"Config path: {config_path}")

    config = load_config(config_path)
    response_path, reports_dir = resolve_inputs_and_outputs(config)

    log_info(f"Resolved GDSC response path: {response_path}")
    log_info(f"Resolved reports directory: {reports_dir}")

    try:
        if response_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(response_path, engine="openpyxl")
        elif response_path.suffix.lower() == ".csv":
            df = pd.read_csv(response_path)
        else:
            log_error(f"Unsupported file format: {response_path.suffix}")
            raise SystemExit(1)
        duplicate_row_count = int(df.duplicated().sum())
        if duplicate_row_count > 0:
            log_warn(f"Detected {duplicate_row_count} fully duplicated rows in the GDSC response table.")
    except ImportError as exc:
        log_error(f"openpyxl is required to read .xlsx files: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        log_error(f"Failed to load Excel file '{response_path}': {exc}")
        raise SystemExit(1) from exc

    schema_summary, per_drug_summary, keys, missing_counts = inspect_dataframe(df)
    write_reports(
        reports_dir=reports_dir,
        schema_summary=schema_summary,
        per_drug_summary=per_drug_summary,
        keys=keys,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        missing_counts=missing_counts,
    )

    log_info("==================== INSPECTION STATUS ====================")
    log_info("GDSC response inspection completed successfully.")


if __name__ == "__main__":
    main()
