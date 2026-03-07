#!/usr/bin/env python3
"""
03_prepare_single_drug_dataset.py

Purpose
-------
Build a clean per-drug modelling dataset by merging GDSC response data with
DepMap expression data.

Inputs
------
- config/default_config.yaml
- paths.raw.gdsc_dir + files.gdsc.response
- paths.raw.depmap_dir + files.depmap.expression

Outputs
-------
- paths.processed.per_drug_dir/<drug_name>.parquet

Usage
-----
python scripts/03_prepare_single_drug_dataset.py --drug "Erlotinib"
python scripts/03_prepare_single_drug_dataset.py --config config/default_config.yaml --drug "Erlotinib"

Notes
-----
- This script prepares data only; it does not train models.
- All configured paths are resolved relative to REPO_ROOT.
- Missing required files/columns cause an immediate exit.
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


# ----------------------------- LOGGING -------------------------------------- #

LOGGER = logging.getLogger("single_drug_dataset_preparer")


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


def require_non_empty_string(value: Any, key_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        log_error(f"Value at '{key_path}' must be a non-empty string.")
        raise SystemExit(1)
    return value.strip()


def resolve_relative_dir(value: Any, key_path: str) -> Path:
    rel = Path(require_non_empty_string(value, key_path))
    if rel.is_absolute():
        log_error(f"Path at '{key_path}' must be relative to REPO_ROOT: {rel}")
        raise SystemExit(1)
    return (REPO_ROOT / rel).resolve()


def resolve_relative_file(dir_path: Path, filename_value: Any, key_path: str) -> Path:
    filename = require_non_empty_string(filename_value, key_path)
    rel = Path(filename)
    if rel.is_absolute():
        log_error(f"Filename at '{key_path}' must be relative, got absolute path: {filename}")
        raise SystemExit(1)
    resolved = (dir_path / rel).resolve()
    if not resolved.exists() or not resolved.is_file():
        log_error(f"Required file does not exist: {resolved}")
        raise SystemExit(1)
    return resolved


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


def resolve_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    paths = require_mapping(require_key(config, "paths", "root"), "paths")
    files = require_mapping(require_key(config, "files", "root"), "files")

    raw = require_mapping(require_key(paths, "raw", "paths"), "paths.raw")
    processed = require_mapping(require_key(paths, "processed", "paths"), "paths.processed")
    gdsc_files = require_mapping(require_key(files, "gdsc", "files"), "files.gdsc")
    depmap_files = require_mapping(require_key(files, "depmap", "files"), "files.depmap")

    gdsc_dir = resolve_relative_dir(require_key(raw, "gdsc_dir", "paths.raw"), "paths.raw.gdsc_dir")
    depmap_dir = resolve_relative_dir(require_key(raw, "depmap_dir", "paths.raw"), "paths.raw.depmap_dir")
    per_drug_dir = resolve_relative_dir(
        require_key(processed, "per_drug_dir", "paths.processed"),
        "paths.processed.per_drug_dir",
    )

    if not gdsc_dir.exists() or not gdsc_dir.is_dir():
        log_error(f"Configured GDSC directory does not exist: {gdsc_dir}")
        raise SystemExit(1)
    if not depmap_dir.exists() or not depmap_dir.is_dir():
        log_error(f"Configured DepMap directory does not exist: {depmap_dir}")
        raise SystemExit(1)

    gdsc_path = resolve_relative_file(
        gdsc_dir,
        require_key(gdsc_files, "response", "files.gdsc"),
        "files.gdsc.response",
    )
    depmap_path = resolve_relative_file(
        depmap_dir,
        require_key(depmap_files, "expression", "files.depmap"),
        "files.depmap.expression",
    )
    model_metadata_path = resolve_relative_file(
        depmap_dir,
        require_key(depmap_files, "model_metadata", "files.depmap"),
        "files.depmap.model_metadata",
    )

    return gdsc_path, depmap_path, model_metadata_path, per_drug_dir


# ----------------------------- DATA LOADING --------------------------------- #

def normalise_column_name(column_name: str) -> str:
    chars: list[str] = []
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


def resolve_required_column(label: str, candidates: list[str]) -> str:
    if not candidates:
        log_error(f"No likely '{label}' column detected.")
        raise SystemExit(1)
    if len(candidates) > 1:
        joined = ", ".join(candidates)
        log_error(f"Ambiguous '{label}' column candidates: {joined}")
        raise SystemExit(1)
    log_info(f"Detected '{label}' column: {candidates[0]}")
    return candidates[0]


def load_gdsc_response(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        if suffix == ".csv":
            log_warn("GDSC response file is CSV (not XLSX); loading via pandas.read_csv.")
            return pd.read_csv(path)
    except ImportError as exc:
        log_error(f"Missing required reader dependency for '{path}': {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        log_error(f"Failed to load GDSC response file '{path}': {exc}")
        raise SystemExit(1) from exc

    log_error(f"Unsupported GDSC response format '{suffix}'. Expected .xlsx or .csv.")
    raise SystemExit(1)


def load_depmap_expression(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        log_error(f"DepMap expression file must be a .csv file, got: {path.name}")
        raise SystemExit(1)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log_error(f"Failed to load DepMap expression file '{path}': {exc}")
        raise SystemExit(1) from exc


def load_model_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        log_error(f"DepMap model metadata file must be a .csv file, got: {path.name}")
        raise SystemExit(1)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log_error(f"Failed to load DepMap model metadata file '{path}': {exc}")
        raise SystemExit(1) from exc


def detect_gdsc_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    gdsc_columns = [str(col) for col in df.columns]

    if "DRUG_NAME" in gdsc_columns:
        drug_name_col = "DRUG_NAME"
    else:
        drug_name_col = resolve_required_column(
            "drug name",
            detect_column_candidates(
                gdsc_columns,
                ["DRUG_NAME", "Drug Name", "DRUG", "COMPOUND_NAME", "Compound", "drug"],
            ),
        )
    if "COSMIC_ID" in gdsc_columns:
        cell_line_id_col = "COSMIC_ID"
    else:
        cell_line_id_col = resolve_required_column(
            "cell line identifier",
            detect_column_candidates(
                gdsc_columns,
                ["COSMIC_ID", "COSMIC ID", "CELL_LINE_NAME", "Cell line name", "SANGER_MODEL_ID"],
            ),
        )
    ln_ic50_col = resolve_required_column(
        "LN_IC50",
        detect_column_candidates(
            gdsc_columns,
            ["LN_IC50", "ln_ic50", "LN IC50", "LNIC50", "LOG_IC50", "log ic50"],
        ),
    )

    return drug_name_col, cell_line_id_col, ln_ic50_col


def detect_depmap_id_column(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    return resolve_required_column(
        "DepMap model identifier",
        detect_column_candidates(
            columns,
            ["ModelID", "MODEL_ID", "DepMap_ID", "DEPMAP_ID", "CCLE_NAME", "CELL_LINE_NAME"],
        ),
    )


def detect_model_metadata_columns(df: pd.DataFrame) -> tuple[str, str]:
    model_columns = [str(col) for col in df.columns]
    if "ModelID" in model_columns:
        depmap_model_id_col = "ModelID"
    else:
        depmap_model_id_col = resolve_required_column(
            "DepMap model identifier",
            detect_column_candidates(model_columns, ["ModelID", "SangerModelID", "ModelIDAlias"]),
        )
    cosmic_id_col = resolve_required_column(
        "COSMIC identifier",
        detect_column_candidates(model_columns, ["COSMIC_ID", "COSMICID"]),
    )
    return depmap_model_id_col, cosmic_id_col


def standardise_join_key(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )


def standardize_cosmic_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA, "<NA>": pd.NA})
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    return cleaned


# ----------------------------- DATA MERGE ----------------------------------- #

def build_single_drug_dataset(
    gdsc_df: pd.DataFrame,
    depmap_df: pd.DataFrame,
    model_df: pd.DataFrame,
    drug_name: str,
    gdsc_drug_col: str,
    gdsc_cell_col: str,
    gdsc_ln_ic50_col: str,
    depmap_id_col: str,
    model_id_col: str,
    model_cosmic_id_col: str,
) -> pd.DataFrame:
    if gdsc_df.empty:
        log_error("GDSC response table is empty.")
        raise SystemExit(1)
    if depmap_df.empty:
        log_error("DepMap expression matrix is empty.")
        raise SystemExit(1)

    mask = gdsc_df[gdsc_drug_col].astype("string").str.strip().str.casefold() == drug_name.strip().casefold()
    gdsc_filtered = gdsc_df.loc[mask].copy()
    if gdsc_filtered.empty:
        log_error(f"No rows found in GDSC response table for drug '{drug_name}'.")
        raise SystemExit(1)
    log_info(f"GDSC drug subset rows: {len(gdsc_filtered)}")

    depmap_with_model = depmap_df.merge(
        model_df[[model_id_col, model_cosmic_id_col]],
        how="left",
        left_on=depmap_id_col,
        right_on=model_id_col,
    )
    log_info(f"DepMap expression rows after adding model metadata: {len(depmap_with_model)}")

    depmap_work = depmap_with_model.copy()
    depmap_work["COSMIC_ID"] = depmap_work[model_cosmic_id_col]

    gdsc_filtered["COSMIC_ID"] = standardize_cosmic_id(gdsc_filtered[gdsc_cell_col])
    depmap_work["COSMIC_ID"] = standardize_cosmic_id(depmap_work["COSMIC_ID"])

    gdsc_filtered = gdsc_filtered[gdsc_filtered["COSMIC_ID"].notna() & (gdsc_filtered["COSMIC_ID"] != "")]
    depmap_work = depmap_work[depmap_work["COSMIC_ID"].notna() & (depmap_work["COSMIC_ID"] != "")]
    log_info(f"GDSC rows remaining after dropping missing COSMIC_ID: {len(gdsc_filtered)}")
    log_info(f"DepMap rows remaining after dropping missing COSMIC_ID: {len(depmap_work)}")
    if gdsc_filtered.empty:
        log_error("No valid GDSC cell line identifiers remain after cleaning.")
        raise SystemExit(1)
    if depmap_work.empty:
        log_error("No valid DepMap model identifiers remain after cleaning.")
        raise SystemExit(1)

    gdsc_unique_ids = set(gdsc_filtered["COSMIC_ID"].dropna().astype(str).unique().tolist())
    depmap_unique_ids = set(depmap_work["COSMIC_ID"].dropna().astype(str).unique().tolist())
    overlap_ids = gdsc_unique_ids & depmap_unique_ids
    log_info(f"Unique COSMIC_ID in GDSC drug subset: {len(gdsc_unique_ids)}")
    log_info(f"Unique COSMIC_ID in DepMap merged expression table: {len(depmap_unique_ids)}")
    log_info(f"COSMIC_ID intersection size: {len(overlap_ids)}")
    log_info(f"GDSC COSMIC_ID sample (first 10): {sorted(gdsc_unique_ids)[:10]}")
    log_info(f"DepMap COSMIC_ID sample (first 10): {sorted(depmap_unique_ids)[:10]}")
    log_info(f"Overlapping COSMIC_ID sample (first 10): {sorted(overlap_ids)[:10]}")

    merged = gdsc_filtered.merge(depmap_work, how="inner", on="COSMIC_ID", suffixes=("_gdsc", "_depmap"))
    log_info(f"Rows after final merge: {len(merged)}")
    if merged.empty:
        log_error("Merge produced zero rows. Check cell line identifier compatibility between GDSC and DepMap.")
        raise SystemExit(1)

    merged["__ln_ic50_numeric"] = pd.to_numeric(merged[gdsc_ln_ic50_col], errors="coerce")
    missing_ln_count = int(merged["__ln_ic50_numeric"].isna().sum())
    if missing_ln_count > 0:
        log_warn(f"Dropping {missing_ln_count} rows with missing/non-numeric LN_IC50.")
    merged = merged[merged["__ln_ic50_numeric"].notna()].copy()
    if merged.empty:
        log_error("No rows remain after removing missing LN_IC50 values.")
        raise SystemExit(1)

    feature_columns = [col for col in depmap_df.columns if col != depmap_id_col]
    if not feature_columns:
        log_error("DepMap expression matrix has no feature columns after removing identifier column.")
        raise SystemExit(1)

    overlap = {"drug_name", "cell_line_id", "depmap_model_id", "LN_IC50"} & set(feature_columns)
    if overlap:
        joined = ", ".join(sorted(overlap))
        log_error(f"Feature columns collide with reserved output columns: {joined}")
        raise SystemExit(1)

    output_df = pd.DataFrame(
        {
            "drug_name": merged[gdsc_drug_col].astype("string"),
            "cell_line_id": merged[gdsc_cell_col].astype("string"),
            "depmap_model_id": merged[depmap_id_col].astype("string"),
            "LN_IC50": merged["__ln_ic50_numeric"].astype(float),
        }
    )
    output_df = pd.concat([output_df, merged[feature_columns].reset_index(drop=True)], axis=1)

    x_df = output_df[feature_columns]
    y_series = output_df["LN_IC50"]
    if len(x_df) != len(y_series):
        log_error("Feature matrix and target vector length mismatch after merge.")
        raise SystemExit(1)
    if y_series.isna().any():
        log_error("Target vector LN_IC50 contains missing values after filtering.")
        raise SystemExit(1)
    if not x_df.index.equals(y_series.index):
        log_error("Feature matrix and target vector index misalignment detected.")
        raise SystemExit(1)

    return output_df


# ----------------------------- OUTPUT --------------------------------------- #

def safe_drug_filename(drug_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", drug_name.strip()).strip("_").lower()
    return cleaned if cleaned else "drug"


def save_dataset(df: pd.DataFrame, out_dir: Path, drug_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_drug_filename(drug_name)}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except ImportError as exc:
        log_error(f"Parquet support is unavailable (install pyarrow or fastparquet): {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        log_error(f"Failed to write parquet output '{out_path}': {exc}")
        raise SystemExit(1) from exc
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a per-drug modelling dataset by merging GDSC response with DepMap expression."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--drug",
        type=str,
        required=True,
        help='Drug name to extract (example: --drug "Erlotinib")',
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

    log_info("==================== PREPARE SINGLE DRUG DATASET ====================")
    log_info(f"REPO_ROOT: {REPO_ROOT}")
    log_info(f"Config path: {config_path}")
    log_info(f"Requested drug: {args.drug}")

    config = load_config(config_path)
    gdsc_path, depmap_path, model_metadata_path, per_drug_dir = resolve_paths(config)

    log_info(f"Resolved GDSC response path: {gdsc_path}")
    log_info(f"Resolved DepMap expression path: {depmap_path}")
    log_info(f"Resolved DepMap model metadata path: {model_metadata_path}")
    log_info(f"Resolved output directory: {per_drug_dir}")

    gdsc_df = load_gdsc_response(gdsc_path)
    depmap_df = load_depmap_expression(depmap_path)
    model_df = load_model_metadata(model_metadata_path)

    log_info(f"GDSC shape: {gdsc_df.shape}")
    log_info(f"DepMap shape: {depmap_df.shape}")
    log_info(f"Model metadata shape: {model_df.shape}")

    gdsc_drug_col, gdsc_cell_col, gdsc_ln_ic50_col = detect_gdsc_columns(gdsc_df)
    depmap_id_col = detect_depmap_id_column(depmap_df)
    model_id_col, model_cosmic_id_col = detect_model_metadata_columns(model_df)

    output_df = build_single_drug_dataset(
        gdsc_df=gdsc_df,
        depmap_df=depmap_df,
        model_df=model_df,
        drug_name=args.drug,
        gdsc_drug_col=gdsc_drug_col,
        gdsc_cell_col=gdsc_cell_col,
        gdsc_ln_ic50_col=gdsc_ln_ic50_col,
        depmap_id_col=depmap_id_col,
        model_id_col=model_id_col,
        model_cosmic_id_col=model_cosmic_id_col,
    )

    out_path = save_dataset(output_df, per_drug_dir, args.drug)
    log_info(f"Final dataset rows: {len(output_df)}")
    log_info(f"Final dataset columns: {output_df.shape[1]}")
    log_info(f"Wrote per-drug dataset: {out_path}")
    log_info("==================== STATUS: SUCCESS ====================")


if __name__ == "__main__":
    main()
