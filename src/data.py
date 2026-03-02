"""
src/data.py

Purpose
-------
Strict data loading and alignment utilities for cross-dataset drug response prediction
(GDSC -> DepMap). This module intentionally avoids any random train/test splitting to
prevent leakage in final evaluation workflows.

Design principles
-----------------
- Fail fast on invalid inputs (missing files, empty tables, mismatched IDs).
- No silent reordering except when explicitly aligning by identifiers.
- No assumptions about response table schema (wide vs long).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Minimum overlap to consider harmonised expression usable. Adjust later if needed.
MIN_SHARED_GENES = 5000


# ----------------------------- DATA LOADING -------------------------------- #

def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    """
    Load a gene-expression matrix with strict input validation.

    Why this exists:
    Expression features are the core model input for both training (GDSC) and
    external testing (DepMap). A dedicated loader enforces consistent file
    handling, fails fast on invalid paths/types, and avoids silent changes that
    could break downstream cell-line alignment.
    """
    matrix = _load_table(path, dataset_kind="expression")
    if matrix.empty:
        msg = f"Expression matrix is empty: {Path(path)}"
        LOGGER.error(msg)
        raise ValueError(msg)
    LOGGER.info("Loaded expression matrix: %s (shape=%s)", Path(path), matrix.shape)
    return matrix


def load_response_matrix(path: str | Path) -> pd.DataFrame:
    """
    Load a drug-response table without forcing a schema.

    Why this exists:
    Response data may arrive in either wide or long format depending on source
    and preprocessing stage. This loader validates existence and non-empty
    content while avoiding assumptions about layout; interpretation should be
    handled in higher-level, config-driven code.
    """
    matrix = _load_table(path, dataset_kind="response")
    if matrix.empty:
        msg = f"Response table is empty: {Path(path)}"
        LOGGER.error(msg)
        raise ValueError(msg)
    LOGGER.info("Loaded response table: %s (shape=%s)", Path(path), matrix.shape)
    return matrix


def _load_table(path: str | Path, dataset_kind: str) -> pd.DataFrame:
    """Read CSV/TSV/Parquet from disk with explicit error reporting."""
    table_path = Path(path)

    if not table_path.exists():
        msg = f"{dataset_kind.capitalize()} file does not exist: {table_path}"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)
    if not table_path.is_file():
        msg = f"{dataset_kind.capitalize()} path is not a file: {table_path}"
        LOGGER.error(msg)
        raise ValueError(msg)

    suffix = table_path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(table_path)
        if suffix == ".tsv":
            return pd.read_csv(table_path, sep="\t")
        if suffix == ".parquet":
            return pd.read_parquet(table_path)
    except Exception as exc:
        msg = f"Failed to read {dataset_kind} file '{table_path}': {exc}"
        LOGGER.error(msg)
        raise ValueError(msg) from exc

    msg = (
        f"Unsupported file type for {dataset_kind}: '{suffix}'. "
        "Supported: .csv, .tsv, .parquet"
    )
    LOGGER.error(msg)
    raise ValueError(msg)


# ----------------------------- ALIGNMENT ----------------------------------- #

def intersect_genes(
    X_gdsc: pd.DataFrame,
    X_depmap: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute strict shared-gene alignment across GDSC and DepMap matrices.

    Why this exists:
    Cross-dataset evaluation is only valid when train and test features represent
    the same genes. This function creates a deterministic, leakage-safe feature
    intersection and fails fast if overlap is too small.
    """
    if X_gdsc.empty:
        msg = "X_gdsc is empty; cannot compute gene intersection."
        LOGGER.error(msg)
        raise ValueError(msg)
    if X_depmap.empty:
        msg = "X_depmap is empty; cannot compute gene intersection."
        LOGGER.error(msg)
        raise ValueError(msg)

    depmap_genes = set(X_depmap.columns)
    shared_genes = [g for g in X_gdsc.columns if g in depmap_genes]
    shared_gene_count = len(shared_genes)

    if shared_gene_count < MIN_SHARED_GENES:
        msg = (
            f"Shared gene intersection too small: {shared_gene_count} "
            f"(minimum required: {MIN_SHARED_GENES})."
        )
        LOGGER.error(msg)
        raise ValueError(msg)

    X_gdsc_aligned = X_gdsc.loc[:, shared_genes]
    X_depmap_aligned = X_depmap.loc[:, shared_genes]

    LOGGER.info("Intersected genes across datasets: %d shared features retained.", shared_gene_count)
    return X_gdsc_aligned, X_depmap_aligned, shared_genes


def align_cell_lines(
    X: pd.DataFrame,
    y: pd.Series,
    id_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features and target to the same cell-line identifier order.

    Contract:
    - `X` must contain `id_column` as a column OR already be indexed by it.
    - `y.index` must contain the same set of identifiers as X.
    - Raises on missing IDs, duplicates, or set mismatch.

    Why this exists:
    In cross-dataset pipelines, accidental row misalignment between features and
    responses causes invalid metrics. This function makes alignment explicit and
    fail-fast.
    """
    if X.empty:
        msg = "X is empty; cannot align cell lines."
        LOGGER.error(msg)
        raise ValueError(msg)
    if y.empty:
        msg = "y is empty; cannot align cell lines."
        LOGGER.error(msg)
        raise ValueError(msg)

    # Support either an ID column or an ID index
    if id_column in X.columns:
        x_ids = X[id_column]
        if x_ids.isna().any():
            msg = f"X contains missing values in id_column '{id_column}'."
            LOGGER.error(msg)
            raise ValueError(msg)
        if x_ids.duplicated().any():
            msg = f"X contains duplicate cell-line IDs in '{id_column}'."
            LOGGER.error(msg)
            raise ValueError(msg)
        X_indexed = X.set_index(id_column, drop=False)
    else:
        # Assume already indexed by IDs
        X_indexed = X.copy()
        x_ids = X_indexed.index
        if X_indexed.index.hasnans:
            msg = "X index contains missing cell-line IDs."
            LOGGER.error(msg)
            raise ValueError(msg)
        if X_indexed.index.duplicated().any():
            msg = "X index contains duplicate cell-line IDs."
            LOGGER.error(msg)
            raise ValueError(msg)

    if y.index.hasnans:
        msg = "y index contains missing cell-line IDs."
        LOGGER.error(msg)
        raise ValueError(msg)
    if y.index.duplicated().any():
        msg = "y index contains duplicate cell-line IDs."
        LOGGER.error(msg)
        raise ValueError(msg)

    x_id_set = set(list(x_ids))
    y_id_set = set(list(y.index))
    if x_id_set != y_id_set:
        missing_in_y = [cid for cid in list(x_ids) if cid not in y_id_set][:5]
        missing_in_x = [cid for cid in list(y.index) if cid not in x_id_set][:5]
        msg = (
            "Cell-line ID mismatch between X and y. "
            f"Missing in y (sample): {missing_in_y}; missing in X (sample): {missing_in_x}"
        )
        LOGGER.error(msg)
        raise ValueError(msg)

    y_aligned = y.reindex(X_indexed.index)
    if not X_indexed.index.equals(y_aligned.index):
        msg = "Failed to align X and y to identical index order."
        LOGGER.error(msg)
        raise ValueError(msg)

    LOGGER.info("Aligned cell lines using '%s': %d matched rows.", id_column, len(X_indexed))
    return X_indexed, y_aligned
