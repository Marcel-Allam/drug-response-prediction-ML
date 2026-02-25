"""01_build_feature_matrix.py

Purpose
-------
Build a clean, aligned machine-learning dataset:
- X: gene expression features (cell lines × genes)
- y: drug response endpoint (cell lines × compounds)

Key design choices
------------------
1) **Join key**: we use a cell line identifier present in BOTH tables.
   DepMap often provides `DepMap_ID` and `CCLE_Name`. Pick one and be consistent.

2) **Leakage control**:
   - We only use **baseline** expression (no post-treatment signals).
   - We build splits later with grouping to avoid duplicated cell line info leakage.

3) **Minimal preprocessing**:
   - Keep it transparent and reproducible.
   - Standardization happens inside the model pipeline (Elastic Net).

Input expectations
------------------
`data/raw/depmap_expression.csv`
`data/raw/depmap_prism_response.csv`

Because DepMap releases vary, you may need to tweak:
- which columns contain identifiers
- the naming convention for compounds
- whether response is AUC, viability, or log-fold-change

This script is written to be easy to adapt: edit the CONFIG section.

"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# -----------------------------
# Configuration (edit as needed)
# -----------------------------
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPRESSION_FILE = RAW_DIR / "depmap_expression.csv"
RESPONSE_FILE = RAW_DIR / "depmap_prism_response.csv"

# Column names vary by DepMap file. Update these to match your downloads.
CELL_LINE_ID_COL_EXPRESSION = "DepMap_ID"
CELL_LINE_ID_COL_RESPONSE = "DepMap_ID"

# For response tables, you may have "wide" format (one column per compound)
# or "long" format (rows: cell_line, compound, response).
# This script assumes WIDE format for simplicity.
def main() -> None:
    # -----------------------------
    # Load expression
    # -----------------------------
    expr = pd.read_csv(EXPRESSION_FILE)
    if CELL_LINE_ID_COL_EXPRESSION not in expr.columns:
        raise ValueError(
            f"Expression file missing expected id column '{CELL_LINE_ID_COL_EXPRESSION}'. "
            f"Columns found: {list(expr.columns)[:20]} ..."
        )

    # Set index to cell line id and keep only numeric gene columns
    expr = expr.set_index(CELL_LINE_ID_COL_EXPRESSION)
    expr_numeric = expr.select_dtypes(include=["number"]).copy()

    # -----------------------------
    # Load response
    # -----------------------------
    resp = pd.read_csv(RESPONSE_FILE)
    if CELL_LINE_ID_COL_RESPONSE not in resp.columns:
        raise ValueError(
            f"Response file missing expected id column '{CELL_LINE_ID_COL_RESPONSE}'. "
            f"Columns found: {list(resp.columns)[:20]} ..."
        )

    resp = resp.set_index(CELL_LINE_ID_COL_RESPONSE)
    resp_numeric = resp.select_dtypes(include=["number"]).copy()

    # -----------------------------
    # Align samples (intersection)
    # -----------------------------
    common_ids = expr_numeric.index.intersection(resp_numeric.index)
    if len(common_ids) < 50:
        raise ValueError(
            f"Too few overlapping cell lines between expression and response: {len(common_ids)}. "
            "Check your identifier columns and file choices."
        )

    X = expr_numeric.loc[common_ids].sort_index()
    y = resp_numeric.loc[common_ids].sort_index()

    # -----------------------------
    # Save processed matrices
    # -----------------------------
    X.to_parquet(OUT_DIR / "X_expression.parquet")
    y.to_parquet(OUT_DIR / "y_drug_response.parquet")

    print("[OK] Saved:")
    print(" -", OUT_DIR / "X_expression.parquet", X.shape)
    print(" -", OUT_DIR / "y_drug_response.parquet", y.shape)

if __name__ == "__main__":
    main()
