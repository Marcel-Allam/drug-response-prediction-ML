"""00_validate_inputs.py

Purpose
-------
Fail fast if the user has not placed the required DepMap files into `data/raw/`.

Why this script exists
----------------------
Public -omics datasets are large and often change filenames across releases.
A common failure mode in portfolio repos is: "It doesn't run for anyone else".
This script makes the required inputs explicit and stops early with actionable errors.

Expected inputs
---------------
Place these files in `data/raw/` (exact filenames expected below):

1) DepMap expression matrix (cell lines × genes)
   - Example filename: `CCLE_expression.csv` or `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
   - You may choose a different DepMap expression file, but then update EXPECTED_FILES.

2) PRISM drug response matrix (cell lines × compounds)
   - Example filename: `secondary-screen-dose-response-curve-parameters.csv`
   - Or one of the Repurposing_*_Data_Matrix CSVs depending on the release.

After you download, rename files to the expected names to keep the workflow stable.

"""

from __future__ import annotations

import sys
from pathlib import Path

# -----------------------------
# Configuration (edit as needed)
# -----------------------------

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# These are the canonical names this repo expects. If your DepMap release uses different names,
# rename your downloaded files OR update this list to match your filenames.
EXPECTED_FILES = [
    "depmap_expression.csv",
    "depmap_prism_response.csv",
]

def main() -> None:
    """Validate that required input files exist."""
    missing = [fname for fname in EXPECTED_FILES if not (RAW_DIR / fname).exists()]

    if missing:
        print("\n[ERROR] Missing required input files in:", RAW_DIR)
        for fname in missing:
            print("  -", fname)

        print("\nFix:")
        print("  1) Download DepMap expression + PRISM drug response from the DepMap portal.")
        print("  2) Put them in data/raw/")
        print("  3) Rename them to match the expected filenames above (or edit EXPECTED_FILES).\n")
        sys.exit(1)

    print("[OK] All required input files found in:", RAW_DIR)

if __name__ == "__main__":
    main()
