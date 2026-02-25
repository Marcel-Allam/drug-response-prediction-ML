"""03_evaluate_and_interpret.py

Purpose
-------
1) Summarize model performance across compounds.
2) Extract top predictive genes (coefficients) for a selected compound.
3) Save lightweight plots for the README.

Notes
-----
- This script performs *simple* interpretation for Elastic Net models.
- For more robust interpretation, consider SHAP (already included in environment.yml),
  but coefficient-based interpretation is a good transparent starting point.

"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
METRIC_FILE = BASE_DIR / "results" / "metrics" / "elasticnet_metrics.json"
MODEL_DIR = BASE_DIR / "results" / "models"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    # -----------------------------
    # Load metrics and plot distribution
    # -----------------------------
    with open(METRIC_FILE, "r") as f:
        metrics = json.load(f)

    if not metrics:
        raise RuntimeError("No metrics found. Did you run 02_train_elasticnet.py successfully?")

    df = pd.DataFrame.from_dict(metrics, orient="index").reset_index().rename(columns={"index": "compound"})

    # Plot R^2 across compounds
    plt.figure()
    df["r2_in_sample"].hist(bins=30)
    plt.xlabel("R^2 (in-sample, baseline)")
    plt.ylabel("Number of compounds")
    plt.title("Elastic Net baseline performance distribution")
    out1 = FIG_DIR / "elasticnet_r2_hist.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()

    # Pick the best compound by R^2 for coefficient inspection
    best = df.sort_values("r2_in_sample", ascending=False).iloc[0]["compound"]
    bundle = joblib.load(MODEL_DIR / f"elasticnet_{best}.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # ElasticNet coefficients live in the enet step (after scaling).
    coefs = model.named_steps["enet"].coef_
    coefs = np.asarray(coefs)

    # Top absolute coefficients
    top_k = 30
    idx = np.argsort(np.abs(coefs))[::-1][:top_k]
    top = pd.DataFrame({
        "gene": [feature_names[i] for i in idx],
        "coef": coefs[idx],
        "abs_coef": np.abs(coefs[idx]),
    }).sort_values("abs_coef", ascending=True)

    # Bar plot
    plt.figure(figsize=(6, 8))
    plt.barh(top["gene"], top["coef"])
    plt.xlabel("Coefficient (scaled features)")
    plt.title(f"Top {top_k} Elastic Net genes\nCompound: {best}")
    out2 = FIG_DIR / "elasticnet_top_genes.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()

    top.to_csv(FIG_DIR / "elasticnet_top_genes.csv", index=False)

    print("[OK] Saved figures:")
    print(" -", out1)
    print(" -", out2)
    print("[OK] Best compound (by in-sample R^2):", best)

if __name__ == "__main__":
    main()
