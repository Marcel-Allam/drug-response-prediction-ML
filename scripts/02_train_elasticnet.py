"""02_train_elasticnet.py

Purpose
-------
Train an **interpretable baseline** model for each compound:
- Elastic Net regression (L1 + L2)
- Cross-validated hyperparameters (alpha, l1_ratio)

Why Elastic Net first?
----------------------
In high-dimensional -omics (p >> n), linear models with regularization are:
- strong baselines
- relatively resistant to overfitting
- interpretable (non-zero coefficients suggest candidate biomarkers)

Approach
--------
We fit a separate model per compound (single-task).
Later, you can extend to:
- multi-task learning
- shared embeddings
- neural nets / transformers

Outputs
-------
- Saved sklearn models in `results/models/`
- Per-compound metrics in `results/metrics/elasticnet_metrics.json`

"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "results" / "models"
METRIC_DIR = BASE_DIR / "results" / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    # -----------------------------
    # Load processed data
    # -----------------------------
    X = pd.read_parquet(DATA_DIR / "X_expression.parquet")
    y = pd.read_parquet(DATA_DIR / "y_drug_response.parquet")

    # Convert to numpy once for speed (keep feature names separately for interpretability)
    X_values = X.values
    feature_names = list(X.columns)

    # -----------------------------
    # CV strategy
    # -----------------------------
    # Simple KFold across cell lines.
    # If you later incorporate replicate measurements or multiple screens per cell line,
    # switch to GroupKFold to avoid leakage.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {}

    # Train a separate Elastic Net model per compound (column in y)
    for compound in y.columns:
        y_vec = y[compound].values.astype(float)

        # Skip compounds with too many missing values
        valid_mask = np.isfinite(y_vec)
        if valid_mask.mean() < 0.8:
            continue

        Xv = X_values[valid_mask]
        yv = y_vec[valid_mask]

        # Pipeline: scale features -> ElasticNetCV
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("enet", ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
                    alphas=np.logspace(-4, 1, 30),
                    cv=cv,
                    max_iter=5000,
                    n_jobs=-1,
                    random_state=42
                )),
            ]
        )

        model.fit(Xv, yv)

        # Evaluate in-sample (baseline). Later you should add a true held-out test split.
        preds = model.predict(Xv)
        rmse = float(np.sqrt(mean_squared_error(yv, preds)))
        r2 = float(r2_score(yv, preds))

        metrics[compound] = {
            "n_samples": int(len(yv)),
            "rmse_in_sample": rmse,
            "r2_in_sample": r2,
            "best_alpha": float(model.named_steps["enet"].alpha_),
            "best_l1_ratio": float(model.named_steps["enet"].l1_ratio_),
        }

        # Save model
        joblib.dump(
            {"model": model, "feature_names": feature_names, "compound": compound},
            MODEL_DIR / f"elasticnet_{compound}.joblib",
        )

    with open(METRIC_DIR / "elasticnet_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Trained {len(metrics)} compound models.")
    print("Saved metrics:", METRIC_DIR / "elasticnet_metrics.json")

if __name__ == "__main__":
    main()
