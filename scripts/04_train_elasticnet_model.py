#!/usr/bin/env python3
"""
04_train_elasticnet_model.py

Train an ElasticNet regression model to predict LN_IC50 from a per-drug dataset
using only gene expression features.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------- CONFIGURATION -------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.yaml"
LOGGER = logging.getLogger("elasticnet_trainer")


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
        description="Train ElasticNet model for a single drug using gene expression features."
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
    return parser.parse_args()


def slugify_drug_name(drug_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", drug_name.strip().lower())
    return slug.strip("_")


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        log_error(f"Expected mapping at '{context}', got {type(value).__name__}.")
        raise SystemExit(1)
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists() or not config_path.is_file():
        log_error(f"Config file does not exist: {config_path}")
        raise SystemExit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
    except Exception as exc:
        log_error(f"Failed to parse YAML config '{config_path}': {exc}")
        raise SystemExit(1) from exc

    return require_mapping(parsed, "root")


def get_project_seed(config: dict[str, Any]) -> int:
    project = require_mapping(config.get("project"), "project")
    seed = project.get("seed")
    if seed is None:
        log_error("Missing required config value: project.seed")
        raise SystemExit(1)
    if not isinstance(seed, int):
        log_error(f"Config value project.seed must be an integer, got {type(seed).__name__}.")
        raise SystemExit(1)
    return seed


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


# ----------------------------- VALIDATION ----------------------------------- #

def validate_inputs(df: pd.DataFrame, gene_cols: list[str]) -> None:
    if "LN_IC50" not in df.columns:
        log_error("Required target column 'LN_IC50' was not found in dataset.")
        raise SystemExit(1)

    if not gene_cols:
        log_error("No gene expression feature columns found using '(...)' pattern.")
        raise SystemExit(1)

    X = df[gene_cols]
    y = df["LN_IC50"]

    if X.isna().any().any():
        log_error("Feature matrix X contains missing values.")
        raise SystemExit(1)

    if y.isna().any():
        log_error("Target vector y contains missing values.")
        raise SystemExit(1)


# ----------------------------- DATA LOADING --------------------------------- #

def load_dataset(input_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(input_path)
    except Exception as exc:
        log_error(f"Failed to read parquet dataset '{input_path}': {exc}")
        raise SystemExit(1) from exc

    return df


# ----------------------------- PREPARATION ---------------------------------- #

def prepare_train_test_data(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, list[str]]:
    gene_cols = [c for c in df.columns if "(" in c and ")" in c]

    X = df[gene_cols]
    y = df["LN_IC50"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, gene_cols


# ----------------------------- MODEL TRAINING ------------------------------- #

def train_model(X_train_scaled: np.ndarray, y_train: pd.Series, random_state: int) -> ElasticNetCV:
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-3, 1, 50),
        cv=5,
        n_jobs=-1,
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)
    return model


# ----------------------------- EVALUATION ----------------------------------- #

def evaluate_model(model: ElasticNetCV, X_test_scaled: np.ndarray, y_test: pd.Series) -> tuple[float, float]:
    y_pred = model.predict(X_test_scaled)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return r2, rmse


# ----------------------------- OUTPUT --------------------------------------- #

def save_outputs(
    metrics_dir: Path,
    models_dir: Path,
    drug_slug: str,
    input_path: Path,
    random_state: int,
    df: pd.DataFrame,
    gene_cols: list[str],
    scaler: StandardScaler,
    model: ElasticNetCV,
    r2: float,
    rmse: float,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{drug_slug}_elasticnet_metrics.json"
    top_features_path = metrics_dir / f"{drug_slug}_elasticnet_top_features.csv"
    all_nonzero_features_path = metrics_dir / f"{drug_slug}_elasticnet_all_nonzero_features.csv"
    model_path = models_dir / f"{drug_slug}_elasticnet.joblib"

    metrics = {
        "dataset": str(input_path),
        "n_samples": int(df.shape[0]),
        "n_features": int(len(gene_cols)),
        "test_size": 0.2,
        "random_state": random_state,
        "r2": r2,
        "rmse": rmse,
        "best_alpha": float(model.alpha_),
        "best_l1_ratio": float(model.l1_ratio_),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    coef_df = pd.DataFrame(
        {
            "feature": gene_cols,
            "coefficient": model.coef_,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df_sorted = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df_sorted.head(25).to_csv(top_features_path, index=False)
    coef_df_sorted[coef_df_sorted["abs_coefficient"] > 0]

    artifact = {
        "scaler": scaler,
        "model": model,
        "feature_columns": gene_cols,
        "target_column": "LN_IC50",
    }
    joblib.dump(artifact, model_path)

    log_info(f"Saved metrics: {metrics_path}")
    log_info(f"Saved top features: {top_features_path}")
    log_info(f"Saved all non-zero features: {all_nonzero_features_path}")
    log_info(f"Saved model artifact: {model_path}")


def main() -> None:
    configure_logging()
    args = parse_args()

    config_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config).resolve()
    config = load_config(config_path)
    random_state = get_project_seed(config)
    paths = require_mapping(config.get("paths"), "paths")
    processed_paths = require_mapping(paths.get("processed"), "paths.processed")
    results_paths = require_mapping(paths.get("results"), "paths.results")

    drug_slug = slugify_drug_name(args.drug)
    if not drug_slug:
        log_error("Drug name produced an empty slug. Provide a valid --drug value.")
        raise SystemExit(1)

    per_drug_dir = resolve_path_from_repo(processed_paths.get("per_drug_dir"), "paths.processed.per_drug_dir")
    metrics_dir = resolve_path_from_repo(results_paths.get("metrics_dir"), "paths.results.metrics_dir")
    models_dir = resolve_path_from_repo(results_paths.get("models_dir", "results/models"), "paths.results.models_dir")
    input_path = per_drug_dir / f"{drug_slug}.parquet"

    if not input_path.exists() or not input_path.is_file():
        log_error(f"Input parquet does not exist: {input_path}")
        raise SystemExit(1)

    log_info(f"Loading dataset: {input_path}")
    df = load_dataset(input_path)

    gene_cols = [c for c in df.columns if "(" in c and ")" in c]
    if "LN_IC50" not in df.columns:
        log_error("Required target column 'LN_IC50' was not found in dataset.")
        raise SystemExit(1)
    if not gene_cols:
        log_error("No gene expression feature columns found using '(...)' pattern.")
        raise SystemExit(1)
    if df[gene_cols].isna().any().any():
        log_error("Feature matrix X contains missing values.")
        raise SystemExit(1)
    if df["LN_IC50"].isna().any():
        log_error("Target vector y contains missing values.")
        raise SystemExit(1)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, gene_cols = prepare_train_test_data(
        df=df,
        random_state=random_state,
    )

    log_info("Training ElasticNetCV model")
    model = train_model(X_train_scaled=X_train_scaled, y_train=y_train, random_state=random_state)

    r2, rmse = evaluate_model(model=model, X_test_scaled=X_test_scaled, y_test=y_test)

    log_info("ElasticNet results")
    log_info(f"R2: {r2:.4f}")
    log_info(f"RMSE: {rmse:.4f}")
    log_info(f"Best alpha: {model.alpha_}")
    log_info(f"Best l1_ratio: {model.l1_ratio_}")

    save_outputs(
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        drug_slug=drug_slug,
        input_path=input_path,
        random_state=random_state,
        df=df,
        gene_cols=gene_cols,
        scaler=scaler,
        model=model,
        r2=r2,
        rmse=rmse,
    )


if __name__ == "__main__":
    main()
