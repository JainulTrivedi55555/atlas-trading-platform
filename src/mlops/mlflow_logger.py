"""
MLflow Logging Helpers
Provides a clean interface for logging ATLAS model training runs.

Phase 16 Fixes:
- Use absolute path for SQLite DB (fixes Windows relative path issue)
- Use tempfile.gettempdir() instead of /tmp/ (fixes Windows artifact issue)
- Auto-create mlruns/ directory if it doesn't exist
"""

import mlflow
import mlflow.sklearn
import logging
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger("atlas.mlflow")

# ── Resolve absolute path to project root ─────────────────────────────────────
# This file is at src/mlops/mlflow_logger.py
# So project root is 3 levels up: src/mlops/ -> src/ -> project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_MLRUNS_DIR   = _PROJECT_ROOT / "mlruns"
_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)   # create if missing

_DB_PATH = _MLRUNS_DIR / "atlas_mlflow.db"

# Use absolute path so it works regardless of working directory
MLFLOW_TRACKING_URI = f"sqlite:///{_DB_PATH}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# ──────────────────────────────────────────────────────────────────────────────


def get_or_create_experiment(experiment_name: str) -> str:
    """Get existing experiment or create new one."""

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def log_training_run(
    ticker: str,
    model_type: str,
    regime: str,
    model,
    params: dict,
    metrics: dict,
    feature_names: list,
    feature_importances: list,
    run_description: str = "",
) -> str:
    """
    Log a complete model training run to MLflow.
    Returns the MLflow run_id for reference.
    """

    experiment_name = f"ATLAS_{ticker}_{model_type}"
    experiment_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:

        # Tags
        mlflow.set_tags({
            "ticker": ticker,
            "model_type": model_type,
            "regime": regime,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "phase": "16",
        })

        # Parameters
        mlflow.log_params(params)

        # Metrics
        mlflow.log_metrics(metrics)

        # Model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Feature importance CSV
        # FIX: use tempfile.gettempdir() instead of hardcoded /tmp/
        # which does not exist on Windows
        if feature_names and feature_importances:

            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": feature_importances,
            }).sort_values("importance", ascending=False)

            tmp_dir = Path(tempfile.gettempdir())
            fi_path = tmp_dir / f"feature_importance_{ticker}_{regime}.csv"
            fi_df.to_csv(fi_path, index=False)

            mlflow.log_artifact(str(fi_path), "feature_importance")

        run_id = run.info.run_id

        logger.info(
            f"MLflow run logged: {experiment_name} | "
            f"accuracy={metrics.get('accuracy', 0):.3f} | "
            f"run_id={run_id[:8]}"
        )

        return run_id


def get_best_run(ticker: str, model_type: str, metric: str = "accuracy") -> dict:
    """
    Retrieve the best run for a ticker+model combination.
    Returns run info dict with params, metrics, run_id.
    """

    experiment_name = f"ATLAS_{ticker}_{model_type}"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        return {}

    best = runs.iloc[0]

    return {
        "run_id": best["run_id"],
        "accuracy": best.get("metrics.accuracy", None),
        "f1_score": best.get("metrics.f1_score", None),
        "trained_at": best.get("tags.trained_at", "unknown"),
        "regime": best.get("tags.regime", "unknown"),
    }


def get_all_experiments_summary() -> list:
    """
    Get a summary of all ATLAS experiments for the dashboard.
    Returns list of dicts with latest run info per experiment.
    """

    experiments = mlflow.search_experiments(
        filter_string="name LIKE 'ATLAS_%'"
    )

    summary = []

    for exp in experiments:

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs.empty:
            r = runs.iloc[0]

            summary.append({
                "experiment": exp.name,
                "latest_run": r.get("tags.trained_at", "unknown"),
                "accuracy": round(r.get("metrics.accuracy", 0), 4),
                "f1_score": round(r.get("metrics.f1_score", 0), 4),
                "run_id": r["run_id"][:8] + "...",
            })

    return summary