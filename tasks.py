"""
Celery tasks for the Hybrid Recommender System.
Heavy recommendation computation is moved here so the API
thread returns immediately with a task_id.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from celery_app import celery_app  # noqa: E402

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.compute_recommendations",
    max_retries=3,
    default_retry_delay=5,
)
def compute_recommendations(self, item_title: str, top_n: int = 10, explain: bool = False):
    """
    Background task: run hybrid recommendation computation.

    Imported lazily inside the task to avoid loading heavy ML models
    at import time on the API process.

    Returns:
        dict with query_item, recommendations, weights, explain flag.

    Raises:
        ValueError  — item not found or models not built.
        Retries up to 3 times on transient failures.
    """
    try:
        from backend.main import models

        if not models["ready"]:
            raise ValueError("Models not built. Call POST /api/build first.")

        recs = models["hybrid"].recommend(item_title, top_n=top_n, explain=explain)

        if not recs:
            raise ValueError(f"Item '{item_title}' not found or no recommendations available.")

        logger.info(
            "compute_recommendations completed: item=%s top_n=%d",
            item_title,
            top_n,
        )

        return {
            "query_item": item_title,
            "recommendations": recs,
            "weights": models["hybrid"].get_weights(),
            "explain": explain,
        }

    except ValueError:
        raise

    except Exception as exc:
        logger.error(
            "compute_recommendations failed for item=%s: %s",
            item_title,
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc)

@celery_app.task(
    bind=True,
    name="tasks.optimize_hyperparameters",
    max_retries=1,
    default_retry_delay=30,
)
def optimize_hyperparameters_task(self, n_trials: int = 20):
    """
    Background task: run hyperparameter optimization using Optuna.
    """
    try:
        from src.model.optimization import run_hyperparameter_optimization
        
        logger.info(f"Starting background hyperparameter optimization with {n_trials} trials.")
        result = run_hyperparameter_optimization(n_trials=n_trials)
        logger.info("Hyperparameter optimization completed successfully.")
        return result
        
    except Exception as exc:
        logger.error(
            "optimize_hyperparameters_task failed: %s",
            exc,
            exc_info=True,
        )
        raise self.retry(exc=exc)

