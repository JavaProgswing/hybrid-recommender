import json
import logging
import os
from pathlib import Path

import optuna

# Adjust sys.path or import according to the project structure
from src.evaluation.evaluation import run_evaluation

logger = logging.getLogger(__name__)

def run_hyperparameter_optimization(n_trials: int = 20) -> dict:
    """
    Runs an Optuna study to find the best blending weights for the hybrid recommender.
    The objective is to maximize the NDCG@10 evaluation metric.
    
    Args:
        n_trials: Number of parameter combinations to try.
        
    Returns:
        A dictionary containing the best weights and the best score.
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

    def objective(trial):
        # Suggest values for alpha, beta, and gamma between 0.0 and 1.0
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        beta = trial.suggest_float("beta", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        
        # If all weights are 0, it's invalid. Optuna should avoid this but we can return 0.0
        if alpha + beta + gamma == 0:
            return 0.0
            
        weights = {"alpha": alpha, "beta": beta, "gamma": gamma}
        
        try:
            # Run evaluation in hybrid mode
            # We use k=10 as the standard benchmarking cutoff
            results = run_evaluation(k=10, mode="hybrid", weights=weights)
            
            # Extract the NDCG metric for hybrid mode
            # evaluation.py returns: {'hybrid': {'precision': 0.x, 'recall': 0.x, 'ndcg': 0.x}}
            hybrid_results = results.get("hybrid", {})
            ndcg_score = hybrid_results.get("ndcg", 0.0)
            
            return ndcg_score
        except Exception as e:
            logger.error(f"Error during evaluation trial: {e}")
            return 0.0

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Optimization finished. Best NDCG@10: {best_score}")
    logger.info(f"Best parameters: {best_params}")
    
    # Save the optimal weights to a JSON file in the models directory
    # so they can be loaded by the backend upon startup or rebuild
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    optimal_weights_path = models_dir / "optimal_weights.json"
    
    output_data = {
        "weights": best_params,
        "metrics": {
            "ndcg_at_10": best_score
        },
        "trials": n_trials
    }
    
    with open(optimal_weights_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    logger.info(f"Saved optimal weights to {optimal_weights_path}")
    
    return output_data
