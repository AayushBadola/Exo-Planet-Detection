import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import optuna
import mlflow
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score # Example metric
import os
from src import data_loader, preprocessor, model_trainer
from src.config_utils import setup_config
from src.logger_utils import setup_logging

logger = None

def objective(trial: optuna.trial.Trial, cfg: DictConfig, X: pd.DataFrame, y: pd.Series):
    global logger
    if logger is None: # Basic logging if not fully set up
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    # --- MLflow Tracking for this Trial ---
    # Each trial is a nested run under the main Optuna study run
    with mlflow.start_run(nested=True) as trial_run:
        mlflow.set_tag("optuna_study_name", trial.study.study_name)
        mlflow.set_tag("optuna_trial_number", trial.number)
        
        # --- Define Hyperparameter Search Space for RandomForest ---
        # This should match the model defined in cfg.model, or be made more generic
        if cfg.model.name == "RandomForestClassifier":
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 50, step=5, log=True), # Log scale for wider search
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                # 'max_features': trial.suggest_float('max_features', 0.1, 1.0), # Example
                'random_state': cfg.random_state, # Keep random state fixed
                'n_jobs': -1
            }
            # Log chosen hyperparameters for this trial
            mlflow.log_params(model_params)
            
            # Create a temporary config for model_trainer for this trial
            trial_model_cfg = OmegaConf.create({'model': {'name': cfg.model.name, 'params': model_params}, 'random_state': cfg.random_state})
            model = model_trainer.get_model_instance(trial_model_cfg)
        else:
            logger.error(f"Hyperparameter tuning not defined for model: {cfg.model.name}")
            # Log an error metric or raise an exception
            mlflow.log_metric("cv_roc_auc_mean", -1.0) # Indicate failure
            return -1.0 # Optuna tries to maximize this, so a very low score for errors

        # --- Cross-Validation ---
        # Use StratifiedKFold for classification
        cv = StratifiedKFold(n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.random_state)
        
        # Define scoring metric (can be a list for multiple metrics with Optuna >3.0)
        # For single objective Optuna, choose one primary metric.
        # Using roc_auc here.
        scoring = make_scorer(roc_auc_score, needs_proba=True, average='weighted' if y.nunique() > 2 else None)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cfg.tuning.get('cv_n_jobs', 1))
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"Trial {trial.number}: Params: {trial.params}, CV ROC AUC Mean: {mean_score:.4f} +/- {std_score:.4f}")
            
            # Log metrics to MLflow for this trial
            mlflow.log_metric("cv_roc_auc_mean", mean_score)
            mlflow.log_metric("cv_roc_auc_std", std_score)
            for i, score in enumerate(scores):
                mlflow.log_metric(f"cv_roc_auc_fold_{i}", score)

            return mean_score # Optuna will try to maximize this value
        except Exception as e:
            logger.error(f"Error during CV for trial {trial.number}: {e}", exc_info=True)
            mlflow.log_metric("cv_roc_auc_mean", -1.0) # Indicate failure
            mlflow.set_tag("trial_status", "failed_cv")
            return -1.0 # Return a very low score if an error occurs

@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_hyperparameter_tuning(cfg: DictConfig) -> None:
    global logger
    cfg.base_dir = get_original_cwd()
    cfg = setup_config(cfg)
    logger = setup_logging(os.path.join(cfg.reports_dir, "tuning.log")) # Separate log for tuning

    logger.info("----------------------------------------------------")
    logger.info("--- Starting Hyperparameter Tuning (Optuna & MLflow) ---")
    logger.info("----------------------------------------------------")
    logger.info(f"Full configuration for tuning run:\n{OmegaConf.to_yaml(cfg)}")

    # --- MLflow Setup for the main Optuna study ---
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    # Create a new experiment or use a specific one for tuning
    tuning_experiment_name = f"{cfg.mlflow_experiment_name} - HyperTuning - {cfg.model.name}"
    mlflow.set_experiment(tuning_experiment_name)

    with mlflow.start_run(run_name="OptunaStudyOrchestrator") as study_orchestrator_run:
        mlflow.log_params(OmegaConf.to_container(cfg.tuning, resolve=True)) # Log tuning specific params
        mlflow.set_tag("tuning_model", cfg.model.name)

        # --- Load and Preprocess Data (once) ---
        logger.info("Loading and preprocessing data for tuning...")
        raw_df = data_loader.load_data(cfg.data_paths.raw_data_file)
        if raw_df is None:
            logger.error("Failed to load data for tuning. Exiting.")
            mlflow.set_tag("tuning_status", "failed_data_load")
            return

        # Preprocess without saving artifacts for each trial, only need X and y
        # If preprocessing itself has tunable params, this needs a different structure
        X, y, _ = preprocessor.preprocess_data(raw_df, cfg, save_artifacts=False)
        if X is None or y is None:
            logger.error("Failed to preprocess data for tuning. Exiting.")
            mlflow.set_tag("tuning_status", "failed_preprocessing")
            return
        
        logger.info(f"Data ready for tuning. X shape: {X.shape}, y shape: {y.shape}")

        # --- Optuna Study ---
        # Optuna can store study results in various backends, e.g., SQLite
        # Default is in-memory. For persistence:
        study_name = f"{cfg.project_name}-Tuning-{cfg.model.name}"
        storage_name = f"sqlite:///{os.path.join(cfg.model_dir, 'optuna_study.db')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction=cfg.tuning.direction, # e.g., "maximize"
            storage=storage_name,
            load_if_exists=True # Resume study if it exists
        )
        
        # MLflow callback for Optuna to log each trial automatically to the parent MLflow run
        # This callback logs to the `study_orchestrator_run`
        # mlflow_optuna_callback = optuna.integration.MLflowCallback(
        #     tracking_uri=mlflow.get_tracking_uri(), # Use current MLflow URI
        #     metric_name="cv_roc_auc_mean" # Primary metric to track
        # )
        # Callbacks can sometimes be tricky with nested runs. Manual logging in `objective` is more explicit.

        logger.info(f"Starting Optuna study: {study_name} with {cfg.tuning.n_trials} trials. Storage: {storage_name}")
        study.optimize(
            lambda trial: objective(trial, cfg, X, y),
            n_trials=cfg.tuning.n_trials,
            timeout=cfg.tuning.get("timeout_seconds", None), # Optional timeout
            # callbacks=[mlflow_optuna_callback] # Add if using MLflow callback
        )

        # --- Log Best Trial Info ---
        logger.info("\n--- Optuna Study Complete ---")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        
        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best value ({cfg.tuning.direction} {cfg.tuning.metric_to_optimize}): {best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # Log best trial's parameters and metrics to the main study orchestrator run in MLflow
        mlflow.log_metric("best_trial_value", best_trial.value)
        mlflow.log_params({f"best_param_{k}": v for k, v in best_trial.params.items()})
        mlflow.set_tag("best_trial_number", best_trial.number)

        # Save the best hyperparameters to a YAML file (optional)
        best_params_path = os.path.join(cfg.model_dir, f"best_params_{cfg.model.name}.yaml")
        best_params_dict = {"model": {"name": cfg.model.name, "params": best_trial.params}}
        with open(best_params_path, 'w') as f:
            OmegaConf.save(config=OmegaConf.create(best_params_dict), f=f)
        logger.info(f"Best hyperparameters saved to: {best_params_path}")
        mlflow.log_artifact(best_params_path, artifact_path="best_hyperparameters")
        
        mlflow.set_tag("tuning_status", "success")

        # You can then update your main `conf/model/<model_name>.yaml` with these best params
        # or create a new config like `conf/model/<model_name>_tuned.yaml` and point to it in `config.yaml`

if __name__ == "__main__":
    run_hyperparameter_tuning()