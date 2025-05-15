import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import optuna
import mlflow
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
import os
from src import data_loader, preprocessor, model_trainer
from src.config_utils import setup_config
from src.logger_utils import setup_logging

logger = None

def objective(trial: optuna.trial.Trial, cfg: DictConfig, X: pd.DataFrame, y: pd.Series):
    global logger
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    with mlflow.start_run(nested=True) as trial_run:
        mlflow.set_tag("optuna_study_name", trial.study.study_name)
        mlflow.set_tag("optuna_trial_number", trial.number)
        
        if cfg.model.name == "RandomForestClassifier":
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 50, step=5, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': cfg.random_state,
                'n_jobs': -1
            }
            mlflow.log_params(model_params)
            
            trial_model_cfg = OmegaConf.create({'model': {'name': cfg.model.name, 'params': model_params}, 'random_state': cfg.random_state})
            model = model_trainer.get_model_instance(trial_model_cfg)
        else:
            logger.error(f"Hyperparameter tuning not defined for model: {cfg.model.name}")
            mlflow.log_metric("cv_roc_auc_mean", -1.0)
            return -1.0

        cv = StratifiedKFold(n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.random_state)
        
        scoring = make_scorer(roc_auc_score, needs_proba=True, average='weighted' if y.nunique() > 2 else None)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cfg.tuning.get('cv_n_jobs', 1))
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"Trial {trial.number}: Params: {trial.params}, CV ROC AUC Mean: {mean_score:.4f} +/- {std_score:.4f}")
            
            mlflow.log_metric("cv_roc_auc_mean", mean_score)
            mlflow.log_metric("cv_roc_auc_std", std_score)
            for i, score in enumerate(scores):
                mlflow.log_metric(f"cv_roc_auc_fold_{i}", score)

            return mean_score
        except Exception as e:
            logger.error(f"Error during CV for trial {trial.number}: {e}", exc_info=True)
            mlflow.log_metric("cv_roc_auc_mean", -1.0)
            mlflow.set_tag("trial_status", "failed_cv")
            return -1.0

@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_hyperparameter_tuning(cfg: DictConfig) -> None:
    global logger
    cfg.base_dir = get_original_cwd()
    cfg = setup_config(cfg)
    logger = setup_logging(os.path.join(cfg.reports_dir, "tuning.log"))

    logger.info("----------------------------------------------------")
    logger.info("--- Starting Hyperparameter Tuning (Optuna & MLflow) ---")
    logger.info("----------------------------------------------------")
    logger.info(f"Full configuration for tuning run:\n{OmegaConf.to_yaml(cfg)}")

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    tuning_experiment_name = f"{cfg.mlflow_experiment_name} - HyperTuning - {cfg.model.name}"
    mlflow.set_experiment(tuning_experiment_name)

    with mlflow.start_run(run_name="OptunaStudyOrchestrator") as study_orchestrator_run:
        mlflow.log_params(OmegaConf.to_container(cfg.tuning, resolve=True))
        mlflow.set_tag("tuning_model", cfg.model.name)

        logger.info("Loading and preprocessing data for tuning...")
        raw_df = data_loader.load_data(cfg.data_paths.raw_data_file_abs) # Corrected to use absolute path
        if raw_df is None:
            logger.error("Failed to load data for tuning. Exiting.")
            mlflow.set_tag("tuning_status", "failed_data_load")
            return

        X, y, _ = preprocessor.preprocess_data(raw_df, cfg, save_artifacts=False)
        if X is None or y is None:
            logger.error("Failed to preprocess data for tuning. Exiting.")
            mlflow.set_tag("tuning_status", "failed_preprocessing")
            return
        
        logger.info(f"Data ready for tuning. X shape: {X.shape}, y shape: {y.shape}")

        study_name = f"{cfg.project_name}-Tuning-{cfg.model.name}"
        # Ensure model_dir exists before creating the SQLite file path there
        os.makedirs(cfg.model_dir, exist_ok=True) 
        storage_name = f"sqlite:///{os.path.join(cfg.model_dir, 'optuna_study.db')}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction=cfg.tuning.direction,
            storage=storage_name,
            load_if_exists=True
        )
        
        logger.info(f"Starting Optuna study: {study_name} with {cfg.tuning.n_trials} trials. Storage: {storage_name}")
        study.optimize(
            lambda trial: objective(trial, cfg, X, y),
            n_trials=cfg.tuning.n_trials,
            timeout=cfg.tuning.get("timeout_seconds", None)
        )

        logger.info("\n--- Optuna Study Complete ---")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        
        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best value ({cfg.tuning.direction} {cfg.tuning.metric_to_optimize}): {best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        mlflow.log_metric("best_trial_value", best_trial.value)
        mlflow.log_params({f"best_param_{k}": v for k, v in best_trial.params.items()})
        mlflow.set_tag("best_trial_number", best_trial.number)

        best_params_path = os.path.join(cfg.model_dir, f"best_params_{cfg.model.name}.yaml")
        best_params_dict = {"model": {"name": cfg.model.name, "params": best_trial.params}}
        
        # Ensure the directory for best_params_path exists
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)

        with open(best_params_path, 'w') as f:
            OmegaConf.save(config=OmegaConf.create(best_params_dict), f=f)
        logger.info(f"Best hyperparameters saved to: {best_params_path}")
        mlflow.log_artifact(best_params_path, artifact_path="best_hyperparameters")
        
        mlflow.set_tag("tuning_status", "success")

if __name__ == "__main__":
    run_hyperparameter_tuning()