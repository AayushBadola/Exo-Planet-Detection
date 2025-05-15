import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
import joblib
from prettytable import PrettyTable # For beautifying the output

# Custom utility modules
from src import data_loader, preprocessor, model_trainer, predict
from src.config_utils import setup_config
from src.logger_utils import setup_logging

# logger = None # Will be initialized after config setup

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main_hydra(cfg: DictConfig) -> None:
    # global logger

    # --- Initial Setup ---
    cfg.base_dir = get_original_cwd()
    cfg = setup_config(cfg)
    logger = setup_logging(cfg.log_file)

    # ---- REMOVE OR COMMENT OUT DEBUG PRINTS FOR NORMAL RUNS ----
    # print("--- MAIN.PY DEBUG: PIPELINE START ---")
    logger.info("----------------------------------------------------")
    logger.info("--- Starting Exoplanet Prediction Pipeline (Hydra) ---")
    logger.info("----------------------------------------------------")
    logger.info(f"Full configuration before MLflow ops:\n{OmegaConf.to_yaml(cfg)}")

    # --- MLflow Setup ---
    # print(f"--- MAIN.PY DEBUG: Setting MLflow tracking URI to: {cfg.mlflow_tracking_uri} ---")
    logger.info(f"Attempting to set MLflow tracking URI to: {cfg.mlflow_tracking_uri}")
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    experiment_name_from_cfg = cfg.mlflow_experiment_name
    # print(f"--- MAIN.PY DEBUG: Target MLflow experiment name: '{experiment_name_from_cfg}' ---")
    logger.info(f"Target MLflow experiment name: '{experiment_name_from_cfg}'")

    active_experiment_id_for_run = None
    client = MlflowClient() # Initialize client

    try:
        experiment_obj = client.get_experiment_by_name(name=experiment_name_from_cfg)
        
        if experiment_obj:
            if experiment_obj.lifecycle_stage == "deleted":
                # print(f"--- MAIN.PY DEBUG: Experiment '{experiment_name_from_cfg}' (ID: {experiment_obj.experiment_id}) found but is deleted. Attempting to recreate via client.create_experiment...")
                logger.info(f"Experiment '{experiment_name_from_cfg}' (ID: {experiment_obj.experiment_id}) found but is deleted. Attempting to recreate.")
                try:
                    # For FileStore, if mlruns/ID exists, create_experiment might fail or make a new one.
                    # It's safer to let it try to create with the same name; it might get a new ID if needed.
                    active_experiment_id_for_run = client.create_experiment(name=experiment_name_from_cfg)
                    experiment_obj = client.get_experiment(active_experiment_id_for_run) 
                    # print(f"--- MAIN.PY DEBUG: MlflowClient.create_experiment SUCCEEDED after finding deleted. New/Restored Experiment Name: '{experiment_obj.name}', ID: '{experiment_obj.experiment_id}', Artifact: '{experiment_obj.artifact_location}' ---")
                    logger.info(f"MLflow experiment '{experiment_obj.name}' (re)created via client: ID='{active_experiment_id_for_run}', ArtifactLocation='{experiment_obj.artifact_location}'")
                except mlflow.exceptions.MlflowException as e_create_deleted:
                    # This can happen if create_experiment with same name is not allowed after deletion or if artifact path clashes
                    # print(f"--- MAIN.PY DEBUG: MlflowClient.create_experiment FAILED for '{experiment_name_from_cfg}' after finding it deleted: {e_create_deleted}. Trying to set existing one as active if possible or creating with a new suffixed name. ---")
                    logger.error(f"Failed to simply re-create experiment '{experiment_name_from_cfg}' after finding it deleted: {e_create_deleted}. This might indicate an issue with the artifact store path or a name clash that create_experiment cannot resolve with the same name.", exc_info=True)
                    # Fallback: If create failed, the original experiment_obj (deleted) is all we have info on.
                    # It's unlikely we can proceed with a deleted experiment.
                    # Forcing an error is safer here.
                    raise RuntimeError(f"Experiment '{experiment_name_from_cfg}' is deleted and could not be cleanly recreated with the same name. Please manually clean up MLflow backend (mlruns, mlflow.db) or use a new experiment name.")
            else: # Experiment exists and is active
                active_experiment_id_for_run = experiment_obj.experiment_id
                # print(f"--- MAIN.PY DEBUG: MlflowClient.get_experiment_by_name SUCCEEDED. Experiment Name: '{experiment_obj.name}', ID: '{experiment_obj.experiment_id}', Artifact: '{experiment_obj.artifact_location}' ---")
                logger.info(f"MLflow experiment '{experiment_obj.name}' found with ID '{active_experiment_id_for_run}', ArtifactLocation='{experiment_obj.artifact_location}'. Setting active.")
        
        else: # Experiment does not exist, create it using MlflowClient
            # print(f"--- MAIN.PY DEBUG: Experiment '{experiment_name_from_cfg}' NOT FOUND by name. Attempting to create with MlflowClient...")
            logger.info(f"Experiment '{experiment_name_from_cfg}' not found. Creating it now via MlflowClient.")
            active_experiment_id_for_run = client.create_experiment(name=experiment_name_from_cfg)
            experiment_obj = client.get_experiment(active_experiment_id_for_run) 
            # print(f"--- MAIN.PY DEBUG: MlflowClient.create_experiment SUCCEEDED. Experiment Name: '{experiment_obj.name}', ID: '{experiment_obj.experiment_id}', Artifact: '{experiment_obj.artifact_location}' ---")
            logger.info(f"MLflow experiment created via client: ID='{active_experiment_id_for_run}', Name='{experiment_obj.name}', ArtifactLocation='{experiment_obj.artifact_location}'")

        if active_experiment_id_for_run:
            mlflow.set_experiment(experiment_id=active_experiment_id_for_run) # Set active context
            # print(f"--- MAIN.PY DEBUG: mlflow.set_experiment(experiment_id='{active_experiment_id_for_run}') called to ensure context for fluent API. ---")
            logger.info(f"Active experiment context successfully set to ID: {active_experiment_id_for_run} ('{experiment_obj.name}')")
        else:
            # print(f"--- MAIN.PY DEBUG: CRITICAL - active_experiment_id_for_run is None after all attempts. Cannot set active experiment. ---")
            logger.error("CRITICAL - Could not determine or set an active experiment ID. MLflow logging will likely fail or go to Default.")
            raise RuntimeError("Failed to set or create the MLflow experiment properly using MlflowClient.")

    except Exception as e_exp_handling:
        # print(f"--- MAIN.PY DEBUG: EXCEPTION during MLflow experiment handling for '{experiment_name_from_cfg}': {e_exp_handling} ---")
        logger.error(f"Exception during MLflow experiment handling for '{experiment_name_from_cfg}': {e_exp_handling}", exc_info=True)
        import traceback
        traceback.print_exc()
        # print(f"--- MAIN.PY DEBUG: Critical failure in MLflow experiment setup. Aborting. ---")
        logger.critical("Critical failure in MLflow experiment setup. Aborting pipeline.")
        return # Abort pipeline


    # print(f"--- MAIN.PY DEBUG: About to start mlflow.start_run(). Determined experiment ID for this run: {active_experiment_id_for_run} ---")
    # Pass experiment_id to start_run for robustness, especially with FileStore.
    with mlflow.start_run(experiment_id=active_experiment_id_for_run, run_name=f"Run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        actual_experiment_id_of_run = run.info.experiment_id 
        # print(f"--- MAIN.PY DEBUG: MLflow Run ID: {run_id} started in Actual Experiment ID: {actual_experiment_id_of_run} ---")
        logger.info(f"MLflow Run ID: {run_id} (Name: {run.data.tags.get('mlflow.runName')}) started in Experiment ID: {actual_experiment_id_of_run}")

        param_groups_to_log = {
            "data_paths": cfg.get("data_paths"),
            "preprocessing": cfg.get("preprocessing"),
            "model": cfg.get("model"),
            "tuning": cfg.get("tuning")
        }
        for group_name, group_cfg in param_groups_to_log.items():
            if group_cfg:
                try:
                    mlflow.log_params({f"{group_name}.{k}": v for k, v in OmegaConf.to_container(group_cfg, resolve=True).items()})
                except Exception as e_param_log:
                    logger.warning(f"Could not log parameters for group '{group_name}': {e_param_log}")
        top_level_params_to_log = ["project_name", "random_state", "test_size", "prediction_sample_size"]
        for param_name in top_level_params_to_log:
            if hasattr(cfg, param_name):
                mlflow.log_param(param_name, getattr(cfg, param_name))
        # print("--- MAIN.PY DEBUG: Parameters logged (or attempted) ---")

        # --- 1. Load Data ---
        # print("--- MAIN.PY DEBUG: Starting Data Loading ---")
        raw_df = data_loader.load_data(cfg.data_paths.raw_data_file_abs)
        if raw_df is None:
            logger.error("Failed to load data. Exiting pipeline.")
            mlflow.set_tag("pipeline_status", "failed_data_load")
            return
        mlflow.log_metric("raw_data_rows", len(raw_df))
        mlflow.log_metric("raw_data_cols", len(raw_df.columns))
        logger.info(f"Raw data loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns.")
        # print("--- MAIN.PY DEBUG: Data Loading Complete ---")

        # --- 2. Preprocess Data ---
        # print("--- MAIN.PY DEBUG: Starting Preprocessing ---")
        X, y, fitted_transformers = preprocessor.preprocess_data(raw_df, cfg, save_artifacts=True)
        if X is None or y is None or X.empty or y.empty:
            logger.error("Failed to preprocess data. Exiting pipeline.")
            mlflow.set_tag("pipeline_status", "failed_preprocessing")
            return
        # print("--- MAIN.PY DEBUG: Preprocessing Complete. Artifacts saved by preprocessor.py ---")
        for artifact_name, artifact_path_key_base in [
            ("scaler", "scaler_path"),
            ("imputer", "imputer_path"),
            ("training_columns", "training_columns_path"),
            ("poly_features", "poly_features_path")
        ]:
            artifact_path = getattr(cfg, artifact_path_key_base, None)
            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, artifact_path="preprocessing_artifacts")
            elif artifact_path_key_base == "poly_features_path" and cfg.preprocessing.get('apply_polynomial_features', False) and not (artifact_path and os.path.exists(artifact_path)):
                 logger.warning(f"Polynomial features enabled but artifact '{artifact_path_key_base}' at path '{artifact_path}' not found or path is None.")
        mlflow.log_param("final_feature_count", len(X.columns))
        # print("--- MAIN.PY DEBUG: Preprocessing Artifacts Logged ---")

        # --- 3. Split Data ---
        # print("--- MAIN.PY DEBUG: Starting Data Splitting ---")
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, cfg)
        if X_train is None or X_train.empty:
            logger.error("Failed to split data or training data is empty. Exiting pipeline.")
            mlflow.set_tag("pipeline_status", "failed_data_split")
            return
        logger.info(f"Data split complete. X_train: {X_train.shape}, y_train: {y_train.shape}")
        # print("--- MAIN.PY DEBUG: Data Splitting Complete ---")

        # --- 4. Train Model ---
        # print("--- MAIN.PY DEBUG: Starting Model Training ---")
        model = model_trainer.train_model(X_train, y_train, cfg)
        if model is None:
            logger.error("Failed to train model. Exiting pipeline.")
            mlflow.set_tag("pipeline_status", "failed_model_training")
            return
        # print("--- MAIN.PY DEBUG: Model Training Complete. Model .joblib saved by model_trainer.py ---")
        
        if not X_train.empty:
            input_example = X_train.head(5)
            try:
                # print("--- MAIN.PY DEBUG: About to infer signature ---")
                signature = infer_signature(input_example, model.predict(input_example))
                # print(f"--- MAIN.PY DEBUG: Signature inferred: {signature}. About to log model with mlflow.sklearn.log_model. ---")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"mlflow_sklearn_{cfg.model.name.lower()}",
                    signature=signature,
                    input_example=input_example
                )
                # print(f"--- MAIN.PY DEBUG: mlflow.sklearn.log_model (with signature) COMPLETED for '{cfg.model.name}'. ---")
                logger.info(f"Model '{cfg.model.name}' (with signature) logged to MLflow.")
            except Exception as sig_e:
                # print(f"--- MAIN.PY DEBUG: EXCEPTION during signature inference or mlflow.sklearn.log_model: {sig_e} ---")
                logger.error(f"Could not infer signature or log model with signature: {sig_e}", exc_info=True)
                import traceback
                traceback.print_exc()
                # print(f"--- MAIN.PY DEBUG: Attempting to log model without signature after exception. ---")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"mlflow_sklearn_{cfg.model.name.lower()}"
                )
                # print(f"--- MAIN.PY DEBUG: mlflow.sklearn.log_model (without signature) COMPLETED for '{cfg.model.name}' (after exception). ---")
                logger.info(f"Model '{cfg.model.name}' (without signature, due to error) logged to MLflow.")
        else:
            logger.warning("X_train is empty, cannot create input_example for MLflow model signature.")
            # print(f"--- MAIN.PY DEBUG: X_train empty, logging model without signature. ---")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"mlflow_sklearn_{cfg.model.name.lower()}"
            )
            # print(f"--- MAIN.PY DEBUG: mlflow.sklearn.log_model (without signature, X_train empty) COMPLETED for '{cfg.model.name}'. ---")
            logger.info(f"Model '{cfg.model.name}' (without signature due to empty X_train) logged to MLflow.")

        # --- 5. Evaluate Model ---
        # print("--- MAIN.PY DEBUG: Proceeding to Evaluate Model ---")
        evaluation_metrics = model_trainer.evaluate_model(model, X_test, y_test, cfg)
        if evaluation_metrics:
            final_metrics_to_log = {}
            for k, v_orig in evaluation_metrics.items():
                v = v_orig
                try:
                    if hasattr(v, 'item'):
                        v = v.item()
                    v_float = float(v)
                    final_metrics_to_log[k] = v_float
                except (ValueError, TypeError):
                    logger.warning(f"Metric '{k}' with value '{v_orig}' (type {type(v_orig)}) is not numeric. Skipping logging this metric to MLflow.")
            
            if final_metrics_to_log:
                mlflow.log_metrics(final_metrics_to_log)
                logger.info(f"Evaluation metrics logged: {final_metrics_to_log}")
                # print(f"--- MAIN.PY DEBUG: Evaluation Metrics Logged: {final_metrics_to_log} ---")
            else:
                logger.warning("No valid numeric evaluation metrics to log after sanitization.")
                # print("--- MAIN.PY DEBUG: No valid numeric evaluation metrics to log. ---")
        else:
            logger.warning("No evaluation metrics returned from model_trainer.evaluate_model.")
            mlflow.set_tag("evaluation_status", "metrics_missing")
            # print("--- MAIN.PY DEBUG: No evaluation metrics returned. ---")

        # --- Feature Importances ---
        # print("--- MAIN.PY DEBUG: Proceeding to Get Feature Importances ---")
        if hasattr(model, 'feature_importances_'):
            model_trainer.get_feature_importances(model, X_train.columns.tolist(), cfg)
            fi_plot_path = os.path.join(cfg.reports_dir, "feature_importances.png")
            if os.path.exists(fi_plot_path):
                mlflow.log_artifact(fi_plot_path, artifact_path="evaluation_plots")
                # print(f"--- MAIN.PY DEBUG: Feature importances plot logged from {fi_plot_path} ---")
        else:
            # print(f"--- MAIN.PY DEBUG: Model does not have feature_importances_. Skipping. ---")
            pass
        
        logger.info("\n--- Exoplanet Prediction Pipeline Completed Successfully ---")
        mlflow.set_tag("pipeline_status", "success")
        # print("--- MAIN.PY DEBUG: Pipeline status 'success' tagged. ---")

        # --- 6. Prediction Demonstration (Optional) ---
        if cfg.get("run_prediction_demo", True):
            # print("--- MAIN.PY DEBUG: Starting Prediction Demonstration ---")
            logger.info("\n--- Demonstrating Prediction on a Sample ---")
            
            demo_artifacts_for_pred = {}
            paths_to_load_for_demo = {
                'scaler': cfg.scaler_path,
                'imputer': cfg.imputer_path,
                'training_columns': cfg.training_columns_path
            }
            if cfg.preprocessing.get('apply_polynomial_features', False) and cfg.poly_features_path:
                 paths_to_load_for_demo['poly_features'] = cfg.poly_features_path

            valid_demo_artifacts = True
            for name, path in paths_to_load_for_demo.items():
                if path and os.path.exists(path):
                    try:
                        demo_artifacts_for_pred[name] = joblib.load(path) 
                    except Exception as e_load:
                        logger.error(f"Prediction demo: Error loading artifact '{name}' from {path}: {e_load}")
                        demo_artifacts_for_pred[name] = None
                        valid_demo_artifacts = False
                elif path:
                    logger.warning(f"Prediction demo: Artifact '{name}' not found at {path} for demo.")
                    demo_artifacts_for_pred[name] = None
                    if name in ['scaler', 'imputer', 'training_columns']: 
                        valid_demo_artifacts = False 
                else:
                    demo_artifacts_for_pred[name] = None
            
            if not demo_artifacts_for_pred.get('training_columns'): 
                logger.error("Prediction demo: training_columns artifact missing. Cannot proceed with prediction demo.")
                valid_demo_artifacts = False

            if model and valid_demo_artifacts: 
                if y_test is not None and not y_test.empty:
                    sample_indices = y_test.sample(n=min(cfg.prediction_sample_size, len(y_test)), random_state=cfg.random_state).index
                    
                    sample_raw_data_for_pred = pd.DataFrame() 
                    if raw_df.index.is_unique and all(idx in raw_df.index for idx in sample_indices):
                         sample_raw_data_for_pred = raw_df.loc[sample_indices].copy()
                    else: 
                        logger.warning("Prediction demo: Could not directly map y_test indices to raw_df. Using head of raw_df for demo.")
                        sample_raw_data_for_pred = raw_df.head(cfg.prediction_sample_size).copy()

                    actual_labels_sample_for_pred = None
                    if cfg.preprocessing.target_column in sample_raw_data_for_pred.columns:
                         sample_raw_data_for_pred_copy = sample_raw_data_for_pred.dropna(subset=[cfg.preprocessing.target_column])
                         if not sample_raw_data_for_pred_copy.empty: # Check if df is not empty after dropna
                            sample_raw_data_for_pred_copy["target_temp_demo"] = sample_raw_data_for_pred_copy[cfg.preprocessing.target_column].apply(
                                lambda x: 1 if x in cfg.preprocessing.positive_labels else (0 if x == cfg.preprocessing.negative_label else -1)
                            )
                            actual_labels_sample_for_pred = sample_raw_data_for_pred_copy["target_temp_demo"][sample_raw_data_for_pred_copy["target_temp_demo"] != -1].copy()
                            actual_labels_sample_for_pred = actual_labels_sample_for_pred.reindex(sample_raw_data_for_pred.index) # Align with original sample_raw_data_for_pred

                    if not sample_raw_data_for_pred.empty:
                        # print(f"--- MAIN.PY DEBUG: Prediction demo: Preprocessing sample of size {len(sample_raw_data_for_pred)} ---")
                        sample_processed = predict.preprocess_for_prediction(sample_raw_data_for_pred, demo_artifacts_for_pred, cfg)
                        
                        if sample_processed is not None and not sample_processed.empty:
                            # print(f"--- MAIN.PY DEBUG: Prediction demo: Making predictions on processed sample of size {len(sample_processed)} ---")
                            predictions_sample, probabilities_sample = predict.make_prediction(model, sample_processed) 

                            if predictions_sample is not None:
                                logger.info("\nSample Predictions (Actual vs. Predicted with Confidence):")
                                
                                prediction_table = PrettyTable()
                                prediction_table.field_names = ["Original Index", "Actual Label", "Predicted Label", "Confidence", "P(Exoplanet=1)"]
                                prediction_table.align["Original Index"] = "r"
                                prediction_table.align["Actual Label"] = "c"
                                prediction_table.align["Predicted Label"] = "c"
                                prediction_table.align["Confidence"] = "r"
                                prediction_table.align["P(Exoplanet=1)"] = "r"

                                for i in range(len(predictions_sample)):
                                    actual_val_str = "N/A"
                                    if actual_labels_sample_for_pred is not None and i < len(actual_labels_sample_for_pred) and not pd.isna(actual_labels_sample_for_pred.iloc[i]):
                                        actual_val_int = int(actual_labels_sample_for_pred.iloc[i])
                                        actual_val_str = "Exoplanet" if actual_val_int == 1 else "Not Exoplanet"
                                    
                                    original_idx = sample_raw_data_for_pred.index[i] if i < len(sample_raw_data_for_pred.index) else "Unknown"
                                    
                                    predicted_class = predictions_sample[i]
                                    predicted_class_str = "Exoplanet" if predicted_class == 1 else "Not Exoplanet"

                                    proba_exoplanet_val_str = "N/A"
                                    confidence_score_str = "N/A"

                                    if probabilities_sample is not None and probabilities_sample.ndim == 2 and i < len(probabilities_sample) and probabilities_sample.shape[1] == 2:
                                        proba_class_0 = probabilities_sample[i][0]
                                        proba_class_1 = probabilities_sample[i][1]
                                        proba_exoplanet_val_str = f'{proba_class_1:.4f}'

                                        if predicted_class == 1:
                                            confidence_score_str = f'{proba_class_1:.4f}'
                                        elif predicted_class == 0:
                                            confidence_score_str = f'{proba_class_0:.4f}'
                                    
                                    prediction_table.add_row([
                                        original_idx,
                                        actual_val_str,
                                        predicted_class_str,
                                        confidence_score_str,
                                        proba_exoplanet_val_str
                                    ])
                                
                                table_string = prediction_table.get_string()
                                logger.info(f"\n{table_string}")
                                print(f"\n{table_string}") # Also print table to console

                            else:
                                logger.error("Prediction demo: Failed to get predictions for the sample.")
                                # print("--- MAIN.PY DEBUG: Prediction demo: Failed to get predictions. ---")
                        else:
                            logger.error("Prediction demo: Could not process sample data for prediction.")
                            # print("--- MAIN.PY DEBUG: Prediction demo: Sample preprocessing failed. ---")
                    else:
                        logger.warning("Prediction demo: No sample data to process.")
                        # print("--- MAIN.PY DEBUG: Prediction demo: No sample data selected. ---")
                else:
                    logger.warning("Prediction demo: y_test is empty or None, cannot select samples for demo.")
                    # print("--- MAIN.PY DEBUG: Prediction demo: y_test is empty/None. ---")
            elif not model:
                logger.error("Prediction demo: Model is None.")
                # print("--- MAIN.PY DEBUG: Prediction demo: Model is None. ---")
            elif not valid_demo_artifacts:
                logger.error("Prediction demo: Could not load essential preprocessing artifacts for the demo.")
                # print("--- MAIN.PY DEBUG: Prediction demo: Invalid artifacts. ---")
        else:
            logger.info("Skipping prediction demonstration as per configuration.")
            # print("--- MAIN.PY DEBUG: Skipping prediction demo. ---")
        
        # print(f"--- MAIN.PY DEBUG: End of 'with mlflow.start_run()' block for run ID {run_id} ---")

    # print(f"--- MAIN.PY DEBUG: PIPELINE END (after 'with mlflow.start_run()' block) ---")


if __name__ == '__main__':
    main_hydra()