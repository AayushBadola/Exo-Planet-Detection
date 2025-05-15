import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import joblib
from prettytable import PrettyTable

from src import data_loader, preprocessor, model_trainer, predict
from src.config_utils import setup_config
from src.logger_utils import setup_logging

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main_hydra(cfg: DictConfig) -> None:
    cfg.base_dir = get_original_cwd()
    cfg = setup_config(cfg)
    logger = setup_logging(cfg.log_file)

    logger.info("----------------------------------------------------")
    logger.info("--- Starting Exoplanet Prediction Pipeline (Hydra) ---")
    logger.info("----------------------------------------------------")
    logger.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    raw_df = data_loader.load_data(cfg.data_paths.raw_data_file_abs)
    if raw_df is None:
        logger.error("Failed to load data. Exiting pipeline.")
        return
    logger.info(f"Raw data loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns.")

    X, y, fitted_transformers = preprocessor.preprocess_data(raw_df, cfg, save_artifacts=True)
    if X is None or y is None or X.empty or y.empty:
        logger.error("Failed to preprocess data. Exiting pipeline.")
        return
    logger.info(f"Preprocessing complete. Shape of X: {X.shape}, y: {y.shape}. Artifacts saved locally to {cfg.model_dir}")
    logger.info(f"Final feature count: {len(X.columns)}")

    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, cfg)
    if X_train is None or X_train.empty:
        logger.error("Failed to split data or training data is empty. Exiting pipeline.")
        return
    logger.info(f"Data split complete. X_train: {X_train.shape}, y_train: {y_train.shape}")

    model = model_trainer.train_model(X_train, y_train, cfg)
    if model is None:
        logger.error("Failed to train model. Exiting pipeline.")
        return
    logger.info(f"Model training complete. Model saved to {cfg.model_path}")
    logger.info(f"Trained model type: {cfg.model.name}")

    evaluation_metrics = model_trainer.evaluate_model(model, X_test, y_test, cfg)
    if evaluation_metrics:
        logger.info(f"Evaluation metrics (local report):")
        for k, v in evaluation_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
        cm_plot_path = os.path.join(cfg.reports_dir, "confusion_matrix.png")
        if os.path.exists(cm_plot_path):
            logger.info(f"Confusion matrix plot saved to: {cm_plot_path}")
        else:
            logger.warning(f"Confusion matrix plot not found at: {cm_plot_path}")
    else:
        logger.warning("No evaluation metrics returned from model_trainer.evaluate_model.")

    if hasattr(model, 'feature_importances_'):
        df_importances = model_trainer.get_feature_importances(model, X_train.columns.tolist(), cfg)
        if df_importances is not None:
            fi_plot_path = os.path.join(cfg.reports_dir, "feature_importances.png")
            if os.path.exists(fi_plot_path):
                logger.info(f"Feature importances plot saved to: {fi_plot_path}")
            else:
                logger.warning(f"Feature importances plot not found at: {fi_plot_path}")
    else:
        logger.info("Model does not have feature_importances_ attribute. Skipping feature importance plot generation.")
    
    logger.info("\n--- Exoplanet Prediction Pipeline Completed Successfully (Local Mode) ---")

    if cfg.get("run_prediction_demo", True):
        logger.info("\n--- Demonstrating Prediction on a Sample (Local Mode) ---")
        
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
                     if not sample_raw_data_for_pred_copy.empty:
                        sample_raw_data_for_pred_copy["target_temp_demo"] = sample_raw_data_for_pred_copy[cfg.preprocessing.target_column].apply(
                            lambda x: 1 if x in cfg.preprocessing.positive_labels else (0 if x == cfg.preprocessing.negative_label else -1)
                        )
                        actual_labels_sample_for_pred = sample_raw_data_for_pred_copy["target_temp_demo"][sample_raw_data_for_pred_copy["target_temp_demo"] != -1].copy()
                        actual_labels_sample_for_pred = actual_labels_sample_for_pred.reindex(sample_raw_data_for_pred.index)

                if not sample_raw_data_for_pred.empty:
                    sample_processed = predict.preprocess_for_prediction(sample_raw_data_for_pred, demo_artifacts_for_pred, cfg)
                    
                    if sample_processed is not None and not sample_processed.empty:
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
                            print(f"\n{table_string}")

                        else:
                            logger.error("Prediction demo: Failed to get predictions for the sample.")
                    else:
                        logger.error("Prediction demo: Could not process sample data for prediction.")
                else:
                    logger.warning("Prediction demo: No sample data to process.")
            elif not model:
                logger.error("Prediction demo: Model is None.")
            elif not valid_demo_artifacts:
                logger.error("Prediction demo: Could not load essential preprocessing artifacts for the demo.")
        else:
            logger.info("Skipping prediction demonstration as per configuration.")

if __name__ == '__main__':
    main_hydra()