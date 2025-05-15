import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from omegaconf import DictConfig
import logging
import mlflow
import re # Import re for sanitization

logger = logging.getLogger(__name__)

def sanitize_metric_name(name: str) -> str:
    """
    Sanitizes a metric name for MLflow logging.
    Replaces disallowed characters with underscores.
    Allowed: alphanumerics, underscores, dashes, periods, spaces, slashes.
    """
    # Replace parentheses and other problematic characters with underscore
    name = re.sub(r'[()\s/]', '_', name) # Replace (), space, / with _
    name = re.sub(r'[^a-zA-Z0-9_.-]', '', name) # Remove any other invalid chars, keep . and -
    name = re.sub(r'__+', '_', name) # Replace multiple underscores with single
    name = name.strip('_') # Remove leading/trailing underscores
    return name

def get_model_instance(cfg: DictConfig):
    # ... (rest of the function is the same)
    model_name = cfg.model.name
    model_params = dict(cfg.model.params) 
    
    if 'random_state' not in model_params and hasattr(RandomForestClassifier(), 'random_state'): 
         model_params['random_state'] = cfg.random_state

    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(**model_params)
    else:
        logger.error(f"Unsupported model name in config: {model_name}")
        raise ValueError(f"Unsupported model name: {model_name}")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig):
    # ... (rest of the function is the same)
    if X_train is None or y_train is None or X_train.empty or y_train.empty:
        logger.error("Training data (X_train or y_train) is None or empty. Skipping model training.")
        return None

    logger.info(f"Starting model training with {cfg.model.name}...")
    
    serializable_model_params = {k: v for k, v in dict(cfg.model.params).items()}
    mlflow.log_params(serializable_model_params)
    mlflow.log_param("model_name", cfg.model.name)

    model = get_model_instance(cfg)
    
    try:
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return None

    try:
        # In config_utils, model_path is already absolute
        joblib.dump(model, cfg.model_path) 
        logger.info(f"Model saved to {cfg.model_path}")
        mlflow.log_artifact(cfg.model_path, artifact_path="model_files") 
    except Exception as e:
        logger.error(f"Error saving model to {cfg.model_path}: {e}", exc_info=True)
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, cfg: DictConfig):
    if model is None or X_test is None or y_test is None or X_test.empty or y_test.empty:
        logger.error("Model or test data is None/empty. Skipping evaluation.")
        return {}

    logger.info("\nEvaluating model...")
    metrics_dict = {}
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] 

        metrics_dict["accuracy"] = accuracy_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {metrics_dict['accuracy']:.4f}")

        report_dict = classification_report(y_test, y_pred, target_names=['Not Exoplanet (0)', 'Exoplanet (1)'], output_dict=True)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Not Exoplanet (0)', 'Exoplanet (1)']))
        
        for label, scores in report_dict.items():
            if isinstance(scores, dict): 
                for metric_name, value in scores.items():
                    # SANITIZE HERE
                    sanitized_key = sanitize_metric_name(f"{label}_{metric_name}")
                    metrics_dict[sanitized_key] = value
            elif label in ["accuracy"]: # Special handling for top-level accuracy if not already captured
                if label not in metrics_dict: # Avoid overwriting if already there
                    metrics_dict[sanitize_metric_name(label)] = scores # scores is a float here
            # For 'macro avg' and 'weighted avg' which are dicts themselves
            elif isinstance(report_dict[label], dict): 
                 for metric_name, value in report_dict[label].items():
                      sanitized_key = sanitize_metric_name(f"{label}_{metric_name}")
                      metrics_dict[sanitized_key] = value


        metrics_dict["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"ROC AUC Score: {metrics_dict['roc_auc']:.4f}")

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics_dict["pr_auc"] = auc(recall, precision) 
        logger.info(f"Precision-Recall AUC Score: {metrics_dict['pr_auc']:.4f}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{conf_matrix}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Exoplanet (0)', 'Exoplanet (1)'],
                    yticklabels=['Not Exoplanet (0)', 'Exoplanet (1)'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {cfg.model.name}')
        
        # In config_utils, reports_dir is already absolute
        cm_plot_path = os.path.join(cfg.reports_dir, "confusion_matrix.png") 
        try:
            plt.savefig(cm_plot_path)
            plt.close() 
            logger.info(f"Confusion matrix plot saved to {cm_plot_path}")
            mlflow.log_artifact(cm_plot_path, artifact_path="evaluation_plots")
        except Exception as e:
            logger.error(f"Could not save confusion matrix plot: {e}", exc_info=True)
            plt.close()

        return metrics_dict

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return metrics_dict 


def get_feature_importances(model, feature_names: list, cfg: DictConfig, top_n=20):
    # ... (rest of the function is the same)
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info(f"\nTop {top_n} Feature Importances ({cfg.model.name}):")
    df_importances = pd.DataFrame({
        'feature': [feature_names[i] for i in indices[:top_n]],
        'importance': importances[indices][:top_n]
    })
    logger.info(f"\n{df_importances.to_string()}") 

    plt.figure(figsize=(12, max(7, top_n * 0.45))) 
    sns.barplot(x='importance', y='feature', data=df_importances, hue='feature', palette='viridis', legend=False, dodge=False)
    plt.title(f'Top {top_n} Feature Importances - {cfg.model.name}')
    plt.tight_layout()

    # In config_utils, reports_dir is already absolute
    fi_plot_path = os.path.join(cfg.reports_dir, "feature_importances.png") 
    try:
        plt.savefig(fi_plot_path)
        plt.close()
        logger.info(f"Feature importances plot saved to {fi_plot_path}")
    except Exception as e:
        logger.error(f"Could not save feature importances plot: {e}", exc_info=True)
        plt.close()
    
    return df_importances

if __name__ == '__main__':
    pass