import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def get_absolute_path(cfg: DictConfig, relative_path_key: str) -> str:
    # cfg.base_dir is set in main.py from get_original_cwd()
    # cfg[relative_path_key] is like "data_dir_rel" which resolves to "data" from config.yaml
    return os.path.join(cfg.base_dir, cfg[relative_path_key])

def setup_config(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False) # Allow adding/modifying keys

    if not hasattr(cfg, 'base_dir') or cfg.base_dir == "???":
        raise ValueError("cfg.base_dir must be set before calling setup_config in main.py.")

    # Resolve main directory paths defined in config.yaml (e.g., data_dir, model_dir)
    # These keys (data_dir, model_dir etc.) are already in cfg from config.yaml, holding "???"
    # We update them to absolute paths.
    cfg.data_dir = get_absolute_path(cfg, "data_dir_rel")
    cfg.model_dir = get_absolute_path(cfg, "model_dir_rel")
    cfg.reports_dir = get_absolute_path(cfg, "reports_dir_rel")
    cfg.log_file = get_absolute_path(cfg, "log_file_rel")

    # Create directories
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.reports_dir, exist_ok=True)
    log_dir_path = os.path.dirname(cfg.log_file)
    if log_dir_path: # Ensure log_dir_path is not empty (e.g. if log_file is in current dir)
        os.makedirs(log_dir_path, exist_ok=True)
    
    # --- Path to the raw data file itself ---
    # This depends on the 'data_paths' group being loaded.
    if 'data_paths' not in cfg:
        # This is the original error point.
        # If this error still occurs, the issue is purely with Hydra loading data_paths group.
        raise KeyError("The 'data_paths' group was not found in the configuration. Ensure 'conf/data_paths/default.yaml' exists, starts with '# @package _group_', and is correctly referenced in config.yaml defaults.")
    
    # cfg.data_paths.raw_data_filename comes from conf/data_paths/default.yaml
    # We create a new key 'raw_data_file_abs' under data_paths group for the absolute path.
    # This key will be used by main.py to load data.
    cfg.data_paths.raw_data_file_abs = os.path.join(cfg.data_dir, cfg.data_paths.raw_data_filename)

    # --- Model and artifact paths ---
    # These keys (model_path, scaler_path etc.) are in cfg from config.yaml, holding "???" or null
    # We update them to their absolute paths. These are used by preprocessor.py, model_trainer.py, predict.py
    cfg.model_path = os.path.join(cfg.model_dir, f"{cfg.model.name.lower()}_model.joblib") # cfg.model.name from model group
    cfg.scaler_path = os.path.join(cfg.model_dir, "scaler.joblib")
    cfg.imputer_path = os.path.join(cfg.model_dir, "imputer.joblib")
    cfg.training_columns_path = os.path.join(cfg.model_dir, "training_columns.joblib")
    
    # cfg.preprocessing comes from preprocessing group.
    # Use .get() for 'apply_polynomial_features' for robustness, though it's defined in your default.yaml.
    if cfg.preprocessing.get('apply_polynomial_features', False):
        cfg.poly_features_path = os.path.join(cfg.model_dir, "poly_features_transformer.joblib")
    else:
        # config.yaml already defaults poly_features_path to null.
        # Explicitly setting to None ensures it if the default was different or key was missing.
        cfg.poly_features_path = None 

    # OmegaConf.set_struct(cfg, True) # Optional: make read-only again
    return cfg