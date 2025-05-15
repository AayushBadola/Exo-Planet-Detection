import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def get_absolute_path(cfg: DictConfig, relative_path_key: str) -> str:
   
    return os.path.join(cfg.base_dir, cfg[relative_path_key])

def setup_config(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    if not hasattr(cfg, 'base_dir') or cfg.base_dir == "???":
        raise ValueError("cfg.base_dir must be set before calling setup_config in main.py.")

   
    cfg.data_dir = get_absolute_path(cfg, "data_dir_rel")
    cfg.model_dir = get_absolute_path(cfg, "model_dir_rel")
    cfg.reports_dir = get_absolute_path(cfg, "reports_dir_rel")
    cfg.log_file = get_absolute_path(cfg, "log_file_rel")

   
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.reports_dir, exist_ok=True)
    log_dir_path = os.path.dirname(cfg.log_file)
    if log_dir_path: 
        os.makedirs(log_dir_path, exist_ok=True)
    
   
    if 'data_paths' not in cfg:
       
        raise KeyError("The 'data_paths' group was not found in the configuration. Ensure 'conf/data_paths/default.yaml' exists, starts with '# @package _group_', and is correctly referenced in config.yaml defaults.")
    
    
    cfg.data_paths.raw_data_file_abs = os.path.join(cfg.data_dir, cfg.data_paths.raw_data_filename)

   
    cfg.model_path = os.path.join(cfg.model_dir, f"{cfg.model.name.lower()}_model.joblib") 
    cfg.scaler_path = os.path.join(cfg.model_dir, "scaler.joblib")
    cfg.imputer_path = os.path.join(cfg.model_dir, "imputer.joblib")
    cfg.training_columns_path = os.path.join(cfg.model_dir, "training_columns.joblib")
    
   
    if cfg.preprocessing.get('apply_polynomial_features', False):
        cfg.poly_features_path = os.path.join(cfg.model_dir, "poly_features_transformer.joblib")
    else:
       
        cfg.poly_features_path = None 

   
    return cfg