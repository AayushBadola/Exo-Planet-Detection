import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from omegaconf import DictConfig
import logging
import joblib
import os

logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def engineer_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    df_eng = df.copy()
    if 'koi_duration' in df_eng.columns and 'koi_period' in df_eng.columns and df_eng['koi_period'].gt(0).all():
        pass
    return df_eng

def preprocess_data(df: pd.DataFrame, cfg: DictConfig, save_artifacts: bool = True):
    if df is None:
        logger.error("Input DataFrame is None. Cannot preprocess.")
        return None, None, {}

    logger.info("Starting preprocessing...")
    df_processed = df.copy()
    df_processed = clean_column_names(df_processed)

    if cfg.preprocessing.target_column not in df_processed.columns:
        logger.error(f"Target column '{cfg.preprocessing.target_column}' not found.")
        return None, None, {}

    df_processed["target"] = df_processed[cfg.preprocessing.target_column].apply(
        lambda x: 1 if x in cfg.preprocessing.positive_labels else (0 if x == cfg.preprocessing.negative_label else -1)
    )
    df_processed = df_processed[df_processed["target"] != -1]
    if df_processed.empty:
        logger.error("No valid target labels found after mapping. Check config.")
        return None, None, {}
    
    y = df_processed["target"]
    X = df_processed.drop(columns=["target", cfg.preprocessing.target_column], errors='ignore')

    features_to_drop_from_cfg = list(cfg.preprocessing.features_to_drop)
    
    if cfg.preprocessing.target_column in features_to_drop_from_cfg:
        features_to_drop_from_cfg.remove(cfg.preprocessing.target_column)
        
    features_to_drop_existing = [col for col in features_to_drop_from_cfg if col in X.columns]
    X = X.drop(columns=features_to_drop_existing, errors='ignore')
    logger.info(f"Dropped columns (from config): {features_to_drop_existing}")

    X = engineer_features(X, cfg)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    if categorical_features:
        logger.warning(f"Found categorical features: {categorical_features}. These will be dropped as per current strategy.")
        X = X.drop(columns=categorical_features, errors='ignore')
        numerical_features = [col for col in numerical_features if col in X.columns]

    transformers = {}

    if numerical_features:
        all_nan_cols = [col for col in numerical_features if X[col].isnull().all()]
        if all_nan_cols:
            logger.warning(f"Numerical columns entirely NaN and will be dropped: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols, errors='ignore')
            numerical_features = [col for col in numerical_features if col not in all_nan_cols]
    
    if not numerical_features:
        logger.error("No numerical features remaining after dropping all-NaN columns or initial selection. Cannot proceed.")
        return None, None, {}

    imputer = SimpleImputer(strategy="median")
    if X[numerical_features].isnull().sum().sum() > 0:
        logger.info(f"Imputing {X[numerical_features].isnull().sum().sum()} missing values in numerical features using median.")
        X_imputed_data = imputer.fit_transform(X[numerical_features])
    else:
        logger.info("No missing values to impute in numerical features. Fitting imputer for consistency.")
        X_imputed_data = imputer.fit_transform(X[numerical_features])

    X_imputed_df = pd.DataFrame(X_imputed_data, columns=numerical_features, index=X.index)
    
    X_non_numeric = X.select_dtypes(exclude=np.number)
    X = pd.concat([X_non_numeric, X_imputed_df], axis=1)
    X = X[X_non_numeric.columns.tolist() + numerical_features]

    transformers['imputer'] = imputer
    if save_artifacts:
        joblib.dump(imputer, cfg.imputer_path)
        logger.info(f"Imputer saved to {cfg.imputer_path}")

    if cfg.preprocessing.get('apply_polynomial_features', False):
        logger.info(f"Applying PolynomialFeatures with degree {cfg.preprocessing.polynomial_degree} to numerical features.")
        poly = PolynomialFeatures(degree=cfg.preprocessing.polynomial_degree, include_bias=False, interaction_only=False)
        
        poly_target_cols = [
            col for col in numerical_features 
            if not col.startswith('koi_fpflag_') and not col.startswith('koi_fittype')
        ]
        if poly_target_cols:
            X_poly_data = poly.fit_transform(X[poly_target_cols])
            poly_feature_names = poly.get_feature_names_out(poly_target_cols)
            X_poly_df = pd.DataFrame(X_poly_data, columns=poly_feature_names, index=X.index)

            X = X.drop(columns=poly_target_cols, errors='ignore')
            X = pd.concat([X, X_poly_df], axis=1)
            
            transformers['poly_features'] = poly
            if save_artifacts:
                joblib.dump(poly, cfg.poly_features_path)
                logger.info(f"PolynomialFeatures transformer saved to {cfg.poly_features_path}")
            
            numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        else:
            logger.warning("No suitable columns found for PolynomialFeatures, or list was empty. Skipping.")

    if not numerical_features or X[numerical_features].empty:
        logger.error("No numerical features available for scaling. Cannot proceed.")
        return None, None, transformers

    scaler = StandardScaler()
    X_scaled_data = scaler.fit_transform(X[numerical_features])
    X_scaled_df = pd.DataFrame(X_scaled_data, columns=numerical_features, index=X.index)
    
    X_non_numeric_after_poly = X.select_dtypes(exclude=np.number)
    X = pd.concat([X_non_numeric_after_poly, X_scaled_df], axis=1)
    if not X_non_numeric_after_poly.empty:
         X = X[X_non_numeric_after_poly.columns.tolist() + numerical_features]
    else:
         X = X[numerical_features]

    transformers['scaler'] = scaler
    if save_artifacts:
        joblib.dump(scaler, cfg.scaler_path)
        logger.info(f"Scaler saved to {cfg.scaler_path}")

    if save_artifacts:
        joblib.dump(list(X.columns), cfg.training_columns_path)
        logger.info(f"Training columns list saved to {cfg.training_columns_path}")

    logger.info(f"Preprocessing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y, transformers

def split_data(X: pd.DataFrame, y: pd.Series, cfg: DictConfig):
    if X is None or y is None or X.empty or y.empty:
        logger.error("Cannot split data: X or y is None or empty.")
        return None, None, None, None
    
    if X.isnull().sum().sum() > 0:
        logger.warning(f"NaNs found in X ({X.isnull().sum().sum()} values) before splitting. This might indicate an issue in earlier preprocessing.")
    
        nan_rows_X = X.isnull().any(axis=1)
        if nan_rows_X.any():
            logger.warning(f"Dropping {nan_rows_X.sum()} rows from X and y due to NaNs in X before split.")
            X = X[~nan_rows_X]
            y = y[~nan_rows_X]
            if X.empty:
                logger.error("X became empty after dropping NaN rows. Cannot split.")
                return None, None, None, None

    if y.isnull().any() or np.isinf(y).any():
        logger.warning("NaNs or Infs found in y before splitting. Cleaning y.")
        y_is_finite = np.isfinite(y)
        y_original_len = len(y)
        X = X.loc[y_is_finite]
        y = y[y_is_finite]
        if len(y) < y_original_len:
            logger.info(f"Dropped {y_original_len - len(y)} rows due to non-finite y values.")
        if y.empty:
            logger.error("y became empty after cleaning non-finite values. Cannot split.")
            return None, None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=cfg.test_size, 
            random_state=cfg.random_state,
            stratify=y if y.nunique() > 1 else None
        )
        logger.info(f"Data split into train and test sets. X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except ValueError as e:
        logger.error(f"Error during train_test_split (e.g., not enough samples for a class for stratify): {e}", exc_info=True)
        logger.info(f"y value counts: {y.value_counts()}")
        logger.warning("Attempting split without stratification due to previous error.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_size, random_state=cfg.random_state
            )
            logger.info(f"Data split (no stratify). X_train: {X_train.shape}, X_test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e_fallback:
            logger.error(f"Fallback split also failed: {e_fallback}", exc_info=True)
            return None, None, None, None

if __name__ == '__main__':
    pass
