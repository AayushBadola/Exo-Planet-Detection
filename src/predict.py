import joblib
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import logging
import os

logger = logging.getLogger(__name__)

def load_trained_model(cfg: DictConfig):
    """Loads the trained model from the path specified in config."""
    model_path = cfg.model_path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None

def load_preprocessing_artifacts(cfg: DictConfig) -> dict:
    """Loads all necessary preprocessing artifacts."""
    artifacts = {}
    paths_to_load = {
        'scaler': cfg.scaler_path,
        'imputer': cfg.imputer_path,
        'training_columns': cfg.training_columns_path
    }
    if cfg.preprocessing.get('apply_polynomial_features', False):
         paths_to_load['poly_features'] = cfg.poly_features_path


    for name, path in paths_to_load.items():
        if path and os.path.exists(path):
            try:
                artifacts[name] = joblib.load(path)
                logger.info(f"Loaded '{name}' from {path}")
            except Exception as e:
                logger.error(f"Error loading '{name}' from {path}: {e}", exc_info=True)
                artifacts[name] = None # Mark as None if loading failed
        elif path: # Path specified but does not exist
            logger.warning(f"Artifact '{name}' not found at specified path: {path}. It might not have been saved (e.g., no NaNs for imputer).")
            artifacts[name] = None
        else: # Path not specified (e.g. imputer path is None)
             artifacts[name] = None
    
    # Critical check for training_columns
    if artifacts.get('training_columns') is None:
        logger.error("Training columns list could not be loaded. Prediction preprocessing will likely fail.")
        # Potentially raise an error or return None for all artifacts
    
    return artifacts


def preprocess_for_prediction(raw_sample_df: pd.DataFrame, artifacts: dict, cfg: DictConfig) -> pd.DataFrame | None:
    """
    Preprocesses raw sample data using loaded artifacts for prediction.
    """
    if raw_sample_df.empty:
        logger.warning("Input raw_sample_df is empty for prediction preprocessing.")
        return None
    if not artifacts or artifacts.get('training_columns') is None or artifacts.get('scaler') is None:
        logger.error("Essential preprocessing artifacts (training_columns, scaler) are missing. Cannot preprocess for prediction.")
        return None

    logger.info("Starting preprocessing for prediction...")
    df_pred = raw_sample_df.copy()
    
    # Import preprocessor locally to use clean_column_names
    # This creates a slight circular dependency if preprocessor imports predict, avoid if possible.
    # Better: move clean_column_names to a utility module or duplicate if small.
    # For now, let's assume it's fine for this structure or clean_column_names is made independent.
    from .preprocessor import clean_column_names # Assuming same level src import
    df_pred = clean_column_names(df_pred)

    # --- Feature Dropping (consistent with training) ---
    # Note: target column is not expected in raw_sample_df for prediction.
    # If it is, it should be removed before this function or handled.
    features_to_drop_from_cfg = list(cfg.preprocessing.features_to_drop)
    cols_to_drop_pred = [
        col for col in features_to_drop_from_cfg
        if col in df_pred.columns and col != cfg.preprocessing.target_column # Exclude target from this list
    ]
    if cfg.preprocessing.target_column in df_pred.columns: # Explicitly drop if present
        cols_to_drop_pred.append(cfg.preprocessing.target_column)

    df_pred = df_pred.drop(columns=cols_to_drop_pred, errors='ignore')
    logger.debug(f"Prediction preproc: Dropped columns (from config): {cols_to_drop_pred}")


    # --- Feature Engineering (consistent with training) ---
    # from .preprocessor import engineer_features # Similar import consideration
    # df_pred = engineer_features(df_pred, cfg) # Assuming engineer_features is stateless or uses no fitted params


    # --- Identify Feature Types (on the current sample) ---
    numerical_features_pred = df_pred.select_dtypes(include=np.number).columns.tolist()
    categorical_features_pred = df_pred.select_dtypes(include='object').columns.tolist()

    if categorical_features_pred:
        logger.debug(f"Prediction preproc: Dropping categorical features from sample: {categorical_features_pred}")
        df_pred = df_pred.drop(columns=categorical_features_pred, errors='ignore')
        numerical_features_pred = [col for col in numerical_features_pred if col in df_pred.columns]

    # --- Handle All-NaN Columns in sample (consistent with training) ---
    if numerical_features_pred:
        all_nan_cols_sample = [col for col in numerical_features_pred if df_pred[col].isnull().all()]
        if all_nan_cols_sample:
            logger.debug(f"Prediction preproc: Dropping entirely NaN columns from sample: {all_nan_cols_sample}")
            df_pred = df_pred.drop(columns=all_nan_cols_sample, errors='ignore')
            numerical_features_pred = [col for col in numerical_features_pred if col not in all_nan_cols_sample]

    if not numerical_features_pred:
        logger.warning("Prediction preproc: No numerical features left in sample after initial drops. Final alignment will try to fill.")
        # Fall through to final alignment which will create 0-filled columns

    # --- Imputation (using loaded imputer) ---
    imputer = artifacts.get('imputer')
    if imputer and numerical_features_pred: # Check if imputer exists and there are num_features
        # Imputer was fit on a specific set of numerical columns during training.
        # We need to apply it to the same columns if they exist in the current sample.
        # SimpleImputer is typically fitted on all numerical columns passed to it.
        # So, `imputer.feature_names_in_` (if available) or by assuming it was fit on all `numerical_features` from training.
        
        # The columns the imputer expects (based on training time `numerical_features` list)
        # For SimpleImputer, it doesn't store names by default unless in a Pipeline.
        # We assume it was fit on the set of numerical columns that existed *before* polynomial features and scaling.
        # This is a bit fragile. A more robust way is to save the list of columns the imputer was fit on.
        # For now, we apply to current `numerical_features_pred`.
        cols_to_impute_in_sample = [col for col in numerical_features_pred if col in df_pred.columns and df_pred[col].isnull().any()]

        if cols_to_impute_in_sample:
            logger.debug(f"Prediction preproc: Imputing missing values in sample columns: {cols_to_impute_in_sample}")
            imputed_data_sample = imputer.transform(df_pred[cols_to_impute_in_sample])
            imputed_df_sample = pd.DataFrame(imputed_data_sample, columns=cols_to_impute_in_sample, index=df_pred.index)
            
            df_pred.drop(columns=cols_to_impute_in_sample, inplace=True)
            df_pred = pd.concat([df_pred, imputed_df_sample], axis=1)
        else:
            logger.debug("Prediction preproc: No NaNs in numerical features of sample or no numerical features to impute.")
    elif numerical_features_pred and df_pred[numerical_features_pred].isnull().sum().sum() > 0:
        logger.warning("Prediction preproc: Sample has NaNs in numerical features, but no imputer artifact loaded. NaNs will remain.")


    # --- Polynomial Features (using loaded transformer) ---
    poly_transformer = artifacts.get('poly_features')
    if poly_transformer and cfg.preprocessing.get('apply_polynomial_features', False):
        # Need to identify the same columns polynomial features were applied to during training
        # This is also fragile without saving the exact list of columns.
        # Assuming poly_target_cols selection logic from preprocessor.py can be replicated or was saved.
        # For this example, let's assume `poly_transformer.feature_names_in_` if sklearn version is new enough,
        # or we try to apply to the columns present in the sample that were likely transformed.
        
        # Heuristic: apply to current numerical columns, excluding flags
        poly_candidate_cols_sample = [
            col for col in df_pred.select_dtypes(include=np.number).columns.tolist()
            if not col.startswith('koi_fpflag_') and not col.startswith('koi_fittype')
            and col in df_pred.columns # ensure col exists
        ]

        # Further ensure these cols were likely what poly was fit on
        # This needs `poly_transformer.feature_names_in_` or a saved list of columns
        # For now, if poly_transformer.feature_names_in_ exists:
        if hasattr(poly_transformer, 'feature_names_in_'):
            poly_input_cols_from_transformer = list(poly_transformer.feature_names_in_)
            actual_poly_cols_in_sample = [col for col in poly_candidate_cols_sample if col in poly_input_cols_from_transformer and col in df_pred.columns]
        else: # Fallback if feature_names_in_ not available (older sklearn)
            actual_poly_cols_in_sample = [col for col in poly_candidate_cols_sample if col in df_pred.columns]


        if actual_poly_cols_in_sample:
            logger.debug(f"Prediction preproc: Applying PolynomialFeatures to sample columns: {actual_poly_cols_in_sample}")
            # Ensure all columns are numeric and no NaNs before transform
            for col in actual_poly_cols_in_sample:
                if df_pred[col].isnull().any():
                    logger.warning(f"NaNs found in column {col} before PolynomialFeatures transform in prediction. Filling with 0.")
                    df_pred[col] = df_pred[col].fillna(0)
                if not pd.api.types.is_numeric_dtype(df_pred[col]):
                     logger.warning(f"Column {col} is not numeric before PolynomialFeatures transform. Attempting conversion or filling with 0.")
                     try:
                        df_pred[col] = pd.to_numeric(df_pred[col])
                     except:
                        df_pred[col] = 0


            poly_data_sample = poly_transformer.transform(df_pred[actual_poly_cols_in_sample])
            poly_feature_names_sample = poly_transformer.get_feature_names_out(actual_poly_cols_in_sample)
            poly_df_sample = pd.DataFrame(poly_data_sample, columns=poly_feature_names_sample, index=df_pred.index)

            df_pred = df_pred.drop(columns=actual_poly_cols_in_sample, errors='ignore')
            df_pred = pd.concat([df_pred, poly_df_sample], axis=1)
        else:
            logger.debug("Prediction preproc: No suitable columns for PolynomialFeatures in sample or transformer not loaded.")

    # --- Scaling (using loaded scaler) ---
    scaler = artifacts.get('scaler')
    training_columns = artifacts.get('training_columns') # These are the columns the model expects (post-all-preprocessing)

    # The scaler was fit on a specific set of numerical features (which became part of training_columns).
    # We need to prepare a DataFrame for the scaler that has these exact columns.
    
    temp_df_for_scaling = pd.DataFrame(index=df_pred.index)
    scaled_cols_present_in_sample = []

    # Scaler expects columns it was fit on. These would be the numerical subset of `training_columns`
    # *before* scaling itself was applied in the training `preprocess_data`.
    # This is the trickiest part to get right without saving explicit column lists for each transformer step.
    # For robust scaling: `training_columns` IS THE FINAL LIST OF COLUMNS THE MODEL EXPECTS.
    # The scaler was fit on a subset of these (the numerical ones).
    # So, `scaler.transform` should be applied to the numerical columns *that are part of `training_columns`*.
    
    # Let's assume scaler was fit on *all* columns that ended up in `training_columns` if they were numeric
    # at the point of scaling during training. `training_columns` are post-scaling.
    # A better way: scaler should be fit on pre-poly, pre-scaling numerical features, and that list saved.
    # Simpler for now: reconstruct or apply to numerical columns within `training_columns`.

    # Identify numerical columns from the `training_columns` list that scaler was likely fit on.
    # This assumes `training_columns` reflects the state *after* scaling.
    # The scaler was fit on the numeric columns that were *inputs* to it.
    # We will ensure `df_pred` has all `training_columns`, fill missing, then scale those among them that are numeric.

    # Ensure df_pred has all training_columns, filling missing ones with 0.
    # This makes df_pred have the same structure as X was just before model.fit().
    for col in training_columns:
        if col not in df_pred.columns:
            logger.debug(f"Prediction preproc: Column '{col}' (expected by model/scaler) missing in sample. Filling with 0 before scaling.")
            df_pred[col] = 0
        # Ensure numeric type for columns that should be scaled (all training_columns for safety)
        if not pd.api.types.is_numeric_dtype(df_pred[col]):
            try:
                df_pred[col] = pd.to_numeric(df_pred[col])
            except ValueError:
                logger.warning(f"Prediction preproc: Could not convert column {col} to numeric for scaling. Filling with 0.")
                df_pred[col] = 0
        
        if df_pred[col].isnull().any(): # Final NaN check for safety
            logger.warning(f"Prediction preproc: NaNs found in column {col} before scaling. Filling with 0.")
            df_pred[col] = df_pred[col].fillna(0)


    # Now `df_pred` should have all `training_columns`. Apply scaler.
    # The `scaler` was fit on the numerical features *before* they were scaled.
    # So we need to apply `scaler.transform` to those columns in `df_pred` that correspond to scaler's input.
    # `scaler.feature_names_in_` is ideal if available.
    cols_to_scale_in_sample = []
    if hasattr(scaler, 'feature_names_in_'):
        cols_to_scale_in_sample = [col for col in scaler.feature_names_in_ if col in df_pred.columns]
    else: # Fallback: assume scaler was fit on all numerical columns present in `training_columns`
          # This is an approximation.
        cols_to_scale_in_sample = [col for col in training_columns if pd.api.types.is_numeric_dtype(df_pred[col])]


    if scaler and cols_to_scale_in_sample:
        logger.debug(f"Prediction preproc: Scaling sample columns: {cols_to_scale_in_sample}")
        scaled_data_sample = scaler.transform(df_pred[cols_to_scale_in_sample])
        scaled_df_sample = pd.DataFrame(scaled_data_sample, columns=cols_to_scale_in_sample, index=df_pred.index)
        
        # Update df_pred with scaled columns
        for col in cols_to_scale_in_sample: # Iterate to update specific columns
            df_pred[col] = scaled_df_sample[col]
    elif not scaler:
        logger.warning("Prediction preproc: Scaler artifact not loaded. Skipping scaling.")
    else: # No cols_to_scale_in_sample
        logger.debug("Prediction preproc: No columns identified for scaling in sample.")

    # --- Final Column Alignment (to match model's exact training input) ---
    final_pred_df = pd.DataFrame(index=df_pred.index)
    missing_for_model = []
    for col in training_columns: # `training_columns` is the source of truth for model input
        if col in df_pred.columns:
            final_pred_df[col] = df_pred[col]
        else:
            # This should have been handled by the fill loop above if `training_columns` is correct
            logger.error(f"Prediction preproc: CRITICAL - Column '{col}' still missing after alignment attempts. Filling with 0.")
            final_pred_df[col] = 0
            missing_for_model.append(col)
    
    if missing_for_model:
        logger.error(f"Prediction preproc: Model input features {missing_for_model} were entirely missing and filled with 0.")

    # Ensure correct order and drop any extra columns that might have crept in
    final_pred_df = final_pred_df[training_columns]

    # Final NaN check on the data to be passed to the model
    if final_pred_df.isnull().sum().sum() > 0:
        nan_sum = final_pred_df.isnull().sum().sum()
        logger.error(f"Prediction preproc: CRITICAL - NaNs detected in final preprocessed data for prediction ({nan_sum} values). Model will likely fail.")
        logger.error(f"Columns with NaNs:\n{final_pred_df.isnull().sum()[final_pred_df.isnull().sum() > 0]}")
        # Consider filling with 0 as a last resort or raising an error
        # final_pred_df = final_pred_df.fillna(0) 
        # logger.warning("Filled final NaNs with 0 before prediction. This is a fallback.")
        return None # Fail fast if NaNs remain

    logger.info("Prediction preprocessing complete.")
    return final_pred_df


def make_prediction(model, processed_input_data_df: pd.DataFrame):
    """
    Makes predictions on fully preprocessed input data.
    """
    if model is None:
        logger.error("Model is not loaded for make_prediction.")
        return None, None
    if processed_input_data_df is None or processed_input_data_df.empty:
        logger.error("Input data for make_prediction is None or empty.")
        return None, None

    logger.info("Making predictions...")
    # Data is assumed to be fully preprocessed and aligned by `preprocess_for_prediction`

    # Double-check alignment if model has feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        if list(processed_input_data_df.columns) != expected_features:
            logger.warning("Column mismatch between processed data and model.feature_names_in_ before prediction. Re-aligning.")
            # This re-alignment is a safety net, ideally preprocess_for_prediction handles it.
            temp_aligned_df = pd.DataFrame(index=processed_input_data_df.index)
            for col in expected_features:
                if col in processed_input_data_df.columns:
                    temp_aligned_df[col] = processed_input_data_df[col]
                else:
                    logger.error(f"CRITICAL: Expected feature '{col}' by model not in preprocessed data for prediction. Filling with 0.")
                    temp_aligned_df[col] = 0
            processed_input_data_df = temp_aligned_df[expected_features]


    try:
        predictions = model.predict(processed_input_data_df)
        probabilities = model.predict_proba(processed_input_data_df)
        logger.info("Predictions generated successfully.")
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Error during model.predict/predict_proba: {e}", exc_info=True)
        logger.error(f"Shape of data passed to model: {processed_input_data_df.shape}")
        logger.error(f"Columns of data passed to model: {processed_input_data_df.columns.tolist()}")
        logger.error(f"NaN check in data passed to model:\n{processed_input_data_df.isnull().sum()[processed_input_data_df.isnull().sum() > 0]}")
        return None, None

if __name__ == '__main__':
    # Example usage - requires a dummy config and saved artifacts
    pass