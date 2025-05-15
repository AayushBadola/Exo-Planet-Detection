import os

# Define the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This gets the 'exoplanet-detection-kepler' directory

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "cumulative.csv")

# Model saving path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = "exoplanet_model.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Target variable and labels
TARGET_COLUMN = "koi_disposition"
POSITIVE_LABELS = ["CONFIRMED", "CANDIDATE"]
NEGATIVE_LABEL = "FALSE POSITIVE" # Assuming we only care about these three for a binary problem

# Features to drop
# These are identifiers, redundant labels, or columns that might cause leakage or are not useful for general prediction.
# 'koi_score' is a pre-computed score, often dropped when building a model from raw features.
# 'koi_tce_delivname' is a TCE delivery name, an identifier.
# 'kepler_name', 'kepoi_name' are identifiers.
# 'koi_pdisposition' is very similar to target, often dropped.
# Some error columns might be too sparse or less predictive; can be refined during EDA.
# For now, we'll keep most koi_fpflag_ and koi_ [feature] columns.
FEATURES_TO_DROP = [
    "rowid", "kepid", "kepoi_name", "kepler_name", "koi_disposition", # koi_disposition is target, will be separated
    "koi_pdisposition", "koi_score", "koi_tce_delivname",
    # The following are text descriptions or non-predictive in raw form for simple models
    "koi_comment",
    # Error columns for text parameters if any, or IDs
    "koi_sparprov", # Source of stellar parameters
]

# Model parameters (Example for RandomForestClassifier)
MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1, # Use all available cores
    "max_depth": 20, # Example: prevent overfitting
    "min_samples_split": 5, # Example
    "min_samples_leaf": 2 # Example
}

# Train-test split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)