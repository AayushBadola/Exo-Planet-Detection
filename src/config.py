import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "cumulative.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = "exoplanet_model.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

TARGET_COLUMN = "koi_disposition"
POSITIVE_LABELS = ["CONFIRMED", "CANDIDATE"]
NEGATIVE_LABEL = "FALSE POSITIVE"

FEATURES_TO_DROP = [
    "rowid", "kepid", "kepoi_name", "kepler_name", "koi_disposition",
    "koi_pdisposition", "koi_score", "koi_tce_delivname",
    "koi_comment",
    "koi_sparprov",
]

MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
