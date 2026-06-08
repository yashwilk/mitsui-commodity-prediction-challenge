from pathlib import Path

# PATHS
# ============================================================

ROOT_DIR   = Path(__file__).resolve().parent
SRC_DIR    = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR   = ROOT_DIR / "logs"
ASSETS_DIR = ROOT_DIR / "assets"

# auto-create directories if they don't exist
for _dir in [MODELS_DIR, LOGS_DIR, ASSETS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# input files 
TRAIN_FILE     = ROOT_DIR / "train.csv"
TEST_FILE      = ROOT_DIR / "test.csv"
LABEL_FILE     = ROOT_DIR / "train_labels.csv"
PAIRS_FILE     = ROOT_DIR / "target_pairs.csv"
LAG_LABELS_DIR = ROOT_DIR / "lagged_test_labels"


# output files
LGBM_PREDICTIONS_FILE     = MODELS_DIR / "lgbm_predictions.csv"
STACKING_PREDICTIONS_FILE = MODELS_DIR / "stacking_predictions.csv"
LGBM_MODELS_FILE          = MODELS_DIR / "lgbm_models.pkl"
STACKING_MODELS_FILE      = MODELS_DIR / "stacking_models.pkl"
TRANSFORMER_MODELS_FILE      = MODELS_DIR / "transformer_models.pkl"
TRANSFORMER_PREDICTIONS_FILE = MODELS_DIR / "transformer_predictions.csv"


# MLFLOW
# ============================================================
 
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT   = "mitsui-commodity-prediction"
 


# FEATURE ENGINEERING
# ============================================================
 
LOG_RETURN_WINDOWS = [1, 3, 5, 10]   # log return horizons (days)
ROLLING_WINDOWS    = [5, 10, 20]     # rolling mean/std windows (days)
PRICE_DIFF_WINDOWS = [1, 3, 5]       # raw price diff horizons (days)
PRICE_LAG_WINDOWS  = [1, 3, 5]       # lagged price levels (days)
SPREAD_WINDOWS     = [1, 5]          # spread features for two-asset targets



# PREPROCESSING
# ============================================================
 
IMPUTER_STRATEGY  = "median"
SCALER_TYPE       = "standard"
MIN_TRAIN_SAMPLES = 50    


# LIGHTGBM HYPERPARAMETERS
# ============================================================
 
LGBM_PARAMS = {
    "n_estimators"    : 100,
    "learning_rate"   : 0.05,
    "max_depth"       : -1,
    "num_leaves"      : 31,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "random_state"    : 42,
    "verbose"         : -1,
}
 
# RANDOM FOREST HYPERPARAMETERS
# ============================================================
 
RF_PARAMS = {
    "n_estimators"    : 100,
    "max_depth"       : 6,
    "min_samples_leaf": 5,
    "random_state"    : 42,
    "n_jobs"          : -1,
}


# XGBOOST BASE LEARNER HYPERPARAMETERS
# ============================================================
 
XGB_PARAMS = {
    "n_estimators"    : 100,
    "learning_rate"   : 0.05,
    "max_depth"       : 4,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "random_state"    : 42,
    "verbosity"       : 0,
}



# XGBOOST META-MODEL HYPERPARAMETERS
# ============================================================
 
XGB_META_PARAMS = {
    "n_estimators" : 50,
    "learning_rate": 0.05,
    "max_depth"    : 3,
    "random_state" : 42,
    "verbosity"    : 0,
}

# TRANSFORMER HYPERPARAMETERS
# ============================================================

TRANSFORMER_PARAMS={
    "d_model":64,# embedding dimension
    "n_heads":8,
    "n_layers":4, # number of encoder layers
    "d_ff":256, # feedforward dimension inside each encoder block
    "dropout":0.1,
    "learning_rate":1e-4,
    "weight_decay":1e-4,   # AdamW weight decay (L2 regularisation)
    "n_epochs":50,
    "batch_size":64

}



# CROSS-VALIDATION
# ============================================================
 
CV_N_SPLITS    = 5
CV_RANDOM_SEED = 42



# EVALUATION
# ============================================================
 
LAGS = [1, 2, 3, 4]
 
LAG_LABEL_FILES = {
    lag: LAG_LABELS_DIR / f"test_labels_lag_{lag}.csv"
    for lag in LAGS
}

# LOGGING
# ============================================================
 
LOG_LEVEL       = "INFO"
LOG_FILE        = LOGS_DIR / "training.log"
LOG_FORMAT      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
 