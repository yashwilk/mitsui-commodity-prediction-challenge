import argparse
import logging
import os
import sys
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMRegressor
 
import config
from src.features import parse_pair, build_dataset_for_target
from src.preprocessing import preprocess_features
from src.model import StackingModel
from src.metrics import spearman_sharpe

# LOGGING
# ============================================================
 
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# TRAIN LGBM
# ============================================================
 
def train_lgbm(
    train_df: pd.DataFrame,
    label_df: pd.DataFrame,
    target_info: list,
) -> dict:
    """
    Train one LightGBM model per target (424 total).
    Returns a dict of fitted models keyed by target name.
    """
    models = {}
 
    for t in tqdm(target_info, desc="Training LightGBM models"):
        target_name = t["target"]
        try:
            X, y = build_dataset_for_target(train_df, label_df, t)
 
            if len(y) < config.MIN_TRAIN_SAMPLES:
                logger.warning(
                    "Skipping %s — only %d samples", target_name, len(y)
                )
                continue
 
            X_processed, imputer, scaler = preprocess_features(X)
 
            model = LGBMRegressor(**config.LGBM_PARAMS)
            model.fit(X_processed, y)
 
            models[target_name] = {
                "model"  : model,
                "imputer": imputer,
                "scaler" : scaler,
                "assets" : t["assets"],
                "lag"    : t["lag"],
            }
 
        except Exception as e:
            logger.error("Error training %s: %s", target_name, e)
            continue
 
    logger.info("Trained %d / %d LightGBM models", len(models), len(target_info))
    return models
 
# TRAIN STACKING
# ============================================================
 
def train_stacking(
    train_df: pd.DataFrame,
    label_df: pd.DataFrame,
    target_info: list,
) -> dict:
    """
    Train one StackingModel per target (424 total).
    Returns a dict of fitted models keyed by target name.
    """
    models = {}
 
    for t in tqdm(target_info, desc="Training Stacking models"):
        target_name = t["target"]
        try:
            X, y = build_dataset_for_target(train_df, label_df, t)
 
            if len(y) < config.MIN_TRAIN_SAMPLES:
                logger.warning(
                    "Skipping %s — only %d samples", target_name, len(y)
                )
                continue
 
            X_processed, imputer, scaler = preprocess_features(X)
 
            model = StackingModel(random_state=config.CV_RANDOM_SEED)
            model.fit(X_processed, y)
 
            models[target_name] = {
                "model"  : model,
                "imputer": imputer,
                "scaler" : scaler,
                "assets" : t["assets"],
                "lag"    : t["lag"],
            }
 
        except Exception as e:
            logger.error("Error training %s: %s", target_name, e)
            continue
 
    logger.info("Trained %d / %d Stacking models", len(models), len(target_info))
    return models


def run_training(model_type: str) -> None:
    logger.info("Loading data...")
    train_df    = pd.read_csv(config.TRAIN_FILE)
    label_df    = pd.read_csv(config.LABEL_FILE)
    pairs_df    = pd.read_csv(config.PAIRS_FILE)
    target_info = parse_pair(pairs_df)

    logger.info(
        "Data loaded — train: %s | labels: %s | targets: %d",
        train_df.shape, label_df.shape, len(target_info),
    )

    # ── MLflow setup ────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # creates the experiment if it doesn't exist yet
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # RUN — everything inside this block is logged to one run
    with mlflow.start_run(run_name=f"{model_type}_{int(time.time())}"):

        # useful for filtering runs by model type, owner etc
        mlflow.set_tags({
            "model_type"    : model_type,
            "python_version": sys.version.split()[0],
            "dataset"       : "mitsui-commodity",
        })

        # ── PARAMS — log all hyperparameters BEFORE training ─
        if model_type == "lgbm":
            mlflow.log_params(config.LGBM_PARAMS)

        elif model_type == "stacking":
            # flatten nested params into a single dict for MLflow
            mlflow.log_params({
                **{f"lgbm_{k}" : v for k, v in config.LGBM_PARAMS.items()},
                **{f"rf_{k}"   : v for k, v in config.RF_PARAMS.items()},
                **{f"xgb_{k}"  : v for k, v in config.XGB_PARAMS.items()},
                **{f"meta_{k}" : v for k, v in config.XGB_META_PARAMS.items()},
                "cv_n_splits"  : config.CV_N_SPLITS,
            })

        # log shared params
        mlflow.log_params({
            "n_targets"         : len(target_info),
            "min_train_samples" : config.MIN_TRAIN_SAMPLES,
            "imputer_strategy"  : config.IMPUTER_STRATEGY,
            "scaler_type"       : config.SCALER_TYPE,
            "spread_windows"    : str(config.SPREAD_WINDOWS),
            "log_return_windows": str(config.LOG_RETURN_WINDOWS),
        })

        start = time.time()

        if model_type == "lgbm":
            models      = train_lgbm(train_df, label_df, target_info)
            models_file = config.LGBM_MODELS_FILE
        else:
            models      = train_stacking(train_df, label_df, target_info)
            models_file = config.STACKING_MODELS_FILE

        elapsed = time.time() - start

        # ── METRICS — log outputs AFTER training ─────────────
        mlflow.log_metrics({
            "models_trained"   : len(models),
            "models_skipped"   : len(target_info) - len(models),
            "training_time_s"  : round(elapsed, 1),
            "training_time_min": round(elapsed / 60, 2),
        })

        logger.info("Training complete in %.1fs", elapsed)

        # ── SAVE MODELS ───────────────────────────────────────
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, models_file)
        logger.info("Models saved to %s", models_file)

        # ── ARTIFACTS — attach files to the run ──────────────
        mlflow.log_artifact(str(models_file))
        mlflow.log_artifact(str(config.ROOT_DIR / "config.py"))

        # ── MODEL REGISTRY ───────────────────────────────────
        # stages: None → Staging → Production → Archived
        registry_name = f"mitsui-{model_type}"

        mlflow.log_artifact(str(models_file), artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name,
        )
        logger.info(
            "Model registered — name: %s | version: %s",
            registered.name,
            registered.version,
        )

        logger.info(
            "MLflow run complete — "
            "experiment: %s | model: %s | run_id: %s",
            config.MLFLOW_EXPERIMENT,
            model_type,
            run_id,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train commodity prediction models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "stacking", "both"],
        default="lgbm",
        help="Which model to train (default: lgbm)",
    )
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
 
    if args.model == "both":
        logger.info("Training both models...")
        run_training("lgbm")
        run_training("stacking")
    else:
        run_training(args.model)
     