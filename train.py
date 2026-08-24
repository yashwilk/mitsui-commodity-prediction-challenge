import argparse
import logging
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
from src.transformer import TransformerModel
from src.metrics import spearman_sharpe

# ============================================================
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


# ============================================================
# SHARED TRAINING LOOP
# ============================================================

def train_all(
    train_df   : pd.DataFrame,
    label_df   : pd.DataFrame,
    target_info: list,
    model_type : str,
) -> dict:
    """
    Train one model per target (424 total).
    Works for lgbm, stacking and transformer.
    Returns dict of fitted models keyed by target name.
    """
    models = {}

    for t in tqdm(target_info, desc=f"Training {model_type} models"):
        target_name = t["target"]
        try:
            X, y = build_dataset_for_target(train_df, label_df, t)

            if len(y) < config.MIN_TRAIN_SAMPLES:
                logger.warning(
                    "Skipping %s — only %d samples", target_name, len(y)
                )
                continue

            X_processed, imputer, scaler = preprocess_features(X)

            if model_type == "lgbm":
                model = LGBMRegressor(**config.LGBM_PARAMS)
            elif model_type == "stacking":
                model = StackingModel(random_state=config.CV_RANDOM_SEED)
            elif model_type == "transformer":
                model = TransformerModel(random_state=config.CV_RANDOM_SEED)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

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

    logger.info(
        "Trained %d / %d %s models",
        len(models), len(target_info), model_type,
    )
    return models


# ============================================================
# MLFLOW TRAINING RUN
# ============================================================

def run_training(model_type: str) -> None:
    """
    Full training pipeline wrapped in a single MLflow run.
    Works for lgbm, stacking and transformer.
    """

    # ── load data ───────────────────────────────────────────
    logger.info("Loading data...")
    train_df    = pd.read_csv(config.TRAIN_FILE)
    label_df    = pd.read_csv(config.LABEL_FILE)
    pairs_df    = pd.read_csv(config.PAIRS_FILE)
    target_info = parse_pair(pairs_df)                                                      #{'target': 'target_1', 'lag': 1, 'assets': ['LME_PB_Close', 'US_Stock_VT_adj_close']}

    logger.info(
        "Data loaded — train: %s | labels: %s | targets: %d",
        train_df.shape, label_df.shape, len(target_info),
    )

    # ── MLflow setup ────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{model_type}_{int(time.time())}"):

        # ── TAGS ────────────────────────────────────────────
        mlflow.set_tags({
            "model_type"    : model_type,
            "python_version": sys.version.split()[0],
            "dataset"       : "mitsui-commodity",
        })

        # ── PARAMS ──────────────────────────────────────────
        if model_type == "lgbm":
            mlflow.log_params(config.LGBM_PARAMS)

        elif model_type == "stacking":
            mlflow.log_params({
                **{f"lgbm_{k}" : v for k, v in config.LGBM_PARAMS.items()},
                **{f"rf_{k}"   : v for k, v in config.RF_PARAMS.items()},
                **{f"xgb_{k}"  : v for k, v in config.XGB_PARAMS.items()},
                **{f"meta_{k}" : v for k, v in config.XGB_META_PARAMS.items()},
                "cv_n_splits"  : config.CV_N_SPLITS,
            })

        elif model_type == "transformer":
            mlflow.log_params(config.TRANSFORMER_PARAMS)

        mlflow.log_params({
            "n_targets"         : len(target_info),
            "min_train_samples" : config.MIN_TRAIN_SAMPLES,
            "imputer_strategy"  : config.IMPUTER_STRATEGY,
            "scaler_type"       : config.SCALER_TYPE,
            "spread_windows"    : str(config.SPREAD_WINDOWS),
            "log_return_windows": str(config.LOG_RETURN_WINDOWS),
        })

        # ── TRAINING ────────────────────────────────────────
        start = time.time()

        models = train_all(train_df, label_df, target_info, model_type)

        if model_type == "lgbm":
            models_file = config.LGBM_MODELS_FILE
        elif model_type == "stacking":
            models_file = config.STACKING_MODELS_FILE
        elif model_type == "transformer":
            models_file = config.TRANSFORMER_MODELS_FILE

        elapsed = time.time() - start

        # ── METRICS ─────────────────────────────────────────
        mlflow.log_metrics({
            "models_trained"   : len(models),
            "models_skipped"   : len(target_info) - len(models),
            "training_time_s"  : round(elapsed, 1),
            "training_time_min": round(elapsed / 60, 2),
        })

        logger.info("Training complete in %.1fs", elapsed)

        # ── SAVE MODELS ──────────────────────────────────────
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, models_file)
        logger.info("Models saved to %s", models_file)

        # ── ARTIFACTS ───────────────────────────────────────
        mlflow.log_artifact(str(models_file))
        mlflow.log_artifact(str(config.ROOT_DIR / "config.py"))

        # ── MODEL REGISTRY ───────────────────────────────────
        mlflow.log_artifact(str(models_file), artifact_path="model")
        run_id    = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=f"mitsui-{model_type}",
        )

        logger.info(
            "Model registered — name: %s | version: %s",
            registered.name,
            registered.version,
        )

        logger.info(
            "MLflow run complete — experiment: %s | model: %s | run_id: %s",
            config.MLFLOW_EXPERIMENT,
            model_type,
            run_id,
        )


# ============================================================
# ENTRY POINT
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train commodity prediction models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "stacking", "transformer", "both"],
        default="lgbm",
        help="Which model to train (default: lgbm)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "both":
        logger.info("Training all models...")
        run_training("lgbm")
        run_training("stacking")
        run_training("transformer")
    else:
        run_training(args.model)