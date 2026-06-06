import argparse
import logging
import sys
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.features import (
    parse_pair,
    create_features_for_assets,
    add_spread_features,
)
from src.preprocessing import preprocess_features

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


def build_test_features(
    test_df: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame | None:

    all_features = []

    for asset in assets:
        if asset not in test_df.columns:
            logger.warning("Asset '%s' not found in test_df — skipping", asset)
            continue
        feat = create_features_for_assets(test_df, asset)
        all_features.append(feat)

    if not all_features:
        return None

    X_test = pd.concat(all_features, axis=1)

    # add spread features for two-asset targets
    if len(assets) == 2:
        X_test = add_spread_features(X_test, test_df, assets)

    return X_test


# PREDICT ALL TARGETS
# ============================================================

def predict_all(
    test_df: pd.DataFrame,
    models_dict: dict,
    target_info: list,
) -> pd.DataFrame:

    target_cols = [t["target"] for t in target_info]
    n_rows      = len(test_df)

    # initialise predictions with zeros
    predictions = pd.DataFrame(
        np.zeros((n_rows, len(target_cols))),
        columns=target_cols,
    )

    for t in target_info:
        target_name = t["target"]
        if target_name not in models_dict:
            logger.warning("No model for target '%s' — skipping", target_name)
            continue

        model_data = models_dict[target_name]
        model      = model_data["model"]
        imputer    = model_data["imputer"]
        scaler     = model_data["scaler"]
        assets     = model_data["assets"]

        # build test features
        X_test = build_test_features(test_df, assets)
        if X_test is None:
            continue

        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = np.log1p(np.abs(X_test)) * np.sign(X_test)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
        )

        predictions[target_name] = model.predict(X_test)

    return predictions


# MLFLOW PREDICTION RUN
# ============================================================

def run_prediction(model_type: str) -> None:
    # ── load models ─────────────────────────────────────────
    if model_type == "lgbm":
        models_file      = config.LGBM_MODELS_FILE
        predictions_file = config.LGBM_PREDICTIONS_FILE
    else:
        models_file      = config.STACKING_MODELS_FILE
        predictions_file = config.STACKING_PREDICTIONS_FILE

    if not models_file.exists():
        raise FileNotFoundError(
            f"No trained models found at {models_file}. "
            f"Run train.py --model {model_type} first."
        )

    logger.info("Loading models from %s ...", models_file)
    models_dict = joblib.load(models_file)
    logger.info("Loaded %d models", len(models_dict))
    # ── load data ───────────────────────────────────────────
    logger.info("Loading test data...")
    test_df     = pd.read_csv(config.TEST_FILE)
    pairs_df    = pd.read_csv(config.PAIRS_FILE)
    target_info = parse_pair(pairs_df)

    logger.info(
        "Test data loaded — shape: %s | targets: %d",
        test_df.shape, len(target_info),
    )

    # ── MLflow run ──────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"predict_{model_type}_{int(time.time())}"):
        mlflow.set_tags({
            "run_type"  : "prediction",
            "model_type": model_type,
        })

        mlflow.log_params({
            "model_type"   : model_type,
            "n_test_rows"  : len(test_df),
            "n_targets"    : len(target_info),
            "models_loaded": len(models_dict),
        })
        start       = time.time()
        predictions = predict_all(test_df, models_dict, target_info)
        elapsed     = time.time() - start

        logger.info("Predictions generated in %.1fs", elapsed)

        n_nan       = predictions.isnull().any().any()
        n_zero_cols = (predictions == 0).all().sum()
        if n_nan:
            logger.warning("NaN values found in predictions")
        if n_zero_cols > 0:
            logger.warning(
                "%d targets have all-zero predictions "
                "(no model or no valid assets)",
                n_zero_cols,
            )

        # ── log metrics ──────────────────────────────────────
        mlflow.log_metrics({
            "prediction_time_s": round(elapsed, 1),
            "n_zero_columns"   : int(n_zero_cols),
            "has_nan"          : int(n_nan),
            "predictions_shape_rows": predictions.shape[0],
            "predictions_shape_cols": predictions.shape[1],
        })
        # ── save predictions ─────────────────────────────────
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_file, index=False)
        logger.info("Predictions saved to %s", predictions_file)
        # ── log artifacts ────────────────────────────────────
        mlflow.log_artifact(str(predictions_file))

        logger.info(
            "Prediction run complete — model: %s | shape: %s",
            model_type, predictions.shape,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "stacking"],
        default="lgbm",
        help="Which model to use for predictions (default: lgbm)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(args.model)
