import argparse
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.metrics import spearman_sharpe
from src.features import parse_pair



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


# EVALUATE PREDICTIONS AGAINST LAG LABELS
# ============================================================

def evaluate_predictions(
    predictions_df: pd.DataFrame,
) -> tuple[dict, float]:

    lag_scores = {}

    for lag, filepath in config.LAG_LABEL_FILES.items():
        if not filepath.exists():
            logger.warning(
                "Lag %d label file not found: %s — skipping",
                lag, filepath,
            )
            continue

        actuals            = pd.read_csv(filepath)
        actual_target_cols = [c for c in actuals.columns if c.startswith("target_")]
        shared_targets     = [c for c in actual_target_cols if c in predictions_df.columns]

        if not shared_targets:
            logger.warning("No shared targets for lag %d — skipping", lag)
            continue

        n_rows          = min(len(predictions_df), len(actuals))
        preds_aligned   = predictions_df[shared_targets].iloc[:n_rows].reset_index(drop=True)
        actuals_aligned = actuals[shared_targets].iloc[:n_rows].reset_index(drop=True)
        score           = spearman_sharpe(preds_aligned, actuals_aligned)
        lag_scores[lag] = score

        logger.info("Lag %d Spearman-Sharpe: %.4f", lag, score)

    valid_scores = [s for s in lag_scores.values() if not np.isnan(s)]
    overall      = float(np.mean(valid_scores)) if valid_scores else float("nan")

    return lag_scores, overall

# PLOT RESULTS
# ============================================================

def plot_lag_comparison(
    lgbm_scores: dict,
    stacking_scores: dict,
    save_path: str,
) -> None:

    lags   = sorted(set(lgbm_scores) | set(stacking_scores))
    lgbm_v = [lgbm_scores.get(l, 0)     for l in lags]
    stk_v  = [stacking_scores.get(l, 0) for l in lags]

    x     = np.arange(len(lags))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 6))
    bars1   = ax.bar(x - width / 2, lgbm_v, width, label="LightGBM", color="#378ADD")
    bars2   = ax.bar(x + width / 2, stk_v,  width, label="Stacking",  color="#1D9E75")
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Spearman-Sharpe Score", fontsize=12)
    ax.set_title(
        "LightGBM vs Stacking by Lag\n"
        "MITSUI&CO. Commodity Prediction Challenge",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Lag {l}\n({l} day ahead)" for l in lags])
    ax.legend()
    ax.set_ylim(0, max(lgbm_v + stk_v) * 1.2 if lgbm_v + stk_v else 7)
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{bar.get_height():.2f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Comparison chart saved to %s", save_path)



# MLFLOW EVALUATION RUN
# ============================================================

def run_evaluation(model_type: str) -> tuple[dict, float]:
    if model_type == "lgbm":
        predictions_file = config.LGBM_PREDICTIONS_FILE
    else:
        predictions_file = config.STACKING_PREDICTIONS_FILE

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"No predictions found at {predictions_file}. "
            f"Run predict.py --model {model_type} first."
        )

    logger.info("Loading predictions from %s ...", predictions_file)
    predictions_df = pd.read_csv(predictions_file)
    logger.info("Predictions shape: %s", predictions_df.shape)

    # ── MLflow run ──────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"evaluate_{model_type}"):

        mlflow.set_tags({
            "run_type"  : "evaluation",
            "model_type": model_type,
        })

        mlflow.log_params({
            "model_type"      : model_type,
            "predictions_rows": predictions_df.shape[0],
            "predictions_cols": predictions_df.shape[1],
        })
        # ── compute scores ───────────────────────────────────
        lag_scores, overall = evaluate_predictions(predictions_df)
        for lag, score in lag_scores.items():
            if not np.isnan(score):
                mlflow.log_metric(f"lag_{lag}_score", round(score, 4))

        if not np.isnan(overall):
            mlflow.log_metric("overall_score", round(overall, 4))

        logger.info("Overall Spearman-Sharpe: %.4f", overall)

        # ── print results table ──────────────────────────────
        print("\n" + "=" * 55)
        print(f"EVALUATION RESULTS — {model_type.upper()}")
        print("=" * 55)
        print(f"{'Lag':<30} {'Score':>10}")
        print("-" * 55)
        for lag, score in sorted(lag_scores.items()):
            print(f"{'Lag ' + str(lag):<30} {score:>10.4f}")
        print("-" * 55)
        print(f"{'Overall':<30} {overall:>10.4f}")
        print("=" * 55)

    return lag_scores, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against lag label files"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "stacking", "both"],
        default="lgbm",
        help="Which model predictions to evaluate (default: lgbm)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model == "both":
        lgbm_lag_scores,     lgbm_overall     = run_evaluation("lgbm")
        stacking_lag_scores, stacking_overall  = run_evaluation("stacking")
        # plot comparison chart only when both are evaluated
        chart_path = str(config.ASSETS_DIR / "lag_comparison.png")
        plot_lag_comparison(lgbm_lag_scores, stacking_lag_scores, chart_path)

        # log chart to MLflow as artifact
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="evaluate_comparison"):
            mlflow.log_artifact(chart_path)
            mlflow.log_metrics({
                "lgbm_overall"    : round(lgbm_overall, 4),
                "stacking_overall": round(stacking_overall, 4),
                "improvement_pct" : round(
                    (stacking_overall - lgbm_overall) / lgbm_overall * 100, 2
                ),
            })

        improvement = (stacking_overall - lgbm_overall) / lgbm_overall * 100
        print(f"\nStacking improvement over LightGBM: {improvement:+.2f}%")

    else:
        run_evaluation(args.model)
