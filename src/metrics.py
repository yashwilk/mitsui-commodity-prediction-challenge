 
import logging
 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
 
logger = logging.getLogger(__name__)

def spearman_sharpe(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> float:
    """
    Competition metric — Sharpe ratio of daily Spearman correlations.
    
    For each trading day:
      - Compute Spearman rank correlation between 424 predictions and actuals
    Final score = mean(daily correlations) / std(daily correlations)
    
    Higher is better. Above 1.0 is good. Above 2.0 is excellent.
    """

    if predictions_df.empty or actuals_df.empty:
        raise ValueError("predictions_df and actuals_df must not be empty")
 
    if predictions_df.shape != actuals_df.shape:
        raise ValueError(
            f"Shape mismatch — predictions: {predictions_df.shape}, "
            f"actuals: {actuals_df.shape}"
        )
    

    daily_corrs = []

    for i in range(len(predictions_df)):
        preds   = predictions_df.iloc[i].values
        actuals = actuals_df.iloc[i].values

        mask = ~np.isnan(actuals)
        if mask.sum() < 2:
            continue

        corr, _ = spearmanr(preds[mask], actuals[mask])

        if np.isnan(corr):
            continue

        daily_corrs.append(corr)

    daily_corrs = np.array(daily_corrs)
    mean = np.mean(daily_corrs)
    std  = np.std(daily_corrs)

    if std < 1e-8:
        return 0.0

    score = float(mean / std)
 
    logger.debug(
        "Spearman-Sharpe: %.4f (mean=%.4f, std=%.4f, n_days=%d)",
        score, mean, std, len(daily_corrs),
    )
 
    return score