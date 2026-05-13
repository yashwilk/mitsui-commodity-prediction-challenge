import numpy as np
from scipy.stats import spearmanr


def spearman_sharpe(predictions_df, actuals_df):
    """
    Competition metric — Sharpe ratio of daily Spearman correlations.
    
    For each trading day:
      - Compute Spearman rank correlation between 424 predictions and actuals
    Final score = mean(daily correlations) / std(daily correlations)
    
    Higher is better. Above 1.0 is good. Above 2.0 is excellent.
    """
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

    return float(mean / std)