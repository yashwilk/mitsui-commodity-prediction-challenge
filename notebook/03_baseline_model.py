import pandas as pd
import numpy as np
from scipy.stats import spearmanr

DATA_PATH = '../'

train  = pd.read_csv(DATA_PATH + 'train.csv')
label  = pd.read_csv(DATA_PATH + 'train_labels.csv')
pairs  = pd.read_csv(DATA_PATH + 'target_pairs.csv')

print(f"Train  : {train.shape}")
print(f"Label  : {label.shape}")
print(f"Pairs  : {pairs.shape}")

#The competition metric
def spearman_sharpe(predictions_df,actuals_df):
    #mean(daily Spearman correlations) / std(daily Spearman correlations)
    #redictions_df : DataFrame, rows=dates, columns=424 targets
     #actuals_df     : pd.DataFrame, rows=dates, columns=424 targets
     
    daily_corrs = []
    
    for i in range(len(predictions_df)):
        preds   = predictions_df.iloc[i].values
        actuals = actuals_df.iloc[i].values
        
        # only score where actuals exist
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
target_cols = [c for c in label.columns if c.startswith('target_')]


noisy_preds = label[target_cols].copy()
noisy_preds = noisy_preds + np.random.normal(0, 0.001, noisy_preds.shape)

noisy_score = spearman_sharpe(noisy_preds, label[target_cols])
print(f"Near-perfect score (with tiny noise): {noisy_score:.4f}")

# also check random predictions
random_preds = pd.DataFrame(
    np.random.normal(0, 0.01, label[target_cols].shape),
    columns=target_cols
)
random_score = spearman_sharpe(random_preds, label[target_cols])
print(f"Random prediction score             : {random_score:.4f}")

# Model 1
# Baseline 1: predict zero for everything
zero_preds=pd.DataFrame(np.zeros(label[target_cols].shape),columns=target_cols)
print(f"Zero predictions shape: {zero_preds.shape}")
print(zero_preds.iloc[0][:5].to_list())
zero_score = spearman_sharpe(zero_preds, label[target_cols])
print(f"\nBaseline 1 score (predict zero): {zero_score:.4f}")

#Baseline 2: predict yesterday's return
yesterday_preds=label[target_cols].shift(1)
print(f"Yesterday predictions shape: {yesterday_preds.shape}")
comparison=pd.DataFrame({'actual':label['target_0'],'yesterday_pred':yesterday_preds['target_0']}).head(10)
print(comparison.to_string())
yesterday_score = spearman_sharpe(yesterday_preds, label[target_cols])
print(f"\nBaseline 2 score (predict yesterday): {yesterday_score:.4f}")
print(f"\nTarget to beat with LightGBM: {yesterday_score:.4f}")
