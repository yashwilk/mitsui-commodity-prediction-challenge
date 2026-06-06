import sys
import pandas as pd
import numpy as np

sys.path.append('../')
from src.metrics import spearman_sharpe

# ============================================================
# Load data
# ============================================================
DATA_PATH = '../'

train  = pd.read_csv(DATA_PATH + 'train.csv')
label  = pd.read_csv(DATA_PATH + 'train_labels.csv')
pairs  = pd.read_csv(DATA_PATH + 'target_pairs.csv')

print(f"Train  : {train.shape}")
print(f"Label  : {label.shape}")
print(f"Pairs  : {pairs.shape}")

target_cols = [c for c in label.columns if c.startswith('target_')]

# ============================================================
# Metric sanity checks
# ============================================================
noisy_preds = label[target_cols].copy()
noisy_preds = noisy_preds + np.random.normal(0, 0.001, noisy_preds.shape)
noisy_score = spearman_sharpe(noisy_preds, label[target_cols])
print(f"Near-perfect score (with tiny noise): {noisy_score:.4f}")

random_preds = pd.DataFrame(
    np.random.normal(0, 0.01, label[target_cols].shape),
    columns=target_cols
)
random_score = spearman_sharpe(random_preds, label[target_cols])
print(f"Random prediction score             : {random_score:.4f}")

# ============================================================
# Baseline 1 — predict zero
# ============================================================
zero_preds = pd.DataFrame(
    np.zeros(label[target_cols].shape),
    columns=target_cols
)
print(f"\nZero predictions shape: {zero_preds.shape}")
zero_score = spearman_sharpe(zero_preds, label[target_cols])
print(f"Baseline 1 score (predict zero): {zero_score}")

# ============================================================
# Baseline 2 — predict yesterday's return
# ============================================================
yesterday_preds = label[target_cols].shift(1)
comparison = pd.DataFrame({
    'actual'        : label['target_0'],
    'yesterday_pred': yesterday_preds['target_0']
}).head(10)
print(f"\n{comparison.to_string()}")

yesterday_score = spearman_sharpe(yesterday_preds, label[target_cols])
print(f"\nBaseline 2 score (predict yesterday): {yesterday_score:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print("BASELINE RESULTS SUMMARY")
print("=" * 50)
print(f"{'Model':<35} {'Score':>10}")
print("-" * 50)
print(f"{'Random predictions':<35} {random_score:>10.4f}")
print(f"{'Baseline 1 — predict zero':<35} {'NaN':>10}")
print(f"{'Baseline 2 — predict yesterday':<35} {yesterday_score:>10.4f}")
print(f"{'Near-perfect (reference ceiling)':<35} {noisy_score:>10.4f}")
print("=" * 50)
print(f"\nTarget to beat with LightGBM: {yesterday_score:.4f}")