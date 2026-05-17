import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


sys.path.append('../')
from src.metrics import spearman_sharpe
from src.features import parse_pair
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

DATA_PATH  = '../'
MODEL_PATH = '../models/'

train  = pd.read_csv(DATA_PATH + 'train.csv')
label  = pd.read_csv(DATA_PATH + 'train_labels.csv')
pairs  = pd.read_csv(DATA_PATH + 'target_pairs.csv')

lgbm_preds     = pd.read_csv(MODEL_PATH + 'lgbm_predictions.csv')
stacking_preds = pd.read_csv(MODEL_PATH + 'stacking_predictions.csv')

target_info = parse_pair(pairs)
target_cols = [c for c in label.columns if c.startswith('target_')]

print(f"Train          : {train.shape}")
print(f"Label          : {label.shape}")
print(f"LGBM preds     : {lgbm_preds.shape}")
print(f"Stacking preds : {stacking_preds.shape}")

#PLOT RESULTS

print("\n" + "=" * 60)
print("COMPLETE RESULTS SUMMARY")
print("=" * 60)

results = {
    'Model': [
        'Random predictions',
        'Baseline 1 — predict zero',
        'Baseline 2 — predict yesterday',
        'LightGBM lag 1',
        'LightGBM lag 2',
        'LightGBM lag 3',
        'LightGBM lag 4',
        'LightGBM overall',
        'Stacking lag 1',
        'Stacking lag 2',
        'Stacking lag 3',
        'Stacking lag 4',
        'Stacking overall',
    ],
    'Score': [
        -0.0100,
        float('nan'),
        2.3343,
        5.4711,
        4.3529,
        4.9086,
        4.6129,
        4.8364,
        5.7926,
        4.6785,
        4.9842,
        4.8601,
        5.0789,
    ]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 7))
plot_data = results_df.dropna()
colors = [
    '#888780',
    '#888780',
    '#378ADD',
    '#378ADD',
    '#378ADD',
    '#378ADD',
    '#185FA5',
    '#1D9E75',
    '#1D9E75',
    '#1D9E75',
    '#1D9E75',
    '#0F6E56',
]


bars = ax.barh(plot_data['Model'], plot_data['Score'], color=colors)
ax.set_xlabel('Spearman-Sharpe Score', fontsize=12)
ax.set_title(
    'Model Performance Comparison\nMITSUI&CO. Commodity Prediction Challenge',
    fontsize=14
)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
ax.axvline(x=2.3343, color='gray', linewidth=0.8,
           linestyle='--', label='Baseline (2.33)')
ax.axvline(x=4.8364, color='#378ADD', linewidth=0.8,
           linestyle='--', label='LightGBM (4.84)')

for bar, score in zip(bars, plot_data['Score']):
    ax.text(
        bar.get_width() + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f'{score:.2f}',
        va='center',
        fontsize=10
    )

ax.legend()
plt.tight_layout()
plt.savefig('../assets/results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to ../assets/results_comparison.png")


#LAG PERFORMANCE ANALYSIS
print("\n" + "=" * 60)
print("LAG PERFORMANCE ANALYSIS")
print("=" * 60)

lag_comparison = pd.DataFrame({
    'Lag'     : [1, 2, 3, 4],
    'LightGBM': [5.4711, 4.3529, 4.9086, 4.6129],
    'Stacking': [5.7926, 4.6785, 4.9842, 4.8601]
})

print(lag_comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
x     = np.arange(4)
width = 0.35

bars1 = ax.bar(x - width/2, lag_comparison['LightGBM'],
               width, label='LightGBM', color='#378ADD')
bars2 = ax.bar(x + width/2, lag_comparison['Stacking'],
               width, label='Stacking',  color='#1D9E75')


ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Spearman-Sharpe Score', fontsize=12)
ax.set_title(
    'LightGBM vs Stacking by Lag\n'
    'MITSUI&CO. Commodity Prediction Challenge',
    fontsize=14
)
ax.set_xticks(x)
ax.set_xticklabels([
    'Lag 1\n(1 day ahead)',
    'Lag 2\n(2 days ahead)',
    'Lag 3\n(3 days ahead)',
    'Lag 4\n(4 days ahead)'
])
ax.legend()
ax.set_ylim(0, 7)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f'{bar.get_height():.2f}',
        ha='center', fontsize=10
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f'{bar.get_height():.2f}',
        ha='center', fontsize=10
    )

plt.tight_layout()
plt.savefig('../assets/lag_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Lag comparison plot saved")
