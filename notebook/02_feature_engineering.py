import sys
import pandas as pd
import numpy as np

sys.path.append('../')
from src.features import parse_pair, create_features_for_assets, build_dataset_for_target
from src.preprocessing import preprocess_features

# ============================================================
# Load data
# ============================================================
DATA_PATH = '../'

train   = pd.read_csv(DATA_PATH + 'train.csv')
label   = pd.read_csv(DATA_PATH + 'train_labels.csv')
targets = pd.read_csv(DATA_PATH + 'target_pairs.csv')

print(f"Train   : {train.shape}")
print(f"Label   : {label.shape}")
print(f"Targets : {targets.shape}")

# ============================================================
# Test parse_pair
# ============================================================
target_info = parse_pair(targets)
for f in target_info[:5]:
    print(f)

# ============================================================
# Test create_features_for_assets
# ============================================================
test_feature = create_features_for_assets(train, 'LME_CA_Close')
print(test_feature.shape)
print(test_feature.columns.tolist())
print(test_feature.isnull().sum())

# ============================================================
# Test build_dataset_for_target
# ============================================================
t = target_info[2]
print(f"Building dataset for: {t}")
print(f"Number of assets: {len(t['assets'])}")

X, y = build_dataset_for_target(train, label, t)

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")
print(f"Any NaN in y: {y.isna().any()}")

# ============================================================
# Test preprocess_features
# ============================================================
X_processed, imputer, scaler = preprocess_features(X)

print(f"X shape after preprocessing : {X_processed.shape}")
print(f"Any NaN remaining           : {X_processed.isnull().any().any()}")
print(f"Any infinity remaining      : {np.isinf(X_processed.values).any()}")
print(f"\nSample values before preprocessing:")
print(X.iloc[:3, :4].to_string())
print(f"\nSample values after preprocessing:")
print(X_processed.iloc[:3, :4].to_string())