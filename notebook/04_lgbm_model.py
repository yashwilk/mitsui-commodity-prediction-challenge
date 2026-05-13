import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../')
from src.features import parse_pair, create_features_for_assets, build_dataset_for_target
from src.preprocessing import preprocess_features
from src.metrics import spearman_sharpe

from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from tqdm import tqdm


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
target_info=parse_pair(pairs)

print(f"Targets: {len(target_cols)}")
print(f"Target info entries: {len(target_info)}")

t=target_info[2]
print(f"Target:{t}")

X,y=build_dataset_for_target(train,label,t)
print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")

X_processed,imputer,scaler=preprocess_features(X)
print(f"X processed shape: {X_processed.shape}")

model=LGBMRegressor(n_estimators=100,learning_rate=0.05,max_depth=-1,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1)

model.fit(X_processed,y)

print(f"\nModel trained successfully")
print(f"Number of trees : {model.n_estimators_}")
print(f"Number of features: {model.n_features_in_}")

importances=pd.Series(model.feature_importances_,index=X_processed.columns).sort_values(ascending=False)
print(importances)



def cv_score_for_target(train_df, label_df, target_dict, n_splits=5):

    X, y = build_dataset_for_target(train_df, label_df, target_dict)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_tr_p, X_val_p, _, _ = preprocess_features(X_tr, X_val)

        m = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=-1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        m.fit(X_tr_p, y_tr)
        preds = m.predict(X_val_p)
        corr, _ = spearmanr(preds, y_val)
        scores.append(corr)
        print(f"  Fold {fold+1}: Spearman = {corr:.4f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\n  Mean Spearman : {mean_score:.4f}")
    print(f"  Std Spearman  : {std_score:.4f}")
    print(f"  Sharpe-like   : {mean_score/std_score:.4f}" if std_score > 0 else "  Sharpe-like   : N/A")

    return scores

print("Running 5-fold TimeSeriesSplit CV for target_2...")
scores = cv_score_for_target(train, label, target_info[2], n_splits=5)








def train_all_targets(train_df, label_df, target_info_list):
    models = {}
    for t in tqdm(target_info_list, desc="Training models"):
        target_name = t['target']
        try:
            X, y = build_dataset_for_target(train_df, label_df, t)
            if len(y) < 50:
                print(f"Skipping {target_name} — too few samples ({len(y)})")
                continue
            X_processed, imputer, scaler = preprocess_features(X)
            model = LGBMRegressor(
                n_estimators     = 100,
                learning_rate    = 0.05,
                num_leaves       = 31,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                random_state     = 42,
                verbose          = -1
            )
            model.fit(X_processed, y)
            models[target_name] = {
                'model'  : model,
                'imputer': imputer,
                'scaler' : scaler,
                'assets' : t['assets'],
                'lag'    : t['lag']
            }
        except Exception as e:
            print(f"Error training {target_name}: {e}")
            continue

    print(f"\nTrained {len(models)} models out of {len(target_info_list)}")
    return models


# train all 424 models
print("Training 424 LightGBM models...")
print("This will take a few minutes — one model per target\n")
models = train_all_targets(train, label, target_info)