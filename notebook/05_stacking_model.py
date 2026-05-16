
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../')
from src.features import parse_pair, create_features_for_assets, build_dataset_for_target
from src.metrics import spearman_sharpe
from src.preprocessing import preprocess_features
from src.model import StackingModel

from sklearn.model_selection import TimeSeriesSplit
import os
import joblib
from tqdm import tqdm


DATA_PATH = '../'

train  = pd.read_csv(DATA_PATH + 'train.csv')
label  = pd.read_csv(DATA_PATH + 'train_labels.csv')
pairs  = pd.read_csv(DATA_PATH + 'target_pairs.csv')

print(f"Train  : {train.shape}")
print(f"Label  : {label.shape}")
print(f"Pairs  : {pairs.shape}")

target_cols = [c for c in label.columns if c.startswith('target_')]
target_info = parse_pair(pairs)

print(f"Targets: {len(target_cols)}")
print(f"Target info entries: {len(target_info)}")



t=target_info[2]
print(f"Target:{t}")

X,y =build_dataset_for_target(train,label,t)
print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")

X_processed,imputer,scaler=preprocess_features(X)
print(f"X processed shape: {X_processed.shape}")


Stackingmodel=StackingModel(random_state=42)# instead of calling Stackingmodel=StackingModel(base_models=['lgbm','xgb','catboost'],meta_model='lgbm',n_splits=5,random_state=42) we add these features in the class itself as default values
Stackingmodel.fit(X_processed,y)
print(f"\nStacking model trained successfully")
Stackingmodel.predict(X_processed)
print(f"Predictions generated successfully")
Stackingmodel._get_meta_features(X_processed)
print(f"Meta-features shape: {Stackingmodel._get_meta_features(X_processed).shape}")



def train_all_stacking(target_info, train, label):

    models={}
    for t in tqdm(target_info, desc="Training stacking models"):
        target_name=t['target']
        try:
            X,y=build_dataset_for_target(train,label,t)
            if len(y)<50:
                print(f"Skipping {target_name} due to insufficient data (only {len(y)} samples)")
                continue
            X_processed,imputer,scaler=preprocess_features(X)

            model=StackingModel(random_state=42)
            model.fit(X_processed,y)
            models[target_name]={
                'model': model,
                'imputer': imputer,
                'scaler': scaler,
                'assets':t['assets'],
                'lags':t['lag']
            }
        except Exception as e:
            print(f"Error training {target_name}: {e}")
            continue

    print(f"\nTrained {len(models)} stacking models out of {len(target_info)}")
    return models


# train all 424 stacking models
print("Training 424 stacking models...")
print("This will take longer than LightGBM — 3 base models + 1 meta model per target\n")
stacking_models = train_all_stacking(target_info, train, label)

# save stacking models dictionary
os.makedirs('../models', exist_ok=True)
joblib.dump(stacking_models, '../models/stacking_models.pkl')
print("Stacking models saved to ../models/stacking_models.pkl")



def predict_all_stacking(test_df, train_df, models_dict, target_info_list):
    """
    For each target, build features from test.csv,
    preprocess using training statistics, and predict
    using the stacking model.
    
    Returns DataFrame with shape (n_test_rows, 424)
    """
    target_cols = [t['target'] for t in target_info_list]
    n_rows      = len(test_df)
    predictions = pd.DataFrame(
        np.zeros((n_rows, len(target_cols))),
        columns=target_cols
    )

    for t in tqdm(target_info_list, desc="Predicting with stacking"):
        target_name = t['target']

        if target_name not in models_dict:
            continue

        model_data = models_dict[target_name]
        model      = model_data['model']
        imputer    = model_data['imputer']
        scaler     = model_data['scaler']
        assets     = model_data['assets']

        # build features for test data
        all_features = []
        for asset in assets:
            if asset not in test_df.columns:
                continue
            feat = create_features_for_assets(test_df, asset)
            all_features.append(feat)

        if not all_features:
            continue

        X_test = pd.concat(all_features, axis=1)

        # add spread features if two assets
        if len(assets) == 2:
            a1     = assets[0]
            a2     = assets[1]
            log_p1 = np.log(test_df[a1])
            log_p2 = np.log(test_df[a2])
            X_test['spread_ret_1d'] = (log_p1.diff(1) - log_p2.diff(1)).shift(1)
            X_test['spread_ret_5d'] = (log_p1.diff(5) - log_p2.diff(5)).shift(1)

        # preprocess using training statistics
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = np.log1p(np.abs(X_test)) * np.sign(X_test)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        # predict using stacking model
        preds = model.predict(X_test)
        predictions[target_name] = preds

    return predictions


# load test data
test = pd.read_csv(DATA_PATH + 'test.csv')
print(f"Test shape: {test.shape}")

# generate predictions
print("\nGenerating stacking predictions on test.csv...")
stacking_predictions = predict_all_stacking(test, train, stacking_models, target_info)

print(f"\nStacking predictions shape: {stacking_predictions.shape}")
print(f"Sample predictions (first 3 rows, first 5 targets):")
print(stacking_predictions.iloc[:3, :5].to_string())
print(f"\nAny NaN in predictions: {stacking_predictions.isnull().any().any()}")




def evaluate_predictions(predictions_df, data_path):
    """
    Compare predictions against 4 lagged test label files.
    Returns per-lag scores and overall average.
    """
    lag_files = {
        1: 'lagged_test_labels/test_labels_lag_1.csv',
        2: 'lagged_test_labels/test_labels_lag_2.csv',
        3: 'lagged_test_labels/test_labels_lag_3.csv',
        4: 'lagged_test_labels/test_labels_lag_4.csv',
    }

    lag_scores = {}

    for lag, filename in lag_files.items():
        filepath = os.path.join(data_path, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found — skipping lag {lag}")
            continue

        actuals = pd.read_csv(filepath)
        actual_target_cols = [c for c in actuals.columns if c.startswith('target_')]
        shared_targets     = [c for c in actual_target_cols if c in predictions_df.columns]

        n_rows          = min(len(predictions_df), len(actuals))
        preds_aligned   = predictions_df[shared_targets].iloc[:n_rows].reset_index(drop=True)
        actuals_aligned = actuals[shared_targets].iloc[:n_rows].reset_index(drop=True)

        score = spearman_sharpe(preds_aligned, actuals_aligned)
        lag_scores[lag] = score
        print(f"Lag {lag} Spearman-Sharpe score: {score:.4f}")

    valid_scores = [s for s in lag_scores.values() if not np.isnan(s)]
    overall      = np.mean(valid_scores) if valid_scores else np.nan

    return lag_scores, overall


print("Evaluating stacking predictions...\n")
stacking_lag_scores, stacking_overall = evaluate_predictions(stacking_predictions, DATA_PATH)

print("\n" + "=" * 60)
print("COMPLETE RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<45} {'Score':>10}")
print("-" * 60)
print(f"{'Random predictions':<45} {'-0.0100':>10}")
print(f"{'Baseline 1 — predict zero':<45} {'NaN':>10}")
print(f"{'Baseline 2 — predict yesterday':<45} {'2.3343':>10}")
print(f"{'LightGBM CV (single target)':<45} {'1.2200':>10}")
print(f"{'LightGBM lag 1':<45} {'5.4711':>10}")
print(f"{'LightGBM lag 2':<45} {'4.3529':>10}")
print(f"{'LightGBM lag 3':<45} {'4.9086':>10}")
print(f"{'LightGBM lag 4':<45} {'4.6129':>10}")
print(f"{'LightGBM overall':<45} {'4.8364':>10}")
print("-" * 60)
for lag, score in stacking_lag_scores.items():
    print(f"{'Stacking ensemble lag ' + str(lag):<45} {score:>10.4f}")
print(f"{'Stacking ensemble overall':<45} {stacking_overall:>10.4f}")
print("=" * 60)

# improvement
improvement = ((stacking_overall - 4.8364) / 4.8364) * 100
print(f"\nStacking improvement over LightGBM: {improvement:+.2f}%")