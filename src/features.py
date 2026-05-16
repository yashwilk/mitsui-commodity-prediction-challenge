import pandas as pd
import numpy as np


def parse_pair(pairs_df):
    """
    Parse target_pairs.csv into a list of dictionaries.
    Each dict contains target name, lag, and list of assets.
    """
    parsed = []
    for _, row in pairs_df.iterrows():
        target = row['target']
        lag    = row['lag']
        assets = [a.strip() for a in row['pair'].split(' - ')]
        parsed.append({
            'target': target,
            'lag'   : lag,
            'assets': assets
        })
    return parsed

"""{'target': 'target_0', 'lag': 1, 'assets': ['US_Stock_VT_adj_close']}
{'target': 'target_1', 'lag': 1, 'assets': ['LME_PB_Close', 'US_Stock_VT_adj_close']}
{'target': 'target_2', 'lag': 1, 'assets': ['LME_CA_Close', 'LME_ZS_Close']}
{'target': 'target_3', 'lag': 1, 'assets': ['LME_AH_Close', 'LME_ZS_Close']}
{'target': 'target_4', 'lag': 1, 'assets': ['LME_AH_Close', 'JPX_Gold_Standard_Futures_Close']}"""


def create_features_for_assets(df, asset_col):
    """
    Build 16 engineered features from one raw price column.
    All features shifted by 1 day to prevent leakage.
    
    Features:
      - Log returns at 1, 3, 5, 10 day horizons
      - Rolling mean of returns at 5, 10, 20 day windows
      - Rolling std (volatility) at 5, 10, 20 day windows
      - Raw price differences at 1, 3, 5 days
      - Lagged price levels at 1, 3, 5 days
    """
    price     = df[asset_col].copy()
    log_price = np.log(price)
    log_ret   = log_price.diff(1)

    feature = pd.DataFrame(index=df.index)

    # log returns
    feature[f'{asset_col}_log_ret_1d']  = log_price.diff(1).shift(1)
    feature[f'{asset_col}_log_ret_3d']  = log_price.diff(3).shift(1)
    feature[f'{asset_col}_log_ret_5d']  = log_price.diff(5).shift(1)
    feature[f'{asset_col}_log_ret_10d'] = log_price.diff(10).shift(1)

    # rolling mean
    feature[f'{asset_col}_roll_mean_5d']  = log_ret.rolling(5).mean().shift(1)
    feature[f'{asset_col}_roll_mean_10d'] = log_ret.rolling(10).mean().shift(1)
    feature[f'{asset_col}_roll_mean_20d'] = log_ret.rolling(20).mean().shift(1)

    # rolling std
    feature[f'{asset_col}_roll_std_5d']  = log_ret.rolling(5).std().shift(1)
    feature[f'{asset_col}_roll_std_10d'] = log_ret.rolling(10).std().shift(1)
    feature[f'{asset_col}_roll_std_20d'] = log_ret.rolling(20).std().shift(1)

    # raw price differences
    feature[f'{asset_col}_diff_1d'] = price.diff(1).shift(1)
    feature[f'{asset_col}_diff_3d'] = price.diff(3).shift(1)
    feature[f'{asset_col}_diff_5d'] = price.diff(5).shift(1)

    # lagged price levels
    feature[f'{asset_col}_price_lag1'] = price.shift(1)
    feature[f'{asset_col}_price_lag3'] = price.shift(3)
    feature[f'{asset_col}_price_lag5'] = price.shift(5)

    return feature

"""(1961, 16) 16 features for one asset"""

"""LME_CA_Close_log_ret_1d  LME_CA_Close_price_lag1  ...
0              NaN                     NaN            ...
1              NaN                     NaN            ...
2           -0.008083                  NaN            ...
3            0.005790                  NaN            ...
4           -0.009434               9000.0            ..."""




def build_dataset_for_target(train_df, label_df, target_dict):
    """
    Build complete feature matrix X and label vector y
    for one specific target.
    
    Uses only the assets defined in target_pairs.csv
    for that target — no irrelevant features included.
    Removes rows where label is NaN.
    """
    target_name = target_dict['target']
    assets      = target_dict['assets']

    all_features = []
    for asset in assets:
        if asset not in train_df.columns:
            print(f"Warning: {asset} not found in train_df — skipping")
            continue
        feat = create_features_for_assets(train_df, asset)
        all_features.append(feat)

    X = pd.concat(all_features, axis=1)

    # add spread features for two-asset targets
    if len(assets) == 2:
        a1     = assets[0]
        a2     = assets[1]
        log_p1 = np.log(train_df[a1])
        log_p2 = np.log(train_df[a2])
        X['spread_ret_1d'] = (log_p1.diff(1) - log_p2.diff(1)).shift(1)
        X['spread_ret_5d'] = (log_p1.diff(5) - log_p2.diff(5)).shift(1)

    y    = label_df[target_name].copy()
    mask = ~y.isna()
    X    = X[mask]
    y    = y[mask]

    return X, y



"""(1875, 34)"""

"""excludes blank"""    """ 34 features for 2 assers"""
"""LME_CA_Close_log_ret_1d  LME_ZS_Close_log_ret_1d  spread_ret_1d  ...
0          -0.004231                  -0.003812          -0.000419   ...
1          +0.006210                  +0.004521          +0.001689   ...
2          -0.008082                  -0.006234          -0.001848   ..."""
