import pandas as pd
import numpy as np

DATA_PATH='../'

train=pd.read_csv(DATA_PATH+'train.csv') #
test=pd.read_csv(DATA_PATH+"train_labels.csv")
targets=pd.read_csv(DATA_PATH+"target_pairs.csv ")

print(f"Train :{train.shape}")
print(f"Test :{test.shape}")
print(f"Targets :{targets.shape}")


def create_features_for_assets(df,asset_col):
    price=df[asset_col].copy()
    log_price=np.log(price)
    log_ret=log_price.diff(1)
    feature=pd.DataFrame(index=df.index)
    feature[f"{asset_col}_log_ret_1d"]=log_price.diff(1).shift(1)#Without shift:date_id=5 feature = log(price_5) - log(price_4)  ← uses today's price, LEAKAGE
    feature[f"{asset_col}_log_ret_3d"]=log_price.diff(3).shift(1)
    feature[f"{asset_col}_log_ret_5d"]=log_price.diff(5).shift(1)
    feature[f'{asset_col}_log_ret_10d'] = log_price.diff(10).shift(1)
    feature[f"{asset_col}_roll_mean_5d"] = log_ret.rolling(5).mean().shift(1)
    feature[f"{asset_col}_roll_mean_10d"]=log_ret.rolling(10).mean().shift(1)
    feature[f"{asset_col}_roll_mean_20d"]=log_ret.rolling(20).mean().shift(1)
    feature[f"{asset_col}_roll_std_5d"]=log_ret.rolling(5).std().shift(1)
    feature[f"{asset_col}_roll_std_10d"]=log_ret.rolling(10).std().shift(1)
    feature[f"{asset_col}_roll_std_20d"]=log_ret.rolling(20).std().shift(1)
    feature[f"{asset_col}_diff_1d"]=price.diff(1).shift(1)
    feature[f"{asset_col}_diff_3d"]=price.diff(3).shift(1)
    feature[f"{asset_col}_diff_5d"]=price.diff(5).shift(1)
    feature[f"{asset_col}_price_lag1"]=price.shift(1)
    feature[f"{asset_col}_price_lag3"]=price.shift(3)
    feature[f"{asset_col}_price_lag5"]=price.shift(5)
    return feature

test_feature=create_features_for_assets(train,'LME_CA_Close')
print(test_feature)
print(test_feature.shape)
print(test_feature.columns.tolist())
print(test_feature.head())
print(test_feature.isnull().sum())