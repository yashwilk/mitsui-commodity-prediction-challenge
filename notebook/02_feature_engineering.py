import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH='../'

train=pd.read_csv(DATA_PATH+'train.csv') #
label=pd.read_csv(DATA_PATH+"train_labels.csv")
targets=pd.read_csv(DATA_PATH+"target_pairs.csv ")

print(f"Train :{train.shape}")
print(f"Label :{label.shape}")
print(f"Targets :{targets.shape}")

def parse_pair(pairs_df):
    parsed=[]
    for _, row in pairs_df.iterrows(): #iterrows()gives each row as a series with the index and row .we use _ to ignore the index
        target=row['target']
        lag=row['lag']
        assets=[a.strip() for a in row['pair'].split('-')]
        parsed.append({'target':target,'lag':lag,'assets':assets})
    return parsed

target_info=parse_pair(targets)
for f in target_info[:5]:
    print(f)
    
def create_features_for_assets(df,asset_col):
    price=df[asset_col].copy()
    log_price=np.log(price)
    log_ret=log_price.diff(1)
    feature=pd.DataFrame(index=df.index)
    feature[f"{asset_col}_log_ret_1d"]=log_price.diff(1).shift(1)#Without shift:date_id=5 feature = log(price_5) - log(price_4)  ← uses today's price, LEAKAGE
    feature[f"{asset_col}_log_ret_3d"]=log_price.diff(3).shift(1) #price momentum 
    feature[f"{asset_col}_log_ret_5d"]=log_price.diff(5).shift(1)
    feature[f'{asset_col}_log_ret_10d'] = log_price.diff(10).shift(1)
    feature[f"{asset_col}_roll_mean_5d"] = log_ret.rolling(5).mean().shift(1) #Rolling mean smooths out the noise in daily returns and shows the underlying trend direction more clearly.
    feature[f"{asset_col}_roll_mean_10d"]=log_ret.rolling(10).mean().shift(1)
    feature[f"{asset_col}_roll_mean_20d"]=log_ret.rolling(20).mean().shift(1)
    feature[f"{asset_col}_roll_std_5d"]=log_ret.rolling(5).std().shift(1)#This captures volatility
    feature[f"{asset_col}_roll_std_10d"]=log_ret.rolling(10).std().shift(1)
    feature[f"{asset_col}_roll_std_20d"]=log_ret.rolling(20).std().shift(1)
    feature[f"{asset_col}_diff_1d"]=price.diff(1).shift(1)#Log returns are scale-independent. A 1% move in copper at $9,000 and a 1% move at $3,000 both show as +0.01. Raw differences capture the absolute magnitude
    feature[f"{asset_col}_diff_3d"]=price.diff(3).shift(1)
    feature[f"{asset_col}_diff_5d"]=price.diff(5).shift(1)
    feature[f"{asset_col}_price_lag1"]=price.shift(1)#They also help the model understand mean reversion.
    feature[f"{asset_col}_price_lag3"]=price.shift(3)
    feature[f"{asset_col}_price_lag5"]=price.shift(5)
    return feature

test_feature=create_features_for_assets(train,'LME_CA_Close')
print(test_feature)
print(test_feature.shape)
print(test_feature.columns.tolist())
print(test_feature.head())
print(test_feature.isnull().sum())



def build_dataset_for_target(train_df, label_df, target_dict):
    target_name = target_dict['target']
    assets      = target_dict['assets']
    
    all_features = []
    for asset in assets:
        if asset not in train_df.columns:
            print(f"Warning: asset {asset} not found in train_df")
            continue
        feat = create_features_for_assets(train_df, asset)
        all_features.append(feat)
    
    # outside the loop
    X = pd.concat(all_features, axis=1)
    
    # len(assets) not len(asset)
    if len(assets) == 2:
        a1     = assets[0]
        a2     = assets[1]
        log_p1 = np.log(train_df[a1])
        log_p2 = np.log(train_df[a2])
        X['spread_ret_1d'] = (log_p1.diff(1) - log_p2.diff(1)).shift(1)
        X['spread_ret_5d'] = (log_p1.diff(5) - log_p2.diff(5)).shift(1)
    
    # outside the loop
    y    = label_df[target_name].copy()
    mask = ~y.isna()
    X    = X[mask]
    y    = y[mask]
    
    return X, y


# test on target_2
t = target_info[2]
print(f"Building dataset for: {t}")
print(f"Number of assets: {len(t['assets'])}")

X, y = build_dataset_for_target(train, label, t)

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"\nFeature columns:")
print(X.columns.tolist())
print(f"\nAny NaN in y: {y.isna().any()}")


def preprocess_features(X_train,X_test=None):
    X_train=X_train.copy()
    X_train=np.log1p(np.abs(X_train))*np.sign(X_train)#(original value : -0.00882 p.abs         : +0.00882 p.log1p       : +0.00878  np.sign(-1)  : -0.00878)
    X_train=X_train.replace([np.inf,-np.inf],np.nan)

    imputer=SimpleImputer(strategy='median')

    X_train=pd.DataFrame(imputer.fit_transform(X_train),columns=X_train.columns)
    scaler=StandardScaler()
    X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)

    if X_test is not None:
        X_test=X_test.copy()
        X_test=np.log1p(np.abs(X_test))*np.sign(X_test)
        X_test=X_test.replace([np.inf,-np.inf],np.nan)
        X_test=pd.DataFrame(imputer.transform(X_test),columns=X_test.columns)
        X_test=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
        return X_train, X_test, imputer, scaler
    return X_train, imputer, scaler


# test on our target_2 dataset
X_processed, imputer, scaler = preprocess_features(X)

print(f"X shape after preprocessing : {X_processed.shape}")
print(f"Any NaN remaining           : {X_processed.isnull().any().any()}")
print(f"Any infinity remaining      : {np.isinf(X_processed.values).any()}")
print(f"\nSample values before preprocessing:")
print(X.iloc[:3, :4].to_string())
print(f"\nSample values after preprocessing:")
print(X_processed.iloc[:3, :4].to_string())