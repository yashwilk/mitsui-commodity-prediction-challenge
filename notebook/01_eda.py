import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',50)
pd.set_option('display.float_format','{:.6f}'.format) 
sns.set_style('whitegrid')

print('Libraries imported successfully!')

DATA_PATH='../data/'
train=pd.read_csv(DATA_PATH+'train.csv') #raw data
test=pd.read_csv(DATA_PATH+'test.csv')#raw data
label=pd.read_csv(DATA_PATH+'train_labels.csv')
pairs=pd.read_csv(DATA_PATH+'target_pairs.csv')



print(f"Train shape:{train.shape}")
print(f"Test shape:{test.shape}")
print(f"Label shape:{label.shape}")
print(f"Pairs shape:{pairs.shape}")

print(train.columns[:10].tolist())
print(test.columns[-10:].tolist())
print(label.columns[:10].tolist())
print(pairs.columns[:10].tolist())

extra_in_test=set(test.columns)-set(train.columns)
print(f"Extra columns in test set: {extra_in_test}")

print(pairs.head(5).to_string())
print(pairs['lag'].value_counts().sort_index())
print(pairs.groupby('lag').size())
pairs['num_assets'] = pairs['pair'].str.count(' - ') + 1
print(pairs['num_assets'].value_counts())
print(pairs[pairs['num_assets']==1].to_string())


print("Train date_id min:", train['date_id'].min())
print("Train date_id max:", train['date_id'].max())
(train['date_id'].diff().dropna()==1).all()
missing = train.isnull().sum()
missing_pct=(missing/len(train)*100).round(2)
missing_df=pd.DataFrame({'missing_count':missing,'missing_pct':missing_pct}).sort_values('missing_pct',ascending=False)
print(missing_df[missing_df['missing_count']>0].head(5))

target_cols = [c for c in label.columns if c.startswith('target_')]
print(len(target_cols))
print(label['target_0'].head(10).to_string())

unique_assets=set()

for pair_str in pairs['pair']:
    assets=pair_str.split(' - ')
    for asset in assets:
        unique_assets.add(asset.strip())
print(f"Unique assets in pairs: {unique_assets}")


train_cols=set(train.columns)
missing_assets = unique_assets - train_cols
present = unique_assets & train_cols
print(f"Assets in pairs but missing in train: {missing_assets}")
print(f"Assets in pairs and present in train: {present}")



target_labels = [c for c in label.columns if c.startswith('target_')]
missing=label[target_labels].isnull().sum()
pct_missing=(missing/len(label)*100).round(2)
missing_df=pd.DataFrame({'missing_count':missing,'missing_pct':pct_missing}).sort_values('missing_pct',ascending=False)
print(missing_df)