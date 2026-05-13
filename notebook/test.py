# test_imports.py
import sys
sys.path.append('../')

from src.features import parse_pair, create_features_for_assets, build_dataset_for_target
from src.metrics import spearman_sharpe
from src.preprocessing import preprocess_features

print("All imports successful")
print("Functions available:")
print(f"  parse_pair                : {callable(parse_pair)}")
print(f"  create_features_for_assets: {callable(create_features_for_assets)}")
print(f"  build_dataset_for_target  : {callable(build_dataset_for_target)}")
print(f"  spearman_sharpe           : {callable(spearman_sharpe)}")
print(f"  preprocess_features       : {callable(preprocess_features)}")