import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


class StackingModel:
    def __init__(self, random_state=42):
        # base learners
        self.lgbm = LGBMRegressor(
            n_estimators     = 100,
            learning_rate    = 0.05,
            num_leaves       = 31,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = random_state,
            verbose          = -1
        )

        self.rf = RandomForestRegressor(
            n_estimators = 100,
            max_depth    = 6,
            min_samples_leaf = 5,
            random_state = random_state,
            n_jobs       = -1
        )

        self.xgb = XGBRegressor(
            n_estimators     = 100,
            learning_rate    = 0.05,
            max_depth        = 4,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = random_state,
            verbosity        = 0
        )

        # meta model
        self.meta = XGBRegressor(
            n_estimators = 50,
            learning_rate = 0.05,
            max_depth    = 3,
            random_state = random_state,
            verbosity    = 0
        )

        self.is_fitted = False

    def fit(self, X, y):
        # Fit base learners
        self.lgbm.fit(X, y)
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        # Generate meta-features
        meta_X = self._get_meta_features(X)
        # Fit meta-model
        self.meta.fit(meta_X, y)
        self.is_fitted = True
        return self
    

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        meta_X = self._get_meta_features(X)
        return self.meta.predict(meta_X)


    def _get_meta_features(self, X):
        return pd.DataFrame({
            'pred_lgbm': self.lgbm.predict(X),
            'pred_rf'  : self.rf.predict(X),
            'pred_xgb' : self.xgb.predict(X),
        })