import logging
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
 
import config
 
logger = logging.getLogger(__name__)


class StackingModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_fitted    = False
        # base learners

        self.lgbm = LGBMRegressor(**{
            **config.LGBM_PARAMS,
            "random_state": random_state,
        })

        self.rf = RandomForestRegressor(**{
            **config.RF_PARAMS,
            "random_state": random_state,
        })
    
        self.xgb = XGBRegressor(**{
            **config.XGB_PARAMS,
            "random_state": random_state,
        })

        # meta model
        self.meta = XGBRegressor(**{
            **config.XGB_META_PARAMS,
            "random_state": random_state,
        })

    def _validate_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError(
                "Model is not fitted yet — call fit() before predict()"
            )
        

  

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingModel":
        
        if len(y) < config.MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Only {len(y)} samples — need at least "
                f"{config.MIN_TRAIN_SAMPLES} to fit stacking model"
            )
        
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
    

    def predict(self,  X: pd.DataFrame) -> np.ndarray:
        self._validate_fitted()
        meta_features = self._get_meta_features(X)
        return self.meta.predict(meta_features)
 


    def _get_meta_features(self,  X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'pred_lgbm': self.lgbm.predict(X),
            'pred_rf'  : self.rf.predict(X),
            'pred_xgb' : self.xgb.predict(X),
        })