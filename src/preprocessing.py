import numpy as np
import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config

logger=logging.getLogger(__name__)


def _log1p_transform(
        X: pd.DataFrame
) -> pd.DataFrame:
    X = np.log1p(np.abs(X)) * np.sign(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

def _validate_input(X:pd.DataFrame,name:str="X"):
    if not isinstance(X,pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(X)}")
    if X.empty:
        raise ValueError(f"{name} is empty — nothing to preprocess")
    if X.shape[0]<2:
        raise ValueError(
            f"{name} has only {X.shape[0]} row(s) — need at least 2"
        )




def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> tuple:
    
    _validate_input(X_train, "X_train")

    # log1p + infinity clipping (same logic for train and test)
    X_train = _log1p_transform(X_train.copy())

    # step 3 — median imputation
    imputer = SimpleImputer(strategy=config.IMPUTER_STRATEGY)
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    # step 4 — standard scaling
    scaler  = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    logger.debug(
        "Preprocessing complete — train shape: %s | NaN remaining: %s",
        X_train.shape,
        X_train.isnull().any().any(),
    )


    if X_test is not None:
        _validate_input(X_test, "X_test")
        # steps 1 & 2 — same transform, no fitting
        X_test = _log1p_transform(X_test.copy())
        # steps 3 & 4 — apply fitted imputer and scaler (transform only)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        X_test = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)
        logger.debug("Test set transformed — shape: %s", X_test.shape)
        return X_train, X_test, imputer, scaler

    return X_train, imputer, scaler


""""imputer, scaler has learned parameters from X_train, to be applied on X_test"""