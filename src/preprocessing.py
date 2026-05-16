import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_features(X_train, X_test=None):

    X_train = X_train.copy()

    # step 1 — log1p transform preserving sign
    X_train = np.log1p(np.abs(X_train)) * np.sign(X_train)

    # step 2 — replace infinities
    X_train = X_train.replace([np.inf, -np.inf], np.nan)

    # step 3 — median imputation
    imputer = SimpleImputer(strategy='median')
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

    if X_test is not None:
        X_test = X_test.copy()
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
        return X_train, X_test, imputer, scaler

    return X_train, imputer, scaler


""""imputer, scaler has learned parameters from X_train, to be applied on X_test"""