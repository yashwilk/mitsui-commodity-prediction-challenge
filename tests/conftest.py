import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_features():
    """Small engineered-feature matrix with some missing values, like real target features before preprocessing."""
    rng = np.random.default_rng(42)
    n_rows = 60
    X = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n_rows),
        "feat_b": rng.normal(100, 10, n_rows),
        "feat_c": rng.normal(0, 0.01, n_rows),
    })
    X.iloc[:5, 0] = np.nan
    X.iloc[10:12, 1] = np.nan
    return X


@pytest.fixture
def synthetic_train_test_features(synthetic_features):
    train = synthetic_features.iloc[:45].reset_index(drop=True)
    test = synthetic_features.iloc[45:].reset_index(drop=True)
    return train, test


@pytest.fixture
def synthetic_regression_data():
    """Feature matrix and target large enough to satisfy config.MIN_TRAIN_SAMPLES."""
    rng = np.random.default_rng(0)
    n_rows = 80
    X = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n_rows),
        "feat_b": rng.normal(0, 1, n_rows),
        "feat_c": rng.normal(0, 1, n_rows),
    })
    y = pd.Series(0.5 * X["feat_a"] - 0.3 * X["feat_b"] + rng.normal(0, 0.01, n_rows))
    return X, y


@pytest.fixture
def perfect_rank_predictions():
    """predictions_df/actuals_df where predicted ranks exactly match actual ranks every day."""
    n_days, n_targets = 10, 6
    actuals = pd.DataFrame(
        np.tile(np.arange(n_targets), (n_days, 1)).astype(float),
        columns=[f"target_{i}" for i in range(n_targets)],
    )
    predictions = actuals.copy()
    return predictions, actuals
