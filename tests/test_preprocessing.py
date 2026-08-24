import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess_features


def test_output_has_no_missing_values(synthetic_features):
    X_processed, imputer, scaler = preprocess_features(synthetic_features)
    assert not X_processed.isnull().any().any()


def test_output_shape_and_columns_match_input(synthetic_features):
    X_processed, imputer, scaler = preprocess_features(synthetic_features)
    assert X_processed.shape == synthetic_features.shape
    assert list(X_processed.columns) == list(synthetic_features.columns)


def test_train_output_is_standardized(synthetic_features):
    X_processed, _, _ = preprocess_features(synthetic_features)
    assert np.allclose(X_processed.mean(), 0, atol=1e-8)
    assert np.allclose(X_processed.std(ddof=0), 1, atol=1e-2)


def test_test_set_is_transformed_with_train_fitted_statistics(synthetic_train_test_features):
    train, test = synthetic_train_test_features
    X_train, X_test, imputer, scaler = preprocess_features(train, test)
    assert X_test.shape == test.shape
    assert not X_test.isnull().any().any()
    # imputer/scaler must have been fit on train only, never refit on test
    assert scaler.n_samples_seen_ == len(train)


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        preprocess_features(pd.DataFrame())


def test_single_row_raises():
    X = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError):
        preprocess_features(X)


def test_non_dataframe_input_raises():
    with pytest.raises(TypeError):
        preprocess_features(np.array([[1.0, 2.0], [3.0, 4.0]]))
