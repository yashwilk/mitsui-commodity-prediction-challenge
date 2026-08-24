import numpy as np
import pandas as pd
import pytest

from src.model import StackingModel


def test_fit_sets_is_fitted(synthetic_regression_data):
    X, y = synthetic_regression_data
    model = StackingModel(random_state=0)
    assert model.is_fitted is False
    model.fit(X, y)
    assert model.is_fitted is True


def test_predict_before_fit_raises(synthetic_regression_data):
    X, _ = synthetic_regression_data
    model = StackingModel()
    with pytest.raises(ValueError):
        model.predict(X)


def test_fit_with_too_few_samples_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0.1, 0.2, 0.3])
    model = StackingModel()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_returns_one_prediction_per_row(synthetic_regression_data):
    X, y = synthetic_regression_data
    model = StackingModel(random_state=0).fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)


def test_same_random_state_gives_reproducible_predictions(synthetic_regression_data):
    X, y = synthetic_regression_data
    model_a = StackingModel(random_state=7).fit(X, y)
    model_b = StackingModel(random_state=7).fit(X, y)
    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X))
