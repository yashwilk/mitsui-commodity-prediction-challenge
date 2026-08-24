import numpy as np
import pandas as pd
import pytest

from src.metrics import spearman_sharpe


def test_perfect_agreement_every_day_returns_zero_via_std_guard(perfect_rank_predictions):
    predictions, actuals = perfect_rank_predictions
    score = spearman_sharpe(predictions, actuals)
    # every day scores a perfect corr of 1.0 -> std across days is 0 -> guarded to 0.0
    assert score == 0.0


def test_consistently_inverse_ranks_score_negative():
    # actuals vary day to day, predictions are anti-correlated with slight
    # per-day noise so daily corr is negative but not identically -1 every
    # day (a constant -1 every day would trip the std==0 guard and read 0.0)
    rng = np.random.default_rng(2)
    n_days, n_targets = 20, 8
    actuals = pd.DataFrame(rng.normal(size=(n_days, n_targets)))
    predictions = -actuals + pd.DataFrame(rng.normal(scale=0.2, size=(n_days, n_targets)))
    score = spearman_sharpe(predictions, actuals)
    assert score < 0


def test_varying_correlation_gives_finite_score():
    rng = np.random.default_rng(1)
    n_days, n_targets = 20, 8
    actuals = pd.DataFrame(rng.normal(size=(n_days, n_targets)))
    predictions = actuals + pd.DataFrame(rng.normal(scale=0.5, size=(n_days, n_targets)))
    score = spearman_sharpe(predictions, actuals)
    assert np.isfinite(score)


def test_empty_dataframe_raises():
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        spearman_sharpe(empty, empty)


def test_shape_mismatch_raises():
    a = pd.DataFrame(np.random.rand(5, 3))
    b = pd.DataFrame(np.random.rand(5, 4))
    with pytest.raises(ValueError):
        spearman_sharpe(a, b)


def test_rows_with_fewer_than_two_valid_actuals_are_skipped():
    predictions = pd.DataFrame([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    actuals = pd.DataFrame([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    # second row has zero valid actuals and is skipped, leaving one perfect-corr day -> std guard -> 0.0
    score = spearman_sharpe(predictions, actuals)
    assert score == 0.0
