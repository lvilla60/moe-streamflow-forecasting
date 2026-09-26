import numpy as np

from src.baseline import persistence_baseline


def test_single_sample_uses_channel_11_last_timestep():
    X = np.zeros((336, 12))
    X[-1, 11] = 7.5
    X[-1, 10] = 99
    result = persistence_baseline(X)
    assert result.shape == (48,)
    np.testing.assert_array_equal(result, 7.5)


def test_batch_returns_one_forecast_per_sample():
    X = np.zeros((2, 336, 12))
    X[0, -1, 11] = 3
    X[1, -1, 11] = 8
    result = persistence_baseline(X)
    assert result.shape == (2, 48)
    np.testing.assert_array_equal(result[0], 3)
    np.testing.assert_array_equal(result[1], 8)
