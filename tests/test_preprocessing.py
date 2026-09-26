import numpy as np

from src.preprocessing import RunningStats, standardize


def test_running_stats_matches_numpy_across_updates():
    values = np.arange(30, dtype=np.float64).reshape(5, 2, 3)
    stats = RunningStats()
    stats.update(values[:2])
    stats.update(values[2:])
    mean, std = stats.finalize()
    np.testing.assert_allclose(mean, values.reshape(-1, 3).mean(axis=0))
    np.testing.assert_allclose(std, values.reshape(-1, 3).std(axis=0))


def test_standardize_has_zero_mean_and_unit_std():
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    result = standardize(values, mean, std)
    np.testing.assert_allclose(result.mean(axis=0), 0)
    np.testing.assert_allclose(result.std(axis=0), 1)


def test_zero_variance_channel_is_safe():
    values = np.array([[1.0, 2.0], [3.0, 2.0]])
    mean, std = RunningStats().update(values).finalize()
    assert std[1] == 0
    result = standardize(values, mean, std)
    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result[:, 1], 0)
