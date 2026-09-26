import numpy as np
import pytest

from src.metrics import mae, nse, rmse


def test_perfect_prediction():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert mae(values, values) == 0.0
    assert rmse(values, values) == 0.0
    assert nse(values, values) == 1.0


def test_known_mae_and_rmse():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(np.sqrt(5 / 3))


def test_nse_zero_denominator_raises():
    with pytest.raises(ValueError, match="zero variance"):
        nse([2, 2], [2, 3])
