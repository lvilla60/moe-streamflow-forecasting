"""Simple streamflow baselines."""

import numpy as np


def persistence_baseline(X, target_channel=11, forecast_horizon=48):
    values = np.asarray(X)
    if values.ndim not in (2, 3) or values.shape[-1] <= target_channel:
        raise ValueError("X must have shape (336, channels) or (N, 336, channels)")
    last = values[..., -1, target_channel]
    return np.repeat(last[..., None], forecast_horizon, axis=-1)
