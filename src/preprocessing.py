"""Streaming preprocessing utilities."""

import numpy as np


class RunningStats:
    """Per-channel population mean and standard deviation accumulator."""

    def __init__(self, channels=None):
        self.count = 0
        self.mean = None if channels is None else np.zeros(channels, dtype=np.float64)
        self.m2 = None if channels is None else np.zeros(channels, dtype=np.float64)

    def update(self, batch):
        values = np.asarray(batch, dtype=np.float64)
        if values.ndim < 1:
            raise ValueError("batch must have a channel dimension")
        flat = values.reshape(-1, values.shape[-1])
        if self.mean is None:
            self.mean = np.zeros(flat.shape[1], dtype=np.float64)
            self.m2 = np.zeros(flat.shape[1], dtype=np.float64)
        if flat.shape[1] != self.mean.size:
            raise ValueError("channel count does not match existing statistics")
        batch_count = flat.shape[0]
        if batch_count == 0:
            return self
        batch_mean = flat.mean(axis=0)
        batch_m2 = np.sum((flat - batch_mean) ** 2, axis=0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta ** 2 * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total
        return self

    def finalize(self):
        if self.count == 0:
            raise ValueError("cannot finalize empty statistics")
        std = np.sqrt(self.m2 / self.count)
        return self.mean.copy(), std


def standardize(X, mean, std):
    values = np.asarray(X)
    mean = np.asarray(mean)
    std = np.asarray(std)
    safe_std = np.where(std == 0, 1.0, std)
    return (values - mean) / safe_std
