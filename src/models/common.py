"""Shared neural-model building blocks."""

import math

import torch
from torch import nn


def validate_sequence(x, input_size):
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch, time, channels], got rank {x.ndim}")
    if x.size(-1) != input_size:
        raise ValueError(f"expected {input_size} input channels, got {x.size(-1)}")


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal relative-position encoding for batch-first tensors."""

    def __init__(self, d_model, max_length=1024):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError("sequence exceeds configured positional-encoding length")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)
