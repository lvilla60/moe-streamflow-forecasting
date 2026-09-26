"""Compute streaming normalization statistics from the training split."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.preprocessing import RunningStats


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Input/train-001.h5")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", default="outputs/train_stats.json")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    x_stats = RunningStats(channels=12)
    y_stats = RunningStats(channels=1)

    with h5py.File(input_path, "r") as h5_file:
        train_indices = np.flatnonzero(h5_file["split"][:] == 0)
        n_samples = len(train_indices)
        n_chunks = (n_samples + args.chunk_size - 1) // args.chunk_size

        for chunk_number, start in enumerate(range(0, n_samples, args.chunk_size), start=1):
            indices = train_indices[start:start + args.chunk_size]
            x_stats.update(h5_file["X"][indices])
            y_stats.update(h5_file["y"][indices][..., None])
            if chunk_number % 100 == 0 or chunk_number == n_chunks:
                print(f"Processed {min(start + len(indices), n_samples):,}/{n_samples:,} train samples")

    x_mean, x_std = x_stats.finalize()
    y_mean, y_std = y_stats.finalize()
    result = {
        "source": "Input/train-001.h5",
        "split": "train",
        "n_samples": int(n_samples),
        "x": {"mean": x_mean.tolist(), "std": x_std.tolist()},
        "y": {"mean": float(y_mean[0]), "std": float(y_std[0])},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Samples processed: {n_samples:,}")
    print(f"X mean/std shape: {x_mean.shape} / {x_std.shape}")
    print(f"y mean/std: {float(y_mean[0])} / {float(y_std[0])}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
