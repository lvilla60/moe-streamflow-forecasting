"""Evaluate persistence on the validation split using bounded-memory scans."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.baseline import persistence_baseline
from src.preprocessing import RunningStats


ROOT = Path(__file__).resolve().parents[1]
TARGET_CHANNEL = 11


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Input/train-001.h5")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", default="outputs/persistence_validation.json")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    with h5py.File(input_path, "r") as h5_file:
        validation_indices = np.flatnonzero(h5_file["split"][:] == 1)
        n_samples = len(validation_indices)
        n_chunks = (n_samples + args.chunk_size - 1) // args.chunk_size

        # First pass: calculate the global validation target mean for NSE.
        y_stats = RunningStats(channels=1)
        for start in range(0, n_samples, args.chunk_size):
            indices = validation_indices[start:start + args.chunk_size]
            y_stats.update(h5_file["y"][indices][..., None])
        y_mean, _ = y_stats.finalize()
        global_y_mean = y_mean[0]

        absolute_error_sum = 0.0
        squared_error_sum = 0.0
        denominator_sum = 0.0
        horizon_absolute_sum = np.zeros(48, dtype=np.float64)
        horizon_squared_sum = np.zeros(48, dtype=np.float64)

        # Second pass: accumulate error sums globally, never per-chunk averages.
        for chunk_number, start in enumerate(range(0, n_samples, args.chunk_size), start=1):
            indices = validation_indices[start:start + args.chunk_size]
            x_batch = h5_file["X"][indices]
            y_true = h5_file["y"][indices].astype(np.float64, copy=False)
            y_pred = persistence_baseline(x_batch, target_channel=TARGET_CHANNEL, forecast_horizon=48)
            error = y_true - y_pred

            absolute_error_sum += np.sum(np.abs(error), dtype=np.float64)
            squared_error_sum += np.sum(error ** 2, dtype=np.float64)
            denominator_sum += np.sum((y_true - global_y_mean) ** 2, dtype=np.float64)
            horizon_absolute_sum += np.sum(np.abs(error), axis=0, dtype=np.float64)
            horizon_squared_sum += np.sum(error ** 2, axis=0, dtype=np.float64)
            if chunk_number % 100 == 0 or chunk_number == n_chunks:
                print(f"Evaluated {min(start + len(indices), n_samples):,}/{n_samples:,} validation samples")

    if denominator_sum == 0:
        raise ValueError("NSE is undefined because validation targets have zero variance")

    value_count = n_samples * 48
    result = {
        "split": "validation",
        "n_samples": int(n_samples),
        "global": {
            "mae": float(absolute_error_sum / value_count),
            "rmse": float(np.sqrt(squared_error_sum / value_count)),
            "nse": float(1.0 - squared_error_sum / denominator_sum),
        },
        "per_horizon": {
            "mae": (horizon_absolute_sum / n_samples).tolist(),
            "rmse": np.sqrt(horizon_squared_sum / n_samples).tolist(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Validation samples: {n_samples:,}")
    print(
        "Global MAE/RMSE/NSE: "
        f"{result['global']['mae']:.6g} / {result['global']['rmse']:.6g} / {result['global']['nse']:.6g}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
