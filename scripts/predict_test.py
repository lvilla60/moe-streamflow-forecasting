"""Generate physical-unit hard-MoE forecasts for test.h5."""

import argparse
import csv
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import CaudalDataset
from src.inference import load_experts, load_router_artifact, predict_router_classes
from src.models import select_expert_predictions
from src.routing import predict_expert_stack
from src.training import prepare_batch


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Input/test.h5")
    parser.add_argument("--lstm-checkpoint", required=True)
    parser.add_argument("--gru-checkpoint", required=True)
    parser.add_argument("--seq2seq-checkpoint", required=True)
    parser.add_argument("--informer-checkpoint", required=True)
    parser.add_argument("--router", required=True, choices=("rf", "lstm", "transformer"))
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, help="write only this many batches for smoke/debug runs")
    parser.add_argument("--output", default="outputs/test_predictions.csv")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")

    device = torch.device(args.device)
    experts, normalization = load_experts([
        resolve(args.lstm_checkpoint), resolve(args.gru_checkpoint),
        resolve(args.seq2seq_checkpoint), resolve(args.informer_checkpoint),
    ], device)
    if any(expert.forecast_horizon != 48 for expert in experts):
        raise ValueError("test CSV requires expert forecast_horizon=48")
    router, _ = load_router_artifact(
        args.router, resolve(args.router_checkpoint), device, normalization
    )
    if args.max_batches is not None:
        print(f"WARNING: --max-batches={args.max_batches}; output may be a partial test submission.")

    dataset = CaudalDataset(resolve(args.input), split="test")
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Id", *(f"q_{index:02d}" for index in range(1, 49))])
            with torch.no_grad():
                for batch in islice(loader, args.max_batches):
                    prepared = prepare_batch(batch, normalization, device, require_target=False)
                    expert_stack = predict_expert_stack(
                        experts, prepared["x"], prepared["decoder_start"]
                    )
                    classes = predict_router_classes(router, prepared["x"])
                    predictions = normalization.inverse_y(
                        select_expert_predictions(expert_stack, classes)
                    ).cpu().numpy()
                    if not torch.isfinite(torch.as_tensor(predictions)).all():
                        raise ValueError("non-finite test predictions; output CSV is incomplete")
                    for sample_id, prediction in zip(batch["Id"], predictions):
                        writer.writerow([int(sample_id), *map(float, prediction)])
                        written += 1
    finally:
        dataset.close()
    print(f"Wrote {written:,}/{len(dataset):,} test forecasts to {output}")


if __name__ == "__main__":
    main()
