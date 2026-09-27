"""Generate sample-level best-expert classes from frozen expert checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import CaudalDataset
from src.inference import load_experts
from src.routing import best_expert_labels, parse_expert_subset, predict_expert_stack
from src.training import Normalization, prepare_batch


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Input/train-001.h5")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--experts", help="ordered comma-separated expert subset")
    parser.add_argument("--lstm-checkpoint")
    parser.add_argument("--gru-checkpoint")
    parser.add_argument("--seq2seq-checkpoint")
    parser.add_argument("--informer-checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="default: outputs/router_labels_<split>.npz")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")

    try:
        expert_names = parse_expert_subset(args.experts)
    except ValueError as error:
        parser.error(str(error))
    checkpoint_by_name = {
        "lstm": args.lstm_checkpoint,
        "gru": args.gru_checkpoint,
        "seq2seq": args.seq2seq_checkpoint,
        "informer": args.informer_checkpoint,
    }
    missing = [name for name in expert_names if not checkpoint_by_name[name]]
    if missing:
        parser.error(f"missing checkpoint argument(s) for selected experts: {missing}")
    device = torch.device(args.device)
    checkpoint_paths = [resolve(checkpoint_by_name[name]) for name in expert_names]
    missing_paths = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing_paths:
        parser.error(f"selected expert checkpoint(s) not found: {missing_paths}")
    experts, normalization = load_experts(
        checkpoint_paths, device, expert_names=expert_names
    )

    dataset = CaudalDataset(resolve(args.input), split=args.split)
    ids, basin_ids, labels, errors = [], [], [], []
    try:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                prepared = prepare_batch(batch, normalization, device)
                normalized_predictions = predict_expert_stack(
                    experts, prepared["x"], prepared["decoder_start"]
                )
                physical_predictions = normalization.inverse_y(normalized_predictions)
                batch_labels, batch_errors = best_expert_labels(
                    prepared["raw_y"], physical_predictions
                )
                ids.append(torch.as_tensor(batch["Id"]).cpu().numpy())
                basin_ids.append(torch.as_tensor(batch["basin_id"]).cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
                errors.append(batch_errors.cpu().numpy())
    finally:
        dataset.close()

    if not labels:
        raise ValueError("no router labels were generated")
    output = resolve(args.output or f"outputs/router_labels_{args.split}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, Id=np.concatenate(ids).astype(np.int64),
        basin_id=np.concatenate(basin_ids).astype(np.int64),
        best_expert=np.concatenate(labels).astype(np.int64),
        expert_errors=np.concatenate(errors).astype(np.float32),
        expert_names=np.asarray(expert_names), split=np.asarray(args.split),
    )
    print(f"Generated {sum(len(item) for item in labels):,} {args.split} labels")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
