"""Evaluate sample-level hard MoE forecasts on the official validation split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import CaudalDataset
from src.inference import load_experts, load_router_artifact, predict_router_classes
from src.models import select_expert_predictions
from src.reporting import (experiment_directory, plot_class_frequency,
                           plot_confusion_matrix, plot_horizon_metric, save_json)
from src.routing import predict_expert_stack, read_router_labels, validate_label_rows
from src.training import (HorizonRegressionAccumulator, RegressionAccumulator,
                          prepare_batch)


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Input/train-001.h5")
    parser.add_argument("--lstm-checkpoint", required=True)
    parser.add_argument("--gru-checkpoint", required=True)
    parser.add_argument("--seq2seq-checkpoint", required=True)
    parser.add_argument("--informer-checkpoint", required=True)
    parser.add_argument("--router", required=True, choices=("rf", "lstm", "transformer"))
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--validation-labels")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", default="outputs/moe_validation.json")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")

    device = torch.device(args.device)
    expert_paths = [resolve(path) for path in (
        args.lstm_checkpoint, args.gru_checkpoint,
        args.seq2seq_checkpoint, args.informer_checkpoint,
    )]
    experts, normalization = load_experts(expert_paths, device)
    router, _ = load_router_artifact(
        args.router, resolve(args.router_checkpoint), device, normalization
    )

    label_lookup = None
    if args.validation_labels:
        ids, labels = read_router_labels(resolve(args.validation_labels), "validation")
        validate_label_rows(resolve(args.input), ids, "validation")
        label_lookup = dict(zip(ids.tolist(), labels.tolist()))

    dataset = CaudalDataset(resolve(args.input), split="validation")
    metrics = RegressionAccumulator()
    horizon_metrics = HorizonRegressionAccumulator()
    frequencies = np.zeros(4, dtype=np.int64)
    confusion = np.zeros((4, 4), dtype=np.int64) if label_lookup is not None else None
    try:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                prepared = prepare_batch(batch, normalization, device)
                expert_stack = predict_expert_stack(
                    experts, prepared["x"], prepared["decoder_start"]
                )
                classes = predict_router_classes(router, prepared["x"])
                prediction = normalization.inverse_y(
                    select_expert_predictions(expert_stack, classes)
                )
                raw_target = prepared["raw_y"].cpu().numpy()
                physical_prediction = prediction.cpu().numpy()
                metrics.update(raw_target, physical_prediction)
                horizon_metrics.update(raw_target, physical_prediction)
                class_array = classes.cpu().numpy()
                frequencies += np.bincount(class_array, minlength=4)
                if label_lookup is not None:
                    # Regression metrics cover every evaluated row; classification
                    # metrics cover only the intersection with supplied labels.
                    for sample_id, predicted_class in zip(batch["Id"], class_array):
                        true_class = label_lookup.get(int(sample_id))
                        if true_class is not None:
                            confusion[true_class, predicted_class] += 1
    finally:
        dataset.close()

    global_metrics = metrics.finalize()
    per_horizon = horizon_metrics.finalize()
    result = {
        "kind": "moe", "model": f"moe_{args.router}",
        "split": "validation",
        "n_samples": int(frequencies.sum()),
        "global": global_metrics,
        "MAE": float(global_metrics["mae"]),
        "RMSE": float(global_metrics["rmse"]),
        "NSE": float(global_metrics["nse"]),
        "per_horizon_mae": per_horizon["mae"],
        "per_horizon_rmse": per_horizon["rmse"],
        "router_class_frequencies": frequencies.tolist(),
    }
    if confusion is not None:
        matched = int(confusion.sum())
        result["router_accuracy"] = float(np.trace(confusion) / matched) if matched else None
        result["router_confusion_matrix"] = confusion.tolist()
        result["router_labeled_samples"] = matched
        result["router_unlabeled_samples"] = int(frequencies.sum()) - matched
        print(f"Router classification metrics cover {matched}/{int(frequencies.sum())} evaluated rows")
    output = resolve(args.output)
    save_json(output, result)
    experiment_name = args.experiment_name or f"moe_{args.router}"
    experiment = experiment_directory(resolve("outputs/experiments"), experiment_name)
    save_json(experiment / "config.json", {
        "kind": "moe", "model": f"moe_{args.router}",
        "experiment_name": experiment_name, "arguments": vars(args),
    })
    save_json(experiment / "metrics.json", result)
    plot_horizon_metric(experiment / "plots/mae_by_horizon.png",
                        result["per_horizon_mae"], "MAE")
    plot_horizon_metric(experiment / "plots/rmse_by_horizon.png",
                        result["per_horizon_rmse"], "RMSE")
    plot_class_frequency(experiment / "plots/router_class_frequency.png", frequencies)
    if confusion is not None:
        plot_confusion_matrix(experiment / "plots/router_confusion_matrix.png", confusion)
    print(json.dumps(result["global"], indent=2))
    print(f"Router class frequencies: {frequencies.tolist()}")
    print(f"Output: {output}")
    print(f"Experiment artifacts: {experiment}")


if __name__ == "__main__":
    main()
