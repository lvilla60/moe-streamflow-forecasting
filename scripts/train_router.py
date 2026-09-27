"""Train a random-forest, LSTM, or Transformer router."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.checkpointing import load_model_from_checkpoint, save_checkpoint
from src.models import RandomForestRouter, create_router, flatten_router_features
from src.reporting import (experiment_directory, plot_accuracy_curve,
                           plot_class_frequency, plot_confusion_matrix,
                           plot_feature_importance, plot_loss_curve,
                           save_history_csv, save_json)
from src.routing import (parse_expert_subset, read_router_labels,
                         require_matching_expert_names, validate_label_rows)
from src.training import Normalization, set_seed


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class LabelledSequenceDataset(Dataset):
    def __init__(self, h5_path, ids, labels, normalization):
        self.h5_path = str(h5_path)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.normalization = normalization
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.h5_path, "r")
            self._pid = os.getpid()
        x = torch.as_tensor(self._file["X"][self.ids[index]], dtype=torch.float32)
        return self.normalization.normalize_x(x), int(self.labels[index])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state


def read_labels(path, required_split):
    return read_router_labels(path, required_split, return_expert_names=True)


def load_flat_features(h5_path, ids, normalization, chunk_size=256):
    """Allocate one float32 feature matrix; normalize bounded chunks in place."""
    mean = np.asarray(normalization.x_mean, dtype=np.float32)
    std = np.asarray(normalization.x_std, dtype=np.float32)
    safe_std = np.where(std == 0, 1.0, std)
    with h5py.File(h5_path, "r") as file:
        source = file["X"]
        shape = (len(ids), int(np.prod(source.shape[1:])))
        gib = np.prod(shape) * np.dtype(np.float32).itemsize / 1024 ** 3
        print(f"RF feature matrix: {gib:.2f} GiB float32 (additional forest/worker memory required)")
        if gib >= 1:
            print("WARNING: large RF allocation; consider --max-samples and a smaller --n-jobs.")
        features = np.empty(shape, dtype=np.float32)
        for start in range(0, len(ids), chunk_size):
            rows, inverse = np.unique(ids[start:start + chunk_size], return_inverse=True)
            raw = np.asarray(source[rows], dtype=np.float32)[inverse]
            raw -= mean
            raw /= safe_std
            features[start:start + len(raw)] = flatten_router_features(raw)
    return features


@torch.no_grad()
def evaluate_neural(model, loader, device):
    model.eval()
    loss_sum, count = 0.0, 0
    confusion = None
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("router must produce at least two class logits")
        if confusion is None:
            confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
        elif logits.shape[1] != confusion.shape[0]:
            raise ValueError("router class count changed during evaluation")
        if torch.any(labels < 0) or torch.any(labels >= logits.shape[1]):
            raise ValueError("router label is outside the configured class range")
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
        count += len(labels)
        np.add.at(confusion, (labels.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()), 1)
    if count == 0 or not np.isfinite(loss_sum):
        raise ValueError("router validation requires nonempty data and finite loss")
    return loss_sum / count, float(np.trace(confusion) / count), confusion


def fit_neural_router(model, loader, validation_loader, optimizer, *, epochs,
                      device, output, model_name, config, normalization,
                      expert_names=None):
    history = {"epoch": [], "train_loss": [], "validation_loss": [],
               "train_accuracy": [], "validation_accuracy": []}
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        loss_sum, count, correct = 0.0, 0, 0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("non-finite router training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
            correct += int((logits.detach().argmax(dim=1) == target).sum())
        if count == 0:
            raise ValueError("router training loader yielded no samples")
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(loss_sum / count))
        history["train_accuracy"].append(float(correct / count))
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={loss_sum / count:.6g}, "
              f"train_accuracy={correct / count:.6g}")

        save = epoch + 1 == epochs
        if validation_loader is not None:
            val_loss, accuracy, confusion = evaluate_neural(model, validation_loader, device)
            history["validation_loss"].append(val_loss)
            history["validation_accuracy"].append(accuracy)
            save = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            print(f"Validation loss={val_loss:.6g}, accuracy={accuracy:.6g}")
            print("Confusion matrix (true rows, predicted columns):")
            print(confusion)
        else:
            history["validation_loss"].append(None)
            history["validation_accuracy"].append(None)
        if save:
            save_checkpoint(
                output, model_name=model_name, model_config=config, model=model,
                optimizer=optimizer, epoch=epoch + 1,
                best_validation_loss=best_loss if validation_loader is not None else None,
                normalization=normalization.to_dict(), training_history=history,
                model_kind="router",
                selection="validation_loss" if validation_loader is not None else "final_epoch",
                expert_names=expert_names,
            )
    print("Saved best validation checkpoint" if validation_loader is not None
          else "No validation labels supplied: saved final model (not validation-selected)")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True, choices=("rf", "lstm", "transformer"))
    parser.add_argument("--input", default="Input/train-001.h5")
    parser.add_argument("--stats", default="outputs/train_stats.json")
    parser.add_argument("--labels", default="outputs/router_labels_train.npz")
    parser.add_argument("--validation-labels")
    parser.add_argument("--experts", help="validate labels against this ordered subset")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    if args.epochs <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("--epochs and --max-samples must be positive")

    set_seed(args.seed)
    h5_path = resolve(args.input)
    normalization = Normalization.from_json(resolve(args.stats))
    ids, labels, expert_names = read_labels(resolve(args.labels), "train")
    if args.experts:
        try:
            requested_experts = parse_expert_subset(args.experts)
        except ValueError as error:
            parser.error(str(error))
        try:
            require_matching_expert_names(
                expert_names, requested_experts, "--experts and training-label expert_names"
            )
        except ValueError as error:
            parser.error(str(error))
    num_classes = len(expert_names)
    validate_label_rows(h5_path, ids, "train")
    if args.max_samples is not None:
        ids, labels = ids[:args.max_samples], labels[:args.max_samples]
    validation = None
    if args.validation_labels:
        validation = read_labels(resolve(args.validation_labels), "validation")
        try:
            require_matching_expert_names(
                expert_names, validation[2], "training and validation label expert_names"
            )
        except ValueError as error:
            parser.error(str(error))
        validate_label_rows(h5_path, validation[0], "validation")
    output = resolve(args.output or f"checkpoints/router_{args.router}.pt")
    experiment_name = args.experiment_name or f"router_{args.router}"
    experiment = experiment_directory(resolve("outputs/experiments"), experiment_name)
    save_json(experiment / "config.json", {
        "kind": "router", "model": args.router, "experiment_name": experiment_name,
        "arguments": vars(args), "expert_names": list(expert_names),
    })

    if args.router == "rf":
        features = load_flat_features(h5_path, ids, normalization)
        router = RandomForestRouter(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs,
            num_classes=num_classes,
        ).fit(features, labels)
        del features
        output = output.with_suffix(".joblib")
        router.save(output, metadata={
            "normalization": normalization.to_dict(),
            "expert_names": list(expert_names),
        })
        result = {
            "kind": "router", "model": "rf", "validation_accuracy": None,
            "confusion_matrix": None, "predicted_class_frequency": None,
            "true_class_frequency": None, "n_estimators": int(args.n_estimators),
            "max_samples": int(len(ids)), "n_jobs": int(args.n_jobs),
            "expert_names": list(expert_names),
        }
        if validation is not None:
            from sklearn.metrics import accuracy_score, confusion_matrix
            val_ids, val_labels, _ = validation
            predictions = router.predict(load_flat_features(h5_path, val_ids, normalization))
            accuracy = float(accuracy_score(val_labels, predictions))
            confusion = confusion_matrix(
                val_labels, predictions, labels=np.arange(num_classes)
            )
            predicted_frequency = np.bincount(predictions, minlength=num_classes)
            true_frequency = np.bincount(val_labels, minlength=num_classes)
            result.update({
                "validation_accuracy": accuracy,
                "confusion_matrix": confusion.tolist(),
                "predicted_class_frequency": predicted_frequency.tolist(),
                "true_class_frequency": true_frequency.tolist(),
            })
            plot_confusion_matrix(experiment / "plots/confusion_matrix.png", confusion,
                                  expert_names)
            plot_class_frequency(experiment / "plots/class_frequency.png",
                                 predicted_frequency, true_frequency, expert_names)
            print(f"Validation accuracy: {accuracy:.6g}")
            print("Confusion matrix:")
            print(confusion)
        importances = getattr(router.model, "feature_importances_", None)
        if importances is not None and len(importances) % 12 == 0:
            channel_importance = np.asarray(importances).reshape(-1, 12).sum(axis=0)
            result["channel_feature_importance"] = channel_importance.tolist()
            plot_feature_importance(experiment / "plots/feature_importance.png",
                                    channel_importance, [str(i) for i in range(1, 13)])
        save_json(experiment / "metrics.json", result)
    else:
        config = (
            {"input_size": 12, "hidden_size": args.hidden_size,
             "num_layers": args.layers, "dropout": args.dropout,
             "num_classes": num_classes}
            if args.router == "lstm" else
            {"input_size": 12, "d_model": args.d_model, "n_heads": args.heads,
             "num_layers": args.layers, "d_ff": args.d_ff,
             "dropout": args.dropout, "num_classes": num_classes}
        )
        device = torch.device(args.device)
        model = create_router(args.router, **config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        dataset = LabelledSequenceDataset(h5_path, ids, labels, normalization)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
        val_dataset, val_loader = None, None
        if validation is not None:
            val_ids, val_labels, _ = validation
            val_dataset = LabelledSequenceDataset(h5_path, val_ids, val_labels, normalization)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
        try:
            history = fit_neural_router(
                model, loader, val_loader, optimizer, epochs=args.epochs,
                device=device, output=output, model_name=args.router,
                config=config, normalization=normalization,
                expert_names=expert_names,
            )
            save_history_csv(experiment / "history.csv", history)
            plot_loss_curve(experiment / "plots/loss_curve.png", history,
                            "Cross-entropy loss")
            plot_accuracy_curve(experiment / "plots/accuracy_curve.png", history)
            best_model, best_state = load_model_from_checkpoint(output, device)
            result = {
                "kind": "router", "model": args.router,
                "best_epoch": int(best_state["epoch"]),
                "validation_loss": None, "validation_accuracy": None,
                "confusion_matrix": None, "predicted_class_frequency": None,
                "true_class_frequency": None,
                "expert_names": list(expert_names),
            }
            if val_loader is not None:
                val_loss, accuracy, confusion = evaluate_neural(best_model, val_loader, device)
                predicted_frequency = confusion.sum(axis=0)
                true_frequency = confusion.sum(axis=1)
                result.update({
                    "validation_loss": float(val_loss),
                    "validation_accuracy": float(accuracy),
                    "confusion_matrix": confusion.tolist(),
                    "predicted_class_frequency": predicted_frequency.tolist(),
                    "true_class_frequency": true_frequency.tolist(),
                })
                plot_confusion_matrix(experiment / "plots/confusion_matrix.png", confusion,
                                      expert_names)
                plot_class_frequency(experiment / "plots/class_frequency.png",
                                     predicted_frequency, true_frequency, expert_names)
            save_json(experiment / "metrics.json", result)
        finally:
            dataset.close()
            if val_dataset is not None:
                val_dataset.close()
    print(f"Router artifact: {output}")
    print(f"Experiment artifacts: {experiment}")


if __name__ == "__main__":
    main()
