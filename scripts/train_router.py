"""Train a random-forest, LSTM, or Transformer router."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.checkpointing import save_checkpoint
from src.models import RandomForestRouter, create_router, flatten_router_features
from src.routing import read_router_labels, validate_label_rows
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
    return read_router_labels(path, required_split)


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
    confusion = np.zeros((4, 4), dtype=np.int64)
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
        count += len(labels)
        np.add.at(confusion, (labels.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()), 1)
    if count == 0 or not np.isfinite(loss_sum):
        raise ValueError("router validation requires nonempty data and finite loss")
    return loss_sum / count, float(np.trace(confusion) / count), confusion


def fit_neural_router(model, loader, validation_loader, optimizer, *, epochs,
                      device, output, model_name, config, normalization):
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), target)
            if not torch.isfinite(loss):
                raise ValueError("non-finite router training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        if count == 0:
            raise ValueError("router training loader yielded no samples")
        history["train_loss"].append(loss_sum / count)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={loss_sum / count:.6g}")

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
        if save:
            save_checkpoint(
                output, model_name=model_name, model_config=config, model=model,
                optimizer=optimizer, epoch=epoch + 1,
                best_validation_loss=best_loss if validation_loader is not None else None,
                normalization=normalization.to_dict(), training_history=history,
                model_kind="router",
                selection="validation_loss" if validation_loader is not None else "final_epoch",
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
    args = parser.parse_args()
    if args.epochs <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("--epochs and --max-samples must be positive")

    set_seed(args.seed)
    h5_path = resolve(args.input)
    normalization = Normalization.from_json(resolve(args.stats))
    ids, labels = read_labels(resolve(args.labels), "train")
    validate_label_rows(h5_path, ids, "train")
    if args.max_samples is not None:
        ids, labels = ids[:args.max_samples], labels[:args.max_samples]
    validation = None
    if args.validation_labels:
        validation = read_labels(resolve(args.validation_labels), "validation")
        validate_label_rows(h5_path, validation[0], "validation")
    output = resolve(args.output or f"checkpoints/router_{args.router}.pt")

    if args.router == "rf":
        features = load_flat_features(h5_path, ids, normalization)
        router = RandomForestRouter(
            n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs
        ).fit(features, labels)
        del features
        output = output.with_suffix(".joblib")
        router.save(output, metadata={"normalization": normalization.to_dict()})
        if validation is not None:
            from sklearn.metrics import accuracy_score, confusion_matrix
            val_ids, val_labels = validation
            predictions = router.predict(load_flat_features(h5_path, val_ids, normalization))
            print(f"Validation accuracy: {accuracy_score(val_labels, predictions):.6g}")
            print("Confusion matrix:")
            print(confusion_matrix(val_labels, predictions, labels=np.arange(4)))
    else:
        config = (
            {"input_size": 12, "hidden_size": args.hidden_size,
             "num_layers": args.layers, "dropout": args.dropout, "num_classes": 4}
            if args.router == "lstm" else
            {"input_size": 12, "d_model": args.d_model, "n_heads": args.heads,
             "num_layers": args.layers, "d_ff": args.d_ff,
             "dropout": args.dropout, "num_classes": 4}
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
            val_ids, val_labels = validation
            val_dataset = LabelledSequenceDataset(h5_path, val_ids, val_labels, normalization)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
        try:
            fit_neural_router(
                model, loader, val_loader, optimizer, epochs=args.epochs,
                device=device, output=output, model_name=args.router,
                config=config, normalization=normalization,
            )
        finally:
            dataset.close()
            if val_dataset is not None:
                val_dataset.close()
    print(f"Router artifact: {output}")


if __name__ == "__main__":
    main()
