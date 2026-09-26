"""Shared normalization, training, and bounded-memory evaluation utilities."""

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpointing import save_checkpoint


def set_seed(seed):
    """Set Python, NumPy, torch, and available CUDA seeds (best effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Normalization:
    x_mean: list
    x_std: list
    y_mean: float
    y_std: float

    def __post_init__(self):
        if not len(self.x_mean) or len(self.x_mean) != len(self.x_std):
            raise ValueError("x_mean and x_std must have the same length")
        if not np.isfinite([*self.x_mean, *self.x_std, self.y_mean, self.y_std]).all():
            raise ValueError("normalization statistics must be finite")
        if np.any(np.asarray(self.x_std) < 0) or self.y_std <= 0:
            raise ValueError("x_std must be nonnegative and y_std must be positive")

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as file:
            values = json.load(file)
        if values.get("split") != "train":
            raise ValueError("normalization statistics must declare split='train'")
        return cls(values["x"]["mean"], values["x"]["std"],
                   values["y"]["mean"], values["y"]["std"])

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return {
            "x_mean": [float(value) for value in self.x_mean],
            "x_std": [float(value) for value in self.x_std],
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
        }

    def _x_tensors(self, reference):
        mean = reference.new_tensor(self.x_mean)
        std = reference.new_tensor(self.x_std)
        safe_std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, safe_std

    def normalize_x(self, x):
        mean, std = self._x_tensors(x)
        return (x - mean) / std

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean

    def decoder_start(self, raw_x, target_channel=11):
        return self.normalize_y(raw_x[:, -1, target_channel])


def prepare_batch(batch, normalization, device, require_target=True):
    raw_x = torch.as_tensor(batch["X"], dtype=torch.float32, device=device)
    prepared = {
        "x": normalization.normalize_x(raw_x),
        "decoder_start": normalization.decoder_start(raw_x),
        "raw_x": raw_x,
    }
    if "y" in batch:
        raw_y = torch.as_tensor(batch["y"], dtype=torch.float32, device=device)
        prepared["raw_y"] = raw_y
        prepared["target"] = normalization.normalize_y(raw_y)
    elif require_target:
        raise ValueError("batch does not contain target y")
    return prepared


def forward_expert(model, x, target=None, teacher_forcing_ratio=0.0, decoder_start=None):
    if getattr(model, "requires_decoder_start", False):
        return model(
            x, target=target, teacher_forcing_ratio=teacher_forcing_ratio,
            decoder_start=decoder_start,
        )
    return model(x)


class RegressionAccumulator:
    """Global regression metrics without retaining predictions or targets."""

    def __init__(self):
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.target_mean = 0.0
        self.target_m2 = 0.0

    def update(self, target, prediction):
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if target.shape != prediction.shape:
            raise ValueError("target and prediction shapes must match")
        if target.size == 0 or not (np.isfinite(target).all() and np.isfinite(prediction).all()):
            raise ValueError("metrics require nonempty, finite targets and predictions")
        target, prediction = target.reshape(-1), prediction.reshape(-1)
        error = target - prediction
        self.absolute_error += float(np.abs(error).sum())
        self.squared_error += float(np.square(error).sum())
        batch_count = target.size
        batch_mean = float(target.mean())
        batch_m2 = float(np.square(target - batch_mean).sum())
        total = self.count + batch_count
        delta = batch_mean - self.target_mean
        self.target_m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.target_mean += delta * batch_count / total
        self.count = total

    def finalize(self):
        if self.count == 0:
            raise ValueError("cannot finalize empty metrics")
        if self.target_m2 == 0:
            raise ValueError("NSE is undefined when targets have zero variance")
        return {
            "mae": self.absolute_error / self.count,
            "rmse": (self.squared_error / self.count) ** 0.5,
            "nse": 1.0 - self.squared_error / self.target_m2,
        }


class HorizonRegressionAccumulator:
    """Per-horizon MAE and RMSE using memory proportional to the horizon."""

    def __init__(self):
        self.count = 0
        self.absolute_error = None
        self.squared_error = None

    def update(self, target, prediction):
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if target.shape != prediction.shape or target.ndim != 2:
            raise ValueError("per-horizon metrics require matching [batch, horizon] arrays")
        if target.size == 0 or not (np.isfinite(target).all() and np.isfinite(prediction).all()):
            raise ValueError("metrics require nonempty, finite targets and predictions")
        error = target - prediction
        if self.absolute_error is None:
            self.absolute_error = np.zeros(target.shape[1], dtype=np.float64)
            self.squared_error = np.zeros(target.shape[1], dtype=np.float64)
        if target.shape[1] != len(self.absolute_error):
            raise ValueError("forecast horizon changed during evaluation")
        self.absolute_error += np.abs(error).sum(axis=0)
        self.squared_error += np.square(error).sum(axis=0)
        self.count += target.shape[0]

    def finalize(self):
        if self.count == 0:
            raise ValueError("cannot finalize empty metrics")
        return {
            "mae": (self.absolute_error / self.count).tolist(),
            "rmse": np.sqrt(self.squared_error / self.count).tolist(),
        }


def train_one_epoch(model, loader, optimizer, normalization, device,
                    teacher_forcing_ratio=0.0, gradient_clip=None,
                    max_batches=None):
    model.train()
    total_loss = 0.0
    total_values = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        prepared = prepare_batch(batch, normalization, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_expert(
            model, prepared["x"], prepared["target"], teacher_forcing_ratio,
            prepared["decoder_start"],
        )
        if prediction.shape != prepared["target"].shape:
            raise ValueError("prediction and target shapes must match")
        loss = nn.functional.mse_loss(prediction, prepared["target"])
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        values = prepared["target"].numel()
        total_loss += float(loss.detach()) * values
        total_values += values
    if total_values == 0:
        raise ValueError("training loader yielded no batches")
    return total_loss / total_values


@torch.no_grad()
def validate_expert(model, loader, normalization, device, max_batches=None,
                    include_per_horizon=False):
    model.eval()
    total_loss = 0.0
    total_values = 0
    metrics = RegressionAccumulator()
    horizon_metrics = HorizonRegressionAccumulator() if include_per_horizon else None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        prepared = prepare_batch(batch, normalization, device)
        prediction = forward_expert(
            model, prepared["x"], target=None, teacher_forcing_ratio=0.0,
            decoder_start=prepared["decoder_start"],
        )
        if prediction.shape != prepared["target"].shape:
            raise ValueError("prediction and target shapes must match")
        loss = nn.functional.mse_loss(prediction, prepared["target"])
        values = prepared["target"].numel()
        total_loss += float(loss) * values
        total_values += values
        physical_prediction = normalization.inverse_y(prediction)
        raw_target = prepared["raw_y"].cpu().numpy()
        physical_prediction = physical_prediction.cpu().numpy()
        metrics.update(raw_target, physical_prediction)
        if horizon_metrics is not None:
            horizon_metrics.update(raw_target, physical_prediction)
    if total_values == 0:
        raise ValueError("validation loader yielded no batches")
    result = (total_loss / total_values, metrics.finalize())
    return (*result, horizon_metrics.finalize()) if horizon_metrics is not None else result


def fit_expert(model, train_loader, validation_loader, normalization, *, model_name,
               model_config, epochs, learning_rate=1e-3, weight_decay=1e-3,
               device="cpu", gradient_clip=1.0, teacher_forcing_ratio=0.0,
               max_train_batches=None, max_validation_batches=None,
               checkpoint_path=None, resume_checkpoint=None):
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = {"epoch": [], "train_loss": [], "validation_loss": [],
               "validation_mae": [], "validation_rmse": [], "validation_nse": []}
    best_loss = float("inf")
    start_epoch = 0
    if resume_checkpoint is not None:
        from .checkpointing import load_checkpoint
        state = load_checkpoint(resume_checkpoint, map_location=device)
        if (state.get("model_kind", "expert") != "expert"
                or state["model_name"] != model_name
                or state["model_config"] != model_config):
            raise ValueError("resume model name/config must exactly match the checkpoint")
        if state.get("normalization") != normalization.to_dict():
            raise ValueError("resume normalization must match the checkpoint")
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is None:
            raise ValueError("resume checkpoint is missing optimizer state")
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state.get("training_history", history)
        best_loss = state.get("best_validation_loss", best_loss)
        start_epoch = state.get("epoch", 0)
        # A resumed run may never improve. Preserve its existing best model
        # even when the user chooses a new output directory.
        if checkpoint_path is not None:
            destination = Path(checkpoint_path)
            if destination.resolve() != Path(resume_checkpoint).resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resume_checkpoint, destination)

    completed_epochs = len(history.get("train_loss", []))
    history.setdefault("epoch", list(range(1, completed_epochs + 1)))
    history.setdefault("validation_mae", [None] * completed_epochs)
    history.setdefault("validation_rmse", [None] * completed_epochs)
    history.setdefault("validation_nse", [None] * completed_epochs)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, normalization, device,
            teacher_forcing_ratio, gradient_clip, max_train_batches,
        )
        validation_loss, physical_metrics = validate_expert(
            model, validation_loader, normalization, device, max_validation_batches
        )
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["validation_loss"].append(validation_loss)
        history["validation_mae"].append(float(physical_metrics["mae"]))
        history["validation_rmse"].append(float(physical_metrics["rmse"]))
        history["validation_nse"].append(float(physical_metrics["nse"]))
        if validation_loss < best_loss:
            best_loss = validation_loss
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path, model_name=model_name, model_config=model_config,
                    model=model, optimizer=optimizer, epoch=epoch + 1,
                    best_validation_loss=best_loss,
                    normalization=normalization.to_dict(), training_history=history,
                )
        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6g}, "
            f"validation_loss={validation_loss:.6g}, "
            f"MAE={physical_metrics['mae']:.6g}, RMSE={physical_metrics['rmse']:.6g}, "
            f"NSE={physical_metrics['nse']:.6g}"
        )
    return history
