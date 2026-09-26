"""Best-expert label generation for sample-level hard routing."""

import numpy as np
import torch
import h5py

from .training import forward_expert


EXPERT_NAMES = ("lstm", "gru", "seq2seq_attention", "informer")


def read_router_labels(path, required_split):
    """Read labels in original-row order and enforce the fixed class mapping."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["split"].item()) != required_split:
            raise ValueError(f"expected {required_split} labels")
        if tuple(data["expert_names"].tolist()) != EXPERT_NAMES:
            raise ValueError("router label expert_names do not match the class mapping")
        ids, labels = data["Id"].copy(), data["best_expert"].copy()
    if (ids.ndim != 1 or labels.shape != ids.shape or not len(ids)
            or ids.dtype.kind not in "iu" or labels.dtype.kind not in "iu"):
        raise ValueError("Id and best_expert must be nonempty integer vectors of equal length")
    if (np.any(ids < 0) or len(np.unique(ids)) != len(ids)
            or np.any(labels < 0) or np.any(labels >= len(EXPERT_NAMES))):
        raise ValueError("invalid/duplicate Id or best_expert outside class range")
    order = np.argsort(ids)
    return ids[order], labels[order]


def validate_label_rows(h5_path, ids, split):
    """Validate row membership, reading only the referenced split flags."""
    code = {"train": 0, "validation": 1}[split]
    with h5py.File(h5_path, "r") as file:
        flags = file["split"]
        if np.any(ids < 0) or np.any(ids >= len(flags)):
            raise ValueError("label Id is outside the HDF5 row range")
        for start in range(0, len(ids), 4096):
            rows = np.unique(ids[start:start + 4096])
            if np.any(flags[rows] != code):
                raise ValueError(f"{split} labels include rows outside the official {split} split")


def best_expert_labels(y_true, expert_predictions):
    """Choose the first minimum-MAE expert for every complete forecast sample."""
    if torch.is_tensor(expert_predictions):
        if expert_predictions.ndim != 3:
            raise ValueError("expert_predictions must have shape [batch, experts, horizon]")
        if y_true.shape != (expert_predictions.size(0), expert_predictions.size(2)):
            raise ValueError("y_true must have shape [batch, horizon]")
        if not (torch.isfinite(y_true).all() and torch.isfinite(expert_predictions).all()):
            raise ValueError("router labels require finite targets and predictions")
        errors = (expert_predictions - y_true.unsqueeze(1)).abs().mean(dim=2)
        return errors.argmin(dim=1), errors
    predictions = np.asarray(expert_predictions)
    targets = np.asarray(y_true)
    if predictions.ndim != 3:
        raise ValueError("expert_predictions must have shape [batch, experts, horizon]")
    if targets.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("y_true must have shape [batch, horizon]")
    if not (np.isfinite(targets).all() and np.isfinite(predictions).all()):
        raise ValueError("router labels require finite targets and predictions")
    errors = np.abs(predictions - targets[:, None, :]).mean(axis=2)
    return errors.argmin(axis=1), errors


@torch.no_grad()
def predict_expert_stack(experts, x, decoder_start):
    if len(experts) != 4:
        raise ValueError("exactly four experts are required in class-map order")
    for expert in experts:
        expert.eval()
    predictions = [
        forward_expert(
            expert, x, target=None, teacher_forcing_ratio=0.0,
            decoder_start=decoder_start,
        )
        for expert in experts
    ]
    horizons = {prediction.shape for prediction in predictions}
    if len(horizons) != 1:
        raise ValueError("all expert prediction shapes must agree")
    return torch.stack(predictions, dim=1)
