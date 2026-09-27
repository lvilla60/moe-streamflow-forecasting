"""Best-expert label generation for sample-level hard routing."""

import numpy as np
import torch
import h5py

from .training import forward_expert


EXPERT_NAMES = ("lstm", "gru", "seq2seq", "informer")
LEGACY_EXPERT_NAMES = ("lstm", "gru", "seq2seq_attention", "informer")
EXPERT_MODEL_NAMES = {
    "lstm": "lstm",
    "gru": "gru",
    "seq2seq": "seq2seq_attention",
    "informer": "informer",
}


def parse_expert_subset(value=None):
    """Return a validated, ordered tuple of public expert names."""
    if value is None:
        names = EXPERT_NAMES
    elif isinstance(value, str):
        names = tuple(name.strip() for name in value.split(","))
    else:
        names = tuple(str(name) for name in value)
    if len(names) < 2:
        raise ValueError("at least two experts are required for routing")
    if any(not name for name in names):
        raise ValueError("expert names cannot be empty")
    unknown = [name for name in names if name not in EXPERT_NAMES]
    if unknown:
        raise ValueError(
            f"unknown expert name(s) {unknown}; expected names from {list(EXPERT_NAMES)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("expert names must not contain duplicates")
    return names


def normalize_stored_expert_names(raw_names):
    """Validate stored metadata, including the historical Seq2Seq name."""
    names = tuple(str(name) for name in raw_names)
    if names == LEGACY_EXPERT_NAMES:
        return EXPERT_NAMES
    return parse_expert_subset(names)


def require_matching_expert_names(reference, candidate, description="expert mappings"):
    reference = parse_expert_subset(reference)
    candidate = parse_expert_subset(candidate)
    if candidate != reference:
        raise ValueError(f"{description} must match exactly and preserve order")
    return reference


def read_router_labels(path, required_split, return_expert_names=False):
    """Read labels in original-row order and validate their local class mapping."""
    with np.load(path, allow_pickle=False) as data:
        required = {"split", "expert_names", "Id", "best_expert"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"router labels are missing required fields: {missing}")
        if str(data["split"].item()) != required_split:
            raise ValueError(f"expected {required_split} labels")
        raw_names = data["expert_names"]
        if raw_names.ndim != 1:
            raise ValueError("expert_names must be a one-dimensional ordered array")
        expert_names = normalize_stored_expert_names(raw_names.tolist())
        ids, labels = data["Id"].copy(), data["best_expert"].copy()
        errors = data["expert_errors"].copy() if "expert_errors" in data.files else None
    if (ids.ndim != 1 or labels.shape != ids.shape or not len(ids)
            or ids.dtype.kind not in "iu" or labels.dtype.kind not in "iu"):
        raise ValueError("Id and best_expert must be nonempty integer vectors of equal length")
    if (np.any(ids < 0) or len(np.unique(ids)) != len(ids)
            or np.any(labels < 0) or np.any(labels >= len(expert_names))):
        raise ValueError("invalid/duplicate Id or best_expert outside class range")
    if errors is not None:
        if (errors.shape != (len(ids), len(expert_names))
                or errors.dtype.kind not in "fiu" or not np.isfinite(errors).all()):
            raise ValueError("expert_errors must be finite with one column per expert_name")
    order = np.argsort(ids)
    result = (ids[order], labels[order])
    return (*result, expert_names) if return_expert_names else result


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
    if not experts:
        raise ValueError("at least one expert is required")
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
