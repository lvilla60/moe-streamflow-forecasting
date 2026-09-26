"""Minimal model checkpoint persistence."""

from pathlib import Path

import torch

from .models import create_expert, create_router


def save_checkpoint(path, *, model_name, model_config, model, optimizer=None,
                    epoch=0, best_validation_loss=float("inf"), normalization=None,
                    training_history=None, model_kind="expert", selection="validation_loss"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_kind": model_kind,
        "model_name": model_name,
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "best_validation_loss": (None if best_validation_loss is None
                                 else float(best_validation_loss)),
        "selection": selection,
        "normalization": normalization,
        "training_history": training_history or {"train_loss": [], "validation_loss": []},
    }, path)


def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_model_from_checkpoint(path, map_location="cpu"):
    checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint.get("model_kind", "expert") == "router":
        model = create_router(checkpoint["model_name"], **checkpoint["model_config"])
    else:
        model = create_expert(checkpoint["model_name"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint
