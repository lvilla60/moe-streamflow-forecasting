"""Artifact loading and router inference shared by MoE command-line workflows."""

import numpy as np
import torch
from torch import nn

from .checkpointing import load_model_from_checkpoint
from .models import RandomForestRouter, flatten_router_features
from .routing import EXPERT_NAMES
from .training import Normalization


def load_experts(checkpoint_paths, device):
    if len(checkpoint_paths) != 4:
        raise ValueError("four expert checkpoints are required")
    experts, states = [], []
    for expected_name, path in zip(EXPERT_NAMES, checkpoint_paths):
        model, state = load_model_from_checkpoint(path, device)
        if state.get("model_kind", "expert") != "expert" or state["model_name"] != expected_name:
            raise ValueError(f"expected {expected_name} checkpoint, got {state['model_name']}")
        model.requires_grad_(False)
        experts.append(model)
        states.append(state)
    reference = states[0]["normalization"]
    if any(state["normalization"] != reference for state in states[1:]):
        raise ValueError("expert checkpoints use different normalization statistics")
    if len({(model.input_size, model.forecast_horizon) for model in experts}) != 1:
        raise ValueError("expert input sizes and forecast horizons must agree")
    return experts, Normalization.from_dict(reference)


def load_router_artifact(router_type, path, device, expected_normalization=None):
    if router_type == "rf":
        router = RandomForestRouter.load(path)
        normalization = router.artifact_metadata.get("normalization")
    else:
        router, state = load_model_from_checkpoint(path, device)
        if state.get("model_kind") != "router" or state["model_name"] != router_type:
            raise ValueError("router checkpoint type does not match --router")
        normalization = state.get("normalization")
    if normalization is None:
        raise ValueError("router artifact is missing normalization metadata; retrain/resave it")
    normalization = Normalization.from_dict(normalization).to_dict()
    if expected_normalization is not None and normalization != expected_normalization.to_dict():
        raise ValueError("router and experts use different normalization statistics")
    return router, normalization


def predict_router_classes(router, normalized_x):
    if isinstance(router, nn.Module):
        return router(normalized_x).argmax(dim=1)
    features = flatten_router_features(normalized_x.detach().cpu().numpy())
    return torch.as_tensor(
        router.predict(features), device=normalized_x.device, dtype=torch.long
    )
