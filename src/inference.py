"""Artifact loading and router inference shared by MoE command-line workflows."""

import numpy as np
import torch
from torch import nn

from .checkpointing import load_model_from_checkpoint
from .models import RandomForestRouter, flatten_router_features
from .routing import (EXPERT_MODEL_NAMES, EXPERT_NAMES,
                      normalize_stored_expert_names, parse_expert_subset)
from .training import Normalization


def load_experts(checkpoint_paths, device, expert_names=None):
    expert_names = parse_expert_subset(expert_names)
    if len(checkpoint_paths) != len(expert_names):
        raise ValueError("checkpoint paths must match the ordered expert subset")
    experts, states = [], []
    for public_name, path in zip(expert_names, checkpoint_paths):
        expected_name = EXPERT_MODEL_NAMES[public_name]
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


def load_router_artifact(router_type, path, device, expected_normalization=None,
                         return_expert_names=False):
    if router_type == "rf":
        router = RandomForestRouter.load(path)
        normalization = router.artifact_metadata.get("normalization")
        raw_expert_names = router.artifact_metadata.get("expert_names")
        num_classes = getattr(router, "num_classes", 4)
    else:
        router, state = load_model_from_checkpoint(path, device)
        if state.get("model_kind") != "router" or state["model_name"] != router_type:
            raise ValueError("router checkpoint type does not match --router")
        normalization = state.get("normalization")
        raw_expert_names = state.get("expert_names")
        num_classes = int(state.get("model_config", {}).get("num_classes", 4))
    if normalization is None:
        raise ValueError("router artifact is missing normalization metadata; retrain/resave it")
    normalization = Normalization.from_dict(normalization).to_dict()
    if expected_normalization is not None and normalization != expected_normalization.to_dict():
        raise ValueError("router and experts use different normalization statistics")
    if raw_expert_names is None:
        if num_classes != 4:
            raise ValueError("router artifact is missing expert_names metadata")
        expert_names = EXPERT_NAMES
    else:
        expert_names = normalize_stored_expert_names(raw_expert_names)
        if len(expert_names) != num_classes:
            raise ValueError("router expert_names do not match its number of classes")
    result = (router, normalization)
    return (*result, expert_names) if return_expert_names else result


def predict_router_classes(router, normalized_x):
    if isinstance(router, nn.Module):
        return router(normalized_x).argmax(dim=1)
    features = flatten_router_features(normalized_x.detach().cpu().numpy())
    return torch.as_tensor(
        router.predict(features), device=normalized_x.device, dtype=torch.long
    )
