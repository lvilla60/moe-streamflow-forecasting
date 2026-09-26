"""Sample-level hard Mixture of Experts inference."""

import numpy as np
import torch
from torch import nn

from .routers import flatten_router_features


def select_expert_predictions(expert_predictions, router_class):
    """Select one complete forecast per sample from [B, experts, horizon]."""
    if expert_predictions.ndim != 3:
        raise ValueError("expert_predictions must have shape [batch, experts, horizon]")
    if router_class.ndim != 1 or router_class.size(0) != expert_predictions.size(0):
        raise ValueError("router_class must have shape [batch]")
    if router_class.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("router_class must contain integer class indices")
    router_class = router_class.to(device=expert_predictions.device, dtype=torch.long)
    if torch.any(router_class < 0) or torch.any(router_class >= expert_predictions.size(1)):
        raise ValueError("router class is outside the available expert range")
    indices = router_class.view(-1, 1, 1).expand(-1, 1, expert_predictions.size(2))
    return expert_predictions.gather(1, indices).squeeze(1)


class HardMoE(nn.Module):
    """Run all experts, classify each sample, and select one 48-step forecast."""

    def __init__(self, experts, router):
        super().__init__()
        if len(experts) != 4:
            raise ValueError("HardMoE requires exactly four experts")
        self.experts = nn.ModuleList(experts)
        self.router = router

    def forward(self, x, decoder_start=None):
        predictions = []
        for expert in self.experts:
            if getattr(expert, "requires_decoder_start", False):
                prediction = expert(x, decoder_start=decoder_start)
            else:
                prediction = expert(x)
            predictions.append(prediction)
        stacked = torch.stack(predictions, dim=1)
        if isinstance(self.router, nn.Module):
            router_class = self.router(x).argmax(dim=1)
        else:
            features = flatten_router_features(x.detach().cpu().numpy())
            classes = self.router.predict(features)
            router_class = torch.as_tensor(classes, device=x.device, dtype=torch.long)
        return select_expert_predictions(stacked, router_class), router_class
