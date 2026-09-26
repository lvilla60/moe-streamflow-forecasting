"""Random-forest and neural routing classifiers."""

import joblib
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .common import SinusoidalPositionalEncoding, validate_sequence


EXPERT_CLASSES = {
    0: "lstm",
    1: "gru",
    2: "seq2seq_attention",
    3: "informer",
}


def flatten_router_features(x):
    """Flatten normalized [B, T, C] histories for the random forest."""
    if x.ndim != 3:
        raise ValueError("router features must have shape [batch, time, channels]")
    return x.reshape(x.shape[0], -1)


class RandomForestRouter:
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, **kwargs
        )
        self.artifact_metadata = {}

    def fit(self, features, labels):
        labels = np.asarray(labels)
        if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= 4):
            raise ValueError("RF labels must be integer expert classes 0..3")
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        # sklearn omits unobserved classes; expose stable columns 0, 1, 2, 3.
        probabilities = np.zeros((len(features), 4), dtype=np.float64)
        probabilities[:, self.model.classes_.astype(int)] = self.model.predict_proba(features)
        return probabilities

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"router": self, "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path):
        artifact = joblib.load(path)
        if isinstance(artifact, cls):
            return artifact
        router = artifact["router"]
        router.artifact_metadata = artifact.get("metadata", {})
        return router


class LSTMRouter(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, num_layers=1, dropout=0.0,
                 num_classes=4):
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=recurrent_dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        validate_sequence(x, self.input_size)
        _, (hidden, _) = self.encoder(x)
        return self.classifier(hidden[-1])


class TransformerRouter(nn.Module):
    def __init__(self, input_size=12, d_model=32, n_heads=4, num_layers=2,
                 d_ff=64, dropout=0.1, num_classes=4, max_length=512):
        super().__init__()
        self.input_size = input_size
        self.projection = nn.Linear(input_size, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        validate_sequence(x, self.input_size)
        encoded = self.encoder(self.position(self.projection(x)))
        return self.classifier(encoded.mean(dim=1))


def create_router(name, **kwargs):
    routers = {"lstm": LSTMRouter, "transformer": TransformerRouter}
    name = name.lower()
    if name not in routers:
        raise ValueError(f"unknown neural router {name!r}; expected {sorted(routers)}")
    return routers[name](**kwargs)
