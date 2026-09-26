import numpy as np
import pytest
import torch

from src.models import (
    LSTMRouter,
    RandomForestRouter,
    create_expert,
    flatten_router_features,
    select_expert_predictions,
)
from src.routing import best_expert_labels, predict_expert_stack


def test_complete_synthetic_expert_router_moe_flow():
    pytest.importorskip("sklearn")
    torch.manual_seed(4)
    batch, sequence, channels, horizon = 8, 12, 12, 4
    x = torch.randn(batch, sequence, channels)
    target = torch.randn(batch, horizon)
    decoder_start = x[:, -1, 11]
    experts = [
        create_expert("lstm", input_size=channels, hidden_size=6, forecast_horizon=horizon),
        create_expert("gru", input_size=channels, hidden_size=6, forecast_horizon=horizon),
        create_expert("seq2seq_attention", input_size=channels, hidden_size=6,
                      forecast_horizon=horizon),
        create_expert("informer", input_size=channels, forecast_horizon=horizon,
                      label_len=4, d_model=8, n_heads=2, encoder_layers=1,
                      decoder_layers=1, d_ff=16, dropout=0.0, factor=2,
                      max_length=32),
    ]

    predictions = predict_expert_stack(experts, x, decoder_start)
    labels, errors = best_expert_labels(target, predictions)
    assert predictions.shape == (batch, 4, horizon)
    assert errors.shape == (batch, 4)

    neural_router = LSTMRouter(input_size=channels, hidden_size=6)
    neural_classes = neural_router(x).argmax(dim=1)
    neural_forecast = select_expert_predictions(predictions, neural_classes)
    assert neural_forecast.shape == (batch, horizon)

    features = flatten_router_features(x.numpy())
    rf_router = RandomForestRouter(n_estimators=8, random_state=2, n_jobs=1)
    rf_router.fit(features, labels.numpy())
    rf_classes = torch.as_tensor(rf_router.predict(features))
    rf_forecast = select_expert_predictions(predictions, rf_classes)
    assert rf_forecast.shape == (batch, horizon)
    assert np.isfinite(rf_forecast.detach().numpy()).all()
