import numpy as np
import pytest
import torch

from src.models import (
    LSTMRouter,
    RandomForestRouter,
    TransformerRouter,
    flatten_router_features,
)


@pytest.mark.parametrize("router", [
    LSTMRouter(input_size=12, hidden_size=8),
    TransformerRouter(input_size=12, d_model=16, n_heads=2, num_layers=1, d_ff=32),
])
def test_neural_router_shape_and_backward(router):
    x = torch.randn(3, 24, 12, requires_grad=True)
    logits = router(x)
    assert logits.shape == (3, 4)
    logits.mean().backward()
    assert x.grad is not None


def test_random_forest_fit_predict_and_persistence(tmp_path):
    pytest.importorskip("sklearn")
    x = np.arange(12 * 6 * 4, dtype=np.float32).reshape(12, 6, 4)
    features = flatten_router_features(x)
    labels = np.arange(12) % 4
    router = RandomForestRouter(n_estimators=8, random_state=3, n_jobs=1)
    router.fit(features, labels)
    predictions = router.predict(features)
    assert predictions.shape == (12,)
    assert router.predict_proba(features).shape == (12, 4)

    path = tmp_path / "router.joblib"
    router.save(path, metadata={"purpose": "test"})
    restored = RandomForestRouter.load(path)
    np.testing.assert_array_equal(restored.predict(features), predictions)
    assert restored.artifact_metadata["purpose"] == "test"
