import pytest
import torch

from src.models import InformerExpert, ProbSparseAttention


def tiny_informer():
    return InformerExpert(
        input_size=12, forecast_horizon=8, label_len=8, d_model=16,
        n_heads=2, encoder_layers=2, decoder_layers=1, d_ff=32,
        dropout=0.0, factor=2, max_length=64,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_informer_shape_finite_and_backward(batch_size):
    model = tiny_informer()
    x = torch.randn(batch_size, 32, 12, requires_grad=True)
    prediction = model(x)
    assert prediction.shape == (batch_size, 8)
    assert torch.isfinite(prediction).all()
    prediction.mean().backward()
    assert x.grad is not None


def test_probsparse_attention_shape_and_gradient():
    attention = ProbSparseAttention(d_model=16, n_heads=2, factor=2)
    x = torch.randn(2, 12, 16, requires_grad=True)
    output = attention(x, x, x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None


def test_informer_causal_mask_shape_and_values():
    mask = InformerExpert.causal_mask(6)
    assert mask.shape == (6, 6)
    assert not mask.diagonal().any()
    assert mask[0, 5]
    assert not mask[5, 0]


def test_decoder_prefix_is_independent_of_future_positions():
    from src.models.informer import InformerDecoderLayer

    torch.manual_seed(13)
    decoder = InformerDecoderLayer(8, 2, 16, dropout=0.0, factor=1).eval()
    x = torch.randn(1, 32, 8)
    x[:, 0] *= 100
    x.requires_grad_()
    memory = torch.randn(1, 12, 8)
    mask = InformerExpert.causal_mask(32)
    reference = decoder(x, memory, mask)
    changed = x.detach().clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 1000
    perturbed = decoder(changed, memory, mask)
    torch.testing.assert_close(reference[:, :8], perturbed[:, :8])
    reference[:, :8, 0].sum().backward()
    assert torch.count_nonzero(x.grad[:, 8:]) == 0
