import pytest
import torch

from src.models import GRUExpert, LSTMExpert, Seq2SeqAttentionExpert


@pytest.mark.parametrize("model_class", [LSTMExpert, GRUExpert])
def test_direct_expert_shape_and_backward(model_class):
    model = model_class(input_size=12, hidden_size=8, forecast_horizon=8)
    x = torch.randn(2, 32, 12, requires_grad=True)
    prediction = model(x)
    assert prediction.shape == (2, 8)
    assert torch.isfinite(prediction).all()
    prediction.mean().backward()
    assert x.grad is not None


def test_direct_expert_rejects_wrong_rank():
    model = LSTMExpert(input_size=12, hidden_size=8, forecast_horizon=8)
    with pytest.raises(ValueError, match="rank"):
        model(torch.randn(32, 12))


def test_seq2seq_attention_inference_teacher_forcing_and_backward():
    model = Seq2SeqAttentionExpert(
        input_size=12, hidden_size=8, forecast_horizon=6
    )
    x = torch.randn(2, 24, 12, requires_grad=True)
    start = torch.randn(2)
    target = torch.randn(2, 6)

    inference, attention = model(x, decoder_start=start, return_attention=True)
    teacher_forced = model(
        x, target=target, teacher_forcing_ratio=1.0, decoder_start=start
    )

    assert inference.shape == teacher_forced.shape == (2, 6)
    assert attention.shape == (2, 6, 24)
    assert torch.isfinite(inference).all()
    teacher_forced.mean().backward()
    assert x.grad is not None


def test_seq2seq_requires_target_scale_decoder_start():
    model = Seq2SeqAttentionExpert(input_size=12, hidden_size=8, forecast_horizon=6)
    with pytest.raises(ValueError, match="decoder_start"):
        model(torch.randn(2, 24, 12))


def test_seq2seq_seeded_teacher_forcing_and_eval_target_independence():
    from src.training import set_seed

    model = Seq2SeqAttentionExpert(hidden_size=4, forecast_horizon=4)
    x, target, start = torch.randn(2, 8, 12), torch.randn(2, 4), torch.randn(2)
    model.train()
    set_seed(21)
    first = model(x, target=target, teacher_forcing_ratio=0.5, decoder_start=start)
    set_seed(21)
    second = model(x, target=target, teacher_forcing_ratio=0.5, decoder_start=start)
    torch.testing.assert_close(first, second)
    model.eval()
    actual = model(x, target=target, teacher_forcing_ratio=1.0, decoder_start=start)
    expected = model(x, decoder_start=start)
    torch.testing.assert_close(actual, expected)
