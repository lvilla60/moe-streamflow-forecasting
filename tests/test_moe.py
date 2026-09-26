import pytest
import torch

from src.models import select_expert_predictions


def test_hard_selection_returns_exact_expert_vectors():
    predictions = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
    classes = torch.tensor([2, 0, 3])
    selected = select_expert_predictions(predictions, classes)
    expected = torch.stack((predictions[0, 2], predictions[1, 0], predictions[2, 3]))
    torch.testing.assert_close(selected, expected)


def test_hard_selection_rejects_invalid_class():
    with pytest.raises(ValueError, match="outside"):
        select_expert_predictions(torch.zeros(2, 4, 5), torch.tensor([0, 4]))
