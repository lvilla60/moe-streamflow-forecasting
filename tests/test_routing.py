import numpy as np
import torch

from src.routing import best_expert_labels


def test_best_expert_labels_and_first_minimum_tie():
    target = torch.zeros(3, 4)
    predictions = torch.tensor([
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        [[3, 3, 3, 3], [2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
    ], dtype=torch.float32)
    labels, errors = best_expert_labels(target, predictions)
    torch.testing.assert_close(labels, torch.tensor([0, 2, 0]))
    assert errors.shape == (3, 4)


def test_best_expert_labels_numpy_path():
    target = np.zeros((2, 3))
    predictions = np.ones((2, 4, 3))
    predictions[1, 3] = 0
    labels, _ = best_expert_labels(target, predictions)
    np.testing.assert_array_equal(labels, [0, 3])
