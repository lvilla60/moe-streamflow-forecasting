import h5py
import numpy as np
import pytest

from src.dataset import CaudalDataset


def create_train_file(path):
    with h5py.File(path, "w") as file:
        file.create_dataset("X", data=np.arange(6 * 336 * 12, dtype=np.float32).reshape(6, 336, 12))
        file.create_dataset("y", data=np.arange(6 * 48, dtype=np.float32).reshape(6, 48))
        file.create_dataset(
            "y_aux",
            data=np.arange(6 * 48 * 11, dtype=np.float32).reshape(6, 48, 11),
        )
        file.create_dataset("basin_id", data=np.arange(100, 106, dtype=np.int32))
        file.create_dataset("split", data=np.array([0, 0, 0, 0, 1, 1], dtype=np.int8))


def create_test_file(path):
    with h5py.File(path, "w") as file:
        file.create_dataset("X", data=np.arange(3 * 336 * 12, dtype=np.float32).reshape(3, 336, 12))
        file.create_dataset("basin_id", data=np.arange(200, 203, dtype=np.int32))


def test_train_validation_and_all_lengths(tmp_path):
    path = tmp_path / "train.h5"
    create_train_file(path)

    assert len(CaudalDataset(path, split="train")) == 4
    assert len(CaudalDataset(path, split="validation")) == 2
    assert len(CaudalDataset(path, split="all")) == 6


def test_training_sample_fields_and_shapes(tmp_path):
    path = tmp_path / "train.h5"
    create_train_file(path)
    dataset = CaudalDataset(path, split="train")

    sample = dataset[0]

    assert set(sample) == {"Id", "X", "basin_id", "y", "y_aux"}
    assert sample["X"].shape == (336, 12)
    assert sample["y"].shape == (48,)
    assert sample["y_aux"].shape == (48, 11)


def test_test_sample_has_only_test_fields(tmp_path):
    path = tmp_path / "test.h5"
    create_test_file(path)

    dataset = CaudalDataset(path, split="test")
    sample = dataset[0]

    assert set(sample) == {"Id", "X", "basin_id"}
    assert "y" not in sample
    assert "y_aux" not in sample


def test_close_is_safe_and_reopens_file(tmp_path):
    path = tmp_path / "train.h5"
    create_train_file(path)
    dataset = CaudalDataset(path, split="train")

    dataset.close()
    dataset.close()
    sample = dataset[0]

    assert sample["Id"] == 0
    assert sample["X"].shape == (336, 12)
    dataset.close()


@pytest.mark.parametrize("split", ["invalid", "test"])
def test_invalid_train_split_raises_value_error(tmp_path, split):
    path = tmp_path / "train.h5"
    create_train_file(path)

    with pytest.raises(ValueError):
        CaudalDataset(path, split=split)


def test_invalid_test_split_raises_value_error(tmp_path):
    path = tmp_path / "test.h5"
    create_test_file(path)

    with pytest.raises(ValueError):
        CaudalDataset(path, split="invalid")
