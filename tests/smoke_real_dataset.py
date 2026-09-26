"""Minimal smoke test for the real HDF5 dataset files."""

from pathlib import Path

from src.dataset import CaudalDataset


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "Input"


def main() -> None:
    train_path = INPUT / "train-001.h5"

    train = CaudalDataset(train_path, split="train")
    try:
        sample = train[0]
        print(f"train samples: {len(train)}")
        print(f"train first Id: {sample['Id']}")
        print(f"train basin_id: {sample['basin_id']}")
        print(f"train X: shape={sample['X'].shape}, dtype={sample['X'].dtype}")
        print(f"train y: shape={sample['y'].shape}, dtype={sample['y'].dtype}")
        print(f"train y_aux: shape={sample['y_aux'].shape}, dtype={sample['y_aux'].dtype}")
    finally:
        train.close()

    validation = CaudalDataset(train_path, split="validation")
    try:
        sample = validation[0]
        print(f"validation samples: {len(validation)}")
        print(f"validation first Id: {sample['Id']}")
        print(f"validation basin_id: {sample['basin_id']}")
        print(f"validation X shape: {sample['X'].shape}")
        print(f"validation y shape: {sample['y'].shape}")
        print(f"validation y_aux shape: {sample['y_aux'].shape}")
    finally:
        validation.close()

    test = CaudalDataset(INPUT / "test.h5", split="test")
    try:
        sample = test[0]
        print(f"test samples: {len(test)}")
        print(f"test first Id: {sample['Id']}")
        print(f"test basin_id: {sample['basin_id']}")
        print(f"test X: shape={sample['X'].shape}, dtype={sample['X'].dtype}")
        print(f"test keys: {sorted(sample)}")
        print(f"test has y: {'y' in sample}")
        print(f"test has y_aux: {'y_aux' in sample}")
    finally:
        test.close()


if __name__ == "__main__":
    main()
