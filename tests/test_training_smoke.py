import torch
from torch.utils.data import DataLoader, Dataset

from src.checkpointing import load_model_from_checkpoint, save_checkpoint
from src.models import LSTMExpert
from src.training import Normalization, fit_expert, prepare_batch


class TinyForecastDataset(Dataset):
    def __init__(self, size=6, history=12, horizon=4):
        generator = torch.Generator().manual_seed(7)
        self.x = torch.randn(size, history, 12, generator=generator)
        self.y = self.x[:, -1, 11:12] + 0.1 * torch.randn(
            size, horizon, generator=generator
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"X": self.x[index], "y": self.y[index], "Id": index, "basin_id": 0}


def normalization():
    return Normalization([0.0] * 12, [1.0] * 12, 0.0, 1.0)


def test_decoder_start_uses_raw_discharge_on_y_scale():
    batch = {"X": torch.zeros(2, 3, 12), "y": torch.zeros(2, 4)}
    batch["X"][:, -1, 11] = torch.tensor([5.0, 7.0])
    stats = Normalization([0.0] * 12, [1.0] * 12, 1.0, 2.0)
    prepared = prepare_batch(batch, stats, "cpu")
    torch.testing.assert_close(prepared["decoder_start"], torch.tensor([2.0, 3.0]))


def test_one_epoch_training_and_checkpoint_round_trip(tmp_path):
    data = TinyForecastDataset()
    loader = DataLoader(data, batch_size=2)
    config = {"input_size": 12, "hidden_size": 6, "forecast_horizon": 4, "dropout": 0.0}
    model = LSTMExpert(**config)
    checkpoint = tmp_path / "lstm.pt"
    history = fit_expert(
        model, loader, loader, normalization(), model_name="lstm",
        model_config=config, epochs=1, max_train_batches=2,
        max_validation_batches=2, checkpoint_path=checkpoint,
    )
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
    restored, state = load_model_from_checkpoint(checkpoint)
    model.eval()
    x = data.x[:2]
    torch.testing.assert_close(model(x), restored(x))
    assert state["epoch"] == 1


def test_explicit_checkpoint_round_trip(tmp_path):
    config = {"input_size": 12, "hidden_size": 5, "forecast_horizon": 4, "dropout": 0.0}
    model = LSTMExpert(**config).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, model_name="lstm", model_config=config, model=model,
        normalization=normalization().to_dict(),
    )
    restored, _ = load_model_from_checkpoint(path)
    x = torch.randn(2, 10, 12)
    torch.testing.assert_close(model(x), restored(x))
