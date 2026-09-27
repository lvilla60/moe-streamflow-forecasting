"""Synthetic regressions for correctness risks identified in the technical audit."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import evaluate_moe, predict_test, train_router
from src import training
from src.checkpointing import load_checkpoint, load_model_from_checkpoint, save_checkpoint
from src.inference import load_router_artifact
from src.models import LSTMExpert, LSTMRouter, RandomForestRouter, select_expert_predictions
from src.routing import EXPERT_NAMES, read_router_labels, validate_label_rows
from src.training import Normalization, RegressionAccumulator


def stats():
    return Normalization([0.0] * 12, [1.0] * 12, 2.0, 3.0)


def test_router_selects_best_validation_epoch(tmp_path, monkeypatch):
    config = {"hidden_size": 4, "dropout": 0.0}
    model = LSTMRouter(**config)
    loader = DataLoader(TensorDataset(torch.randn(2, 4, 12), torch.tensor([0, 1])), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = iter([3.0, 1.0, 2.0])
    states = []

    def validation(model, loader, device):
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return next(losses), 0.5, np.eye(4, dtype=int)

    monkeypatch.setattr(train_router, "evaluate_neural", validation)
    path = tmp_path / "router.pt"
    history = train_router.fit_neural_router(
        model, loader, loader, optimizer, epochs=3, device="cpu", output=path,
        model_name="lstm", config=config, normalization=stats(),
    )
    restored, checkpoint = load_model_from_checkpoint(path)
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_validation_loss"] == 1.0
    assert checkpoint["training_history"]["validation_accuracy"] == [0.5, 0.5]
    assert history["validation_loss"] == [3.0, 1.0, 2.0]
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, states[1][key])


def test_router_without_validation_saves_explicit_final_model(tmp_path):
    model = LSTMRouter(hidden_size=4)
    loader = DataLoader(TensorDataset(torch.randn(2, 4, 12), torch.tensor([0, 1])), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "router.pt"
    train_router.fit_neural_router(
        model, loader, None, optimizer, epochs=1, device="cpu", output=path,
        model_name="lstm", config={"hidden_size": 4}, normalization=stats(),
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["best_validation_loss"] is None
    assert checkpoint["selection"] == "final_epoch"


def test_router_validation_weighted_loss_no_grad():
    model = LSTMRouter(hidden_size=4).eval()
    x, y = torch.randn(5, 4, 12), torch.tensor([0, 1, 2, 3, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    loss, accuracy, confusion = train_router.evaluate_neural(model, loader, "cpu")
    with torch.no_grad():
        logits = model(x)
        expected_loss = torch.nn.functional.cross_entropy(logits, y)
    assert loss == pytest.approx(float(expected_loss), rel=1e-6)
    assert accuracy == pytest.approx(float((logits.argmax(1) == y).float().mean()))
    assert confusion.sum() == 5
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("metadata", [None, {"x_mean": [0.0] * 12, "x_std": [1.0] * 12,
                                          "y_mean": 99.0, "y_std": 3.0}])
def test_router_normalization_cannot_be_missing_or_mismatched(tmp_path, metadata):
    path = tmp_path / "router.pt"
    save_checkpoint(path, model_name="lstm", model_config={"hidden_size": 4},
                    model=LSTMRouter(hidden_size=4), model_kind="router", normalization=metadata)
    with pytest.raises(ValueError, match="normalization"):
        load_router_artifact("lstm", path, "cpu", stats())


def test_resume_rejects_changed_scale_or_same_shape_config_and_preserves_best(tmp_path, monkeypatch):
    config = {"hidden_size": 4, "forecast_horizon": 4, "dropout": 0.0}
    model = LSTMExpert(**config)
    optimizer = torch.optim.AdamW(model.parameters())
    # Populate AdamW state, rather than testing an empty optimizer round trip.
    model(torch.randn(2, 4, 12)).square().mean().backward()
    optimizer.step()
    original_optimizer = optimizer.state_dict()
    source = tmp_path / "source.pt"
    save_checkpoint(source, model_name="lstm", model_config=config, model=model,
                    optimizer=optimizer, epoch=1, best_validation_loss=0.25,
                    normalization=stats().to_dict(),
                    training_history={"train_loss": [0.5], "validation_loss": [0.25]})
    destination = tmp_path / "resumed" / "best.pt"
    options = dict(model_name="lstm", model_config=config, epochs=2,
                   checkpoint_path=destination, resume_checkpoint=source)
    incompatible = Normalization([0.0] * 12, [1.0] * 12, 9.0, 3.0)
    with pytest.raises(ValueError, match="normalization"):
        training.fit_expert(model, [], [], incompatible, **options)
    with pytest.raises(ValueError, match="config"):
        training.fit_expert(model, [], [], stats(), **{**options, "model_config": {**config, "dropout": 0.5}})

    def train(model, loader, restored_optimizer, *args):
        restored_state = restored_optimizer.state_dict()["state"]
        for index, state in original_optimizer["state"].items():
            torch.testing.assert_close(restored_state[index]["exp_avg"], state["exp_avg"])
        return 0.2

    monkeypatch.setattr(training, "train_one_epoch", train)
    monkeypatch.setattr(training, "validate_expert", lambda *args: (0.8, {"mae": 1, "rmse": 1, "nse": 0}))
    history = training.fit_expert(model, [], [], stats(), **options)
    assert history["validation_loss"] == [0.25, 0.8]
    assert destination.read_bytes() == source.read_bytes()
    assert load_checkpoint(destination)["epoch"] == 1


def test_normalization_rejects_nontraining_stats_and_invalid_scales(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"split": "validation", "x": {"mean": [0], "std": [1]},
                                "y": {"mean": 0, "std": 1}}))
    with pytest.raises(ValueError, match="train"):
        Normalization.from_json(path)
    with pytest.raises(ValueError, match="finite"):
        Normalization([0], [float("nan")], 0, 1)
    zero_channel = Normalization([2], [0], 1, 2)
    torch.testing.assert_close(zero_channel.normalize_x(torch.tensor([[2.0]])), torch.zeros(1, 1))


def test_global_metrics_match_direct_calculation_across_unequal_batches():
    target = np.arange(28, dtype=float).reshape(7, 4) + 10000
    prediction = target + np.arange(7)[:, None] / 3
    metrics = RegressionAccumulator()
    metrics.update(target[:2], prediction[:2])
    metrics.update(target[2:], prediction[2:])
    result = metrics.finalize()
    error = target - prediction
    assert result["mae"] == pytest.approx(np.abs(error).mean())
    assert result["rmse"] == pytest.approx(np.sqrt(np.square(error).mean()))
    assert result["nse"] == pytest.approx(1 - np.square(error).sum() / np.square(target - target.mean()).sum())
    with pytest.raises(ValueError, match="shapes"):
        metrics.update(target, prediction.reshape(4, 7))


def write_synthetic_rows(path):
    with h5py.File(path, "w") as file:
        file["X"] = np.arange(6 * 4 * 12, dtype=np.float32).reshape(6, 4, 12)
        file["split"] = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)


def test_label_row_order_split_isolation_and_rf_float32(tmp_path):
    source = tmp_path / "rows.h5"
    write_synthetic_rows(source)
    labels_path = tmp_path / "labels.npz"
    np.savez(labels_path, Id=[4, 0, 2], best_expert=[3, 1, 2],
             expert_names=EXPERT_NAMES, split="train")
    ids, labels = read_router_labels(labels_path, "train")
    np.testing.assert_array_equal(ids, [0, 2, 4])
    np.testing.assert_array_equal(labels, [1, 2, 3])
    validate_label_rows(source, ids, "train")
    with pytest.raises(ValueError, match="official validation"):
        validate_label_rows(source, ids, "validation")
    features = train_router.load_flat_features(source, ids[::-1], stats(), chunk_size=2)
    with h5py.File(source, "r") as file:
        expected = np.stack([file["X"][int(row)] for row in ids[::-1]]).reshape(3, -1)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features, expected)


def test_rf_probabilities_keep_fixed_class_columns_and_create_parent(tmp_path):
    router = RandomForestRouter(n_estimators=2, n_jobs=1).fit(
        np.array([[0], [1], [2], [3]], dtype=np.float32), np.array([1, 1, 3, 3]))
    probabilities = router.predict_proba([[0], [3]])
    assert probabilities.shape == (2, 4)
    np.testing.assert_array_equal(probabilities[:, [0, 2]], 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    router.save(tmp_path / "nested" / "router.joblib")


def test_gather_rejects_fractional_classes():
    with pytest.raises(ValueError, match="integer"):
        select_expert_predictions(torch.zeros(2, 4, 3), torch.tensor([0.5, 1.0]))


class SyntheticInferenceData:
    """No HDF5 reads: exercise CLI iteration, CSV/JSON output, and row mapping."""

    def __init__(self, path, split):
        self.split = split
        self.closed = False
        self.reads = 0

    def __len__(self):
        return 5

    def __getitem__(self, index):
        self.reads += 1
        item = {"Id": index, "basin_id": 0, "X": torch.zeros(4, 12)}
        if self.split == "validation":
            item["y"] = torch.arange(48, dtype=torch.float32) + index
        return item

    def close(self):
        self.closed = True


def patch_inference_cli(monkeypatch, module, output, extra_args):
    from types import SimpleNamespace

    instances = []

    def dataset(path, split):
        instance = SyntheticInferenceData(path, split)
        instances.append(instance)
        return instance

    unused = output.parent / "unused"
    unused.write_bytes(b"placeholder")

    def resolve(path):
        path = Path(path)
        if path.is_absolute():
            return path
        if path.as_posix() == "outputs/experiments":
            return output.parent / "experiments"
        return unused

    monkeypatch.setattr(
        module, "resolve", resolve,
    )

    monkeypatch.setattr(module, "CaudalDataset", dataset)
    monkeypatch.setattr(
        module, "load_experts",
        lambda *args, **kwargs: ([SimpleNamespace(forecast_horizon=48)] * 4, stats()),
    )
    monkeypatch.setattr(
        module, "load_router_artifact",
        lambda *args, **kwargs: (object(), stats().to_dict(), EXPERT_NAMES),
    )
    monkeypatch.setattr(module, "predict_router_classes", lambda router, x: torch.zeros(len(x), dtype=torch.long))
    monkeypatch.setattr(module, "predict_expert_stack", lambda experts, x, start: torch.zeros(len(x), 4, 48))
    monkeypatch.setattr(sys, "argv", ["script", "--lstm-checkpoint", "unused", "--gru-checkpoint", "unused",
                                    "--seq2seq-checkpoint", "unused", "--informer-checkpoint", "unused",
                                    "--router", "lstm", "--router-checkpoint", "unused", "--batch-size", "2",
                                    "--output", str(output), *extra_args])
    return instances


@pytest.mark.parametrize("limit, expected", [(1, 2), (None, 5)])
def test_prediction_cli_limit_and_full_csv(tmp_path, monkeypatch, limit, expected):
    import csv

    output = tmp_path / "predictions.csv"
    extra = [] if limit is None else ["--max-batches", str(limit)]
    instances = patch_inference_cli(monkeypatch, predict_test, output, extra)
    predict_test.main()
    with output.open(newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["Id", *[f"q_{hour:02d}" for hour in range(1, 49)]]
    assert len(rows) == expected + 1
    assert [int(row[0]) for row in rows[1:]] == list(range(expected))
    assert all(float(value) == 2.0 for row in rows[1:] for value in row[1:])
    assert instances[0].reads == expected
    assert instances[0].closed


@pytest.mark.parametrize("label_id, matched", [(1, 1), (20, 0)])
def test_moe_partial_label_coverage(tmp_path, monkeypatch, label_id, matched):
    output = tmp_path / "metrics.json"
    labels = tmp_path / "subset.npz"
    np.savez(labels, Id=[label_id], best_expert=[0], expert_names=EXPERT_NAMES, split="validation")
    patch_inference_cli(monkeypatch, evaluate_moe, output, ["--validation-labels", str(labels)])
    # Row membership is tested separately using synthetic HDF5 above.
    monkeypatch.setattr(evaluate_moe, "validate_label_rows", lambda *args: None)
    evaluate_moe.main()
    result = json.loads(output.read_text())
    assert result["n_samples"] == 5
    assert result["router_labeled_samples"] == matched
    assert result["router_unlabeled_samples"] == 5 - matched
    assert result["router_accuracy"] == (1.0 if matched else None)
    assert np.asarray(result["router_confusion_matrix"]).sum() == matched
