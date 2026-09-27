import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_router import evaluate_neural
from src.checkpointing import save_checkpoint
from src.inference import load_router_artifact
from src.models import (LSTMRouter, RandomForestRouter, TransformerRouter,
                        select_expert_predictions)
from src.routing import (EXPERT_NAMES, best_expert_labels, parse_expert_subset,
                         read_router_labels, require_matching_expert_names)
from src.training import Normalization


def stats():
    return Normalization([0.0] * 12, [1.0] * 12, 0.0, 1.0)


def write_labels(path, names, labels, errors=None, split="train"):
    values = {
        "Id": np.arange(len(labels), dtype=np.int64),
        "best_expert": np.asarray(labels, dtype=np.int64),
        "expert_names": np.asarray(names),
        "split": np.asarray(split),
    }
    if errors is not None:
        values["expert_errors"] = np.asarray(errors, dtype=np.float32)
    np.savez(path, **values)


def test_default_and_binary_expert_subsets_preserve_local_order():
    assert parse_expert_subset() == ("lstm", "gru", "seq2seq", "informer")
    assert parse_expert_subset("lstm,informer") == ("lstm", "informer")
    assert parse_expert_subset("lstm,gru") == ("lstm", "gru")


@pytest.mark.parametrize("value", ["lstm,lstm", "lstm,unknown", "lstm"])
def test_invalid_expert_subsets_are_rejected(value):
    with pytest.raises(ValueError):
        parse_expert_subset(value)


def test_binary_labels_only_compare_selected_experts():
    target = torch.zeros(2, 3)
    # A hypothetical GRU would be exact, but is absent from the local stack.
    lstm_and_informer = torch.tensor([
        [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
    ])
    labels, errors = best_expert_labels(target, lstm_and_informer)
    torch.testing.assert_close(labels, torch.tensor([1, 0]))
    assert errors.shape == (2, 2)

    lstm_and_gru = torch.tensor([[[2.0] * 3, [0.0] * 3]])
    labels, _ = best_expert_labels(target[:1], lstm_and_gru)
    torch.testing.assert_close(labels, torch.tensor([1]))


def test_label_metadata_round_trip_and_legacy_four_class(tmp_path):
    binary = tmp_path / "binary.npz"
    write_labels(binary, ("lstm", "informer"), [0, 1], [[1, 2], [2, 1]])
    ids, labels, names = read_router_labels(binary, "train", return_expert_names=True)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_array_equal(labels, [0, 1])
    assert names == ("lstm", "informer")

    legacy = tmp_path / "legacy.npz"
    write_labels(legacy, ("lstm", "gru", "seq2seq_attention", "informer"),
                 [0, 1, 2, 3])
    _, _, names = read_router_labels(legacy, "train", return_expert_names=True)
    assert names == EXPERT_NAMES


def test_label_metadata_validation_errors(tmp_path):
    invalid_index = tmp_path / "invalid_index.npz"
    write_labels(invalid_index, ("lstm", "informer"), [0, 2])
    with pytest.raises(ValueError, match="outside class range"):
        read_router_labels(invalid_index, "train")

    invalid_errors = tmp_path / "invalid_errors.npz"
    write_labels(invalid_errors, ("lstm", "informer"), [0, 1], [[1], [2]])
    with pytest.raises(ValueError, match="expert_errors"):
        read_router_labels(invalid_errors, "train")

    with pytest.raises(ValueError, match="match exactly"):
        require_matching_expert_names(("lstm", "informer"), ("lstm", "gru"),
                                      "training and validation mappings")


@pytest.mark.parametrize("router", [
    LSTMRouter(input_size=12, hidden_size=4, num_classes=2),
    TransformerRouter(input_size=12, d_model=8, n_heads=2, num_layers=1,
                      d_ff=16, dropout=0.0, num_classes=2),
])
def test_neural_routers_support_two_local_classes(router):
    logits = router(torch.randn(3, 6, 12))
    assert logits.shape == (3, 2)


@pytest.mark.parametrize("num_classes", [2, 4])
def test_neural_evaluation_confusion_shape(num_classes):
    model = LSTMRouter(hidden_size=4, num_classes=num_classes)
    labels = torch.arange(6) % num_classes
    loader = DataLoader(TensorDataset(torch.randn(6, 5, 12), labels), batch_size=2)
    _, _, confusion = evaluate_neural(model, loader, "cpu")
    assert confusion.shape == (num_classes, num_classes)


def test_random_forest_two_and_four_class_probability_shapes():
    pytest.importorskip("sklearn")
    features = np.arange(16, dtype=np.float32).reshape(8, 2)
    binary = RandomForestRouter(n_estimators=4, n_jobs=1, num_classes=2).fit(
        features, np.arange(8) % 2
    )
    assert binary.predict_proba(features).shape == (8, 2)

    four = RandomForestRouter(n_estimators=4, n_jobs=1).fit(
        features, np.arange(8) % 4
    )
    assert four.predict_proba(features).shape == (8, 4)


def test_router_checkpoint_mapping_round_trip_and_legacy_fallback(tmp_path):
    binary_path = tmp_path / "binary.pt"
    binary_config = {"input_size": 12, "hidden_size": 4, "num_classes": 2}
    save_checkpoint(
        binary_path, model_name="lstm", model_config=binary_config,
        model=LSTMRouter(**binary_config), model_kind="router",
        normalization=stats().to_dict(), expert_names=("lstm", "informer"),
    )
    _, _, names = load_router_artifact(
        "lstm", binary_path, "cpu", return_expert_names=True
    )
    assert names == ("lstm", "informer")

    legacy_path = tmp_path / "legacy.pt"
    legacy_config = {"input_size": 12, "hidden_size": 4, "num_classes": 4}
    save_checkpoint(
        legacy_path, model_name="lstm", model_config=legacy_config,
        model=LSTMRouter(**legacy_config), model_kind="router",
        normalization=stats().to_dict(),
    )
    _, _, names = load_router_artifact(
        "lstm", legacy_path, "cpu", return_expert_names=True
    )
    assert names == EXPERT_NAMES


def test_random_forest_artifact_mapping_round_trip(tmp_path):
    pytest.importorskip("sklearn")
    path = tmp_path / "binary.joblib"
    router = RandomForestRouter(n_estimators=2, n_jobs=1, num_classes=2).fit(
        np.arange(8, dtype=np.float32).reshape(4, 2), np.array([0, 1, 0, 1])
    )
    router.save(path, metadata={
        "normalization": stats().to_dict(),
        "expert_names": ["lstm", "informer"],
    })
    restored, _, names = load_router_artifact(
        "rf", path, "cpu", return_expert_names=True
    )
    assert names == ("lstm", "informer")
    assert restored.predict_proba([[0.0, 1.0]]).shape == (1, 2)


def test_binary_router_without_mapping_is_not_guessed(tmp_path):
    path = tmp_path / "missing_mapping.pt"
    config = {"input_size": 12, "hidden_size": 4, "num_classes": 2}
    save_checkpoint(
        path, model_name="lstm", model_config=config, model=LSTMRouter(**config),
        model_kind="router", normalization=stats().to_dict(),
    )
    with pytest.raises(ValueError, match="missing expert_names"):
        load_router_artifact("lstm", path, "cpu", return_expert_names=True)


def test_binary_local_class_one_selects_informer_not_global_gru():
    # The stack order is [lstm, informer], so local class 1 selects Informer.
    stack = torch.tensor([[[10.0, 11.0], [40.0, 41.0]]])
    selected = select_expert_predictions(stack, torch.tensor([1]))
    torch.testing.assert_close(selected, torch.tensor([[40.0, 41.0]]))
