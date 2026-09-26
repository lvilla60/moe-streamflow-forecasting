import csv
import json

import numpy as np
import pytest

from scripts.summarize_experiments import summarize
from src.reporting import (experiment_directory, plot_confusion_matrix, plot_horizon_metric,
                           plot_loss_curve, save_history_csv, save_json)
from src.training import HorizonRegressionAccumulator


def test_reporting_writes_machine_readable_files_and_headless_plots(tmp_path):
    history = {
        "epoch": [1, 2], "train_loss": [2.0, 1.0],
        "validation_loss": [2.5, 1.5],
    }
    save_history_csv(tmp_path / "history.csv", history)
    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[1] == {"epoch": "2", "train_loss": "1.0", "validation_loss": "1.5"}

    metrics = {"MAE": 1.25, "per_horizon_mae": [1.0, 1.5]}
    save_json(tmp_path / "metrics.json", metrics)
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics

    plot_loss_curve(tmp_path / "loss.png", history)
    plot_horizon_metric(tmp_path / "horizon.png", [1.0, 2.0], "MAE")
    plot_confusion_matrix(tmp_path / "confusion.png", np.eye(4, dtype=int))
    for name in ("loss.png", "horizon.png", "confusion.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_summarize_handles_sparse_metrics_and_existing_persistence(tmp_path):
    experiments = tmp_path / "outputs" / "experiments"
    save_json(experiments / "expert_a" / "metrics.json",
              {"kind": "expert", "model": "lstm", "best_epoch": 2,
               "MAE": 1.0, "RMSE": 2.0, "NSE": 0.5})
    save_json(experiments / "router_a" / "metrics.json",
              {"kind": "router", "model": "rf", "validation_accuracy": 0.75})
    save_json(experiments.parent / "persistence_validation.json",
              {"global": {"mae": 3.0, "rmse": 4.0, "nse": -0.1}})
    output = experiments / "summary.csv"
    rows = summarize(experiments, output)
    assert [row["experiment_name"] for row in rows] == [
        "expert_a", "router_a", "persistence_validation"
    ]
    assert rows[1]["MAE"] == ""
    assert rows[2]["MAE"] == 3.0
    assert output.exists()


def test_horizon_accumulator_is_global_and_bounded():
    target = np.arange(15, dtype=float).reshape(5, 3)
    prediction = target + np.array([[1.0, 2.0, 3.0]])
    accumulator = HorizonRegressionAccumulator()
    accumulator.update(target[:2], prediction[:2])
    accumulator.update(target[2:], prediction[2:])
    result = accumulator.finalize()
    np.testing.assert_allclose(result["mae"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result["rmse"], [1.0, 2.0, 3.0])
    assert accumulator.absolute_error.shape == (3,)
    assert accumulator.squared_error.shape == (3,)


def test_experiment_name_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="single directory"):
        experiment_directory(tmp_path, "../outside")
