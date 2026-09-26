"""Small, headless-safe experiment artifact helpers."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_LABELS = ("LSTM", "GRU", "Seq2Seq-Attention", "Informer")


def experiment_directory(root, experiment_name):
    if (not experiment_name or Path(experiment_name).is_absolute()
            or len(Path(experiment_name).parts) != 1
            or experiment_name in {".", ".."}):
        raise ValueError("experiment name must be a single directory name")
    path = Path(root) / experiment_name
    (path / "plots").mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_history_csv(path, history):
    """Write either a list of row dictionaries or a column-oriented dictionary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(history, dict):
        lengths = {len(values) for values in history.values()}
        if len(lengths) > 1:
            raise ValueError("history columns must have equal lengths")
        rows = [dict(zip(history, values)) for values in zip(*history.values())]
        fieldnames = list(history)
    else:
        rows = list(history)
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_line_plot(path, x, series, *, xlabel, ylabel):
    fig, axis = plt.subplots()
    for label, values in series.items():
        if len(values):
            axis.plot(x[:len(values)], values, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if sum(bool(len(values)) for values in series.values()) > 1:
        axis.legend()
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def _has_values(values):
    return bool(values) and any(value is not None for value in values)


def plot_loss_curve(path, history, ylabel="Loss"):
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    series = {"Train": history.get("train_loss", [])}
    if _has_values(history.get("validation_loss", [])):
        series["Validation"] = history["validation_loss"]
    _save_line_plot(path, epochs, series, xlabel="Epoch", ylabel=ylabel)


def plot_metric_curve(path, history, metric, ylabel=None):
    epochs = history.get("epoch", list(range(1, len(history.get(metric, [])) + 1)))
    _save_line_plot(path, epochs, {metric.replace("_", " ").title(): history.get(metric, [])},
                    xlabel="Epoch", ylabel=ylabel or metric.replace("_", " ").upper())


def plot_accuracy_curve(path, history):
    epochs = history.get("epoch", list(range(1, len(history.get("train_accuracy", [])) + 1)))
    series = {"Train": history.get("train_accuracy", [])}
    if _has_values(history.get("validation_accuracy", [])):
        series["Validation"] = history["validation_accuracy"]
    _save_line_plot(path, epochs, series, xlabel="Epoch", ylabel="Accuracy")


def plot_horizon_metric(path, values, metric):
    values = list(values)
    _save_line_plot(path, list(range(1, len(values) + 1)), {metric.upper(): values},
                    xlabel="Forecast hour", ylabel=metric.upper())


def plot_confusion_matrix(path, confusion, class_labels=CLASS_LABELS):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (len(class_labels), len(class_labels)):
        raise ValueError("confusion matrix shape must match class labels")
    fig, axis = plt.subplots()
    image = axis.imshow(confusion)
    axis.set_xticks(range(len(class_labels)), class_labels, rotation=30, ha="right")
    axis.set_yticks(range(len(class_labels)), class_labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_class_frequency(path, predicted, true=None, class_labels=CLASS_LABELS):
    predicted = np.asarray(predicted)
    positions = np.arange(len(class_labels))
    fig, axis = plt.subplots()
    if true is None:
        axis.bar(positions, predicted, label="Predicted")
    else:
        width = 0.4
        axis.bar(positions - width / 2, predicted, width=width, label="Predicted")
        axis.bar(positions + width / 2, np.asarray(true), width=width, label="True")
        axis.legend()
    axis.set_xticks(positions, class_labels, rotation=30, ha="right")
    axis.set_xlabel("Router class")
    axis.set_ylabel("Count")
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(path, values, labels):
    fig, axis = plt.subplots()
    positions = np.arange(len(values))
    axis.bar(positions, values)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_xlabel("Input channel")
    axis.set_ylabel("Aggregated feature importance")
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)
