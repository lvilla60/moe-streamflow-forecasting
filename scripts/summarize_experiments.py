"""Create a compact comparison table from completed experiment metrics."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("experiment_name", "kind", "model", "best_epoch", "MAE", "RMSE",
          "NSE", "validation_accuracy")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _value(metrics, upper_name):
    if metrics.get(upper_name) is not None:
        return metrics[upper_name]
    return metrics.get("global", {}).get(upper_name.lower(), "")


def summarize(experiments_root, output):
    experiments_root = Path(experiments_root)
    rows = []
    if experiments_root.exists():
        for directory in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
            metrics_path = directory / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            config = _read_json(directory / "config.json") if (directory / "config.json").exists() else {}
            rows.append({
                "experiment_name": directory.name,
                "kind": metrics.get("kind", config.get("kind", "")),
                "model": metrics.get("model", config.get("model", "")),
                "best_epoch": metrics.get("best_epoch", ""),
                "MAE": _value(metrics, "MAE"),
                "RMSE": _value(metrics, "RMSE"),
                "NSE": _value(metrics, "NSE"),
                "validation_accuracy": metrics.get(
                    "validation_accuracy", metrics.get("router_accuracy", "")
                ),
            })

    # Keep the existing persistence evaluator unchanged while making its default
    # artifact available for comparison when present next to experiments/.
    persistence_path = experiments_root.parent / "persistence_validation.json"
    if persistence_path.exists():
        metrics = _read_json(persistence_path)
        rows.append({
            "experiment_name": "persistence_validation", "kind": "baseline",
            "model": "persistence", "best_epoch": "",
            "MAE": _value(metrics, "MAE"), "RMSE": _value(metrics, "RMSE"),
            "NSE": _value(metrics, "NSE"), "validation_accuracy": "",
        })

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", default="outputs/experiments")
    parser.add_argument("--output", default="outputs/experiments/summary.csv")
    args = parser.parse_args()
    experiments_root = Path(args.experiments_root)
    output = Path(args.output)
    if not experiments_root.is_absolute():
        experiments_root = ROOT / experiments_root
    if not output.is_absolute():
        output = ROOT / output
    rows = summarize(experiments_root, output)
    print(f"Wrote {len(rows)} completed experiments to {output}")


if __name__ == "__main__":
    main()
