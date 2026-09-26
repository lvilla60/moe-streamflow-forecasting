"""Train one forecasting expert on the official train/validation split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import CaudalDataset
from src.models import create_expert
from src.training import Normalization, fit_expert, set_seed


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def model_config(args):
    common = {"input_size": 12, "forecast_horizon": 48, "dropout": args.dropout}
    if args.model in {"lstm", "gru", "seq2seq_attention"}:
        return {**common, "hidden_size": args.hidden_size, "num_layers": args.layers}
    return {
        **common, "label_len": args.label_len, "d_model": args.d_model,
        "n_heads": args.heads, "encoder_layers": args.layers,
        "decoder_layers": args.decoder_layers, "d_ff": args.d_ff,
        "factor": args.factor,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("lstm", "gru", "seq2seq_attention", "informer"))
    parser.add_argument("--input", default="Input/train-001.h5")
    parser.add_argument("--stats", default="outputs/train_stats.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--label-len", type=int, default=48)
    args = parser.parse_args()

    set_seed(args.seed)
    config = model_config(args)
    model = create_expert(args.model, **config)
    normalization = Normalization.from_json(resolve(args.stats))
    train_data = CaudalDataset(resolve(args.input), split="train")
    validation_data = CaudalDataset(resolve(args.input), split="validation")
    try:
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )
        validation_loader = DataLoader(
            validation_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        checkpoint = resolve(args.checkpoint_dir) / f"{args.model}_best.pt"
        fit_expert(
            model, train_loader, validation_loader, normalization,
            model_name=args.model, model_config=config, epochs=args.epochs,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            device=args.device, gradient_clip=args.gradient_clip,
            teacher_forcing_ratio=(args.teacher_forcing_ratio
                                   if args.model == "seq2seq_attention" else 0.0),
            max_train_batches=args.max_train_batches,
            max_validation_batches=args.max_validation_batches,
            checkpoint_path=checkpoint,
            resume_checkpoint=None if args.resume is None else resolve(args.resume),
        )
        print(f"Best checkpoint: {checkpoint}")
    finally:
        train_data.close()
        validation_data.close()


if __name__ == "__main__":
    main()
