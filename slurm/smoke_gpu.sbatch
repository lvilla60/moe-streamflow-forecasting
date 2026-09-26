#!/usr/bin/env bash
#SBATCH --job-name=moe-gpu-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p logs checkpoints/khipu_smoke outputs/experiments

module purge
module load gnu12/12.4.0
module load python3/3.11.11
module load cuda/12.8

VENV_PATH="${VENV_PATH:-$HOME/.venvs/moe-streamflow-py311-gnu12-cu128}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Virtual environment not found: $VENV_PATH" >&2
    exit 1
fi
source "$VENV_PATH/bin/activate"

hostname
date
pwd
command -v python
python --version
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable"; fi
python -c 'import torch; print("torch CUDA available:", torch.cuda.is_available())'
python -c 'import torch, sys; sys.exit(0) if torch.cuda.is_available() else sys.exit("CUDA is unavailable; refusing to run the GPU smoke test")'

python -m scripts.train_expert \
    --model lstm \
    --epochs 1 \
    --batch-size 32 \
    --num-workers 2 \
    --max-train-batches 2 \
    --max-validation-batches 2 \
    --device cuda \
    --checkpoint-dir checkpoints/khipu_smoke \
    --experiment-name lstm_khipu_smoke
