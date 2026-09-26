#!/usr/bin/env bash
#SBATCH --job-name=moe-train-expert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p logs checkpoints outputs/experiments

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
python -c 'import torch, sys; sys.exit(0) if torch.cuda.is_available() else sys.exit("CUDA is unavailable; expert training requires a GPU")'

MODEL="${MODEL:-lstm}"
case "$MODEL" in lstm|gru|seq2seq_attention|informer) ;; *) echo "Invalid MODEL: $MODEL" >&2; exit 2 ;; esac
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL}_khipu}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/${EXPERIMENT_NAME}}"
INPUT="${INPUT:-Input/train-001.h5}"
STATS="${STATS:-outputs/train_stats.json}"

set -- python -m scripts.train_expert \
    --model "$MODEL" --input "$INPUT" --stats "$STATS" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY" \
    --device cuda --checkpoint-dir "$CHECKPOINT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --hidden-size "${HIDDEN_SIZE:-32}" --layers "${LAYERS:-1}" \
    --dropout "${DROPOUT:-0.1}" \
    --teacher-forcing-ratio "${TEACHER_FORCING_RATIO:-0.5}" \
    --gradient-clip "${GRADIENT_CLIP:-1.0}" \
    --d-model "${D_MODEL:-64}" --heads "${HEADS:-4}" \
    --decoder-layers "${DECODER_LAYERS:-1}" --d-ff "${D_FF:-128}" \
    --factor "${FACTOR:-5}" --label-len "${LABEL_LEN:-48}"

if [ -n "${RESUME:-}" ]; then set -- "$@" --resume "$RESUME"; fi
"$@"
