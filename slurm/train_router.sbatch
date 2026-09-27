#!/usr/bin/env bash
#SBATCH --job-name=moe-train-router
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
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
if [ ! -f "$VENV_PATH/bin/activate" ]; then echo "Virtual environment not found: $VENV_PATH" >&2; exit 1; fi
source "$VENV_PATH/bin/activate"

hostname
date
pwd
command -v python
python --version
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable"; fi
python -c 'import torch; print("torch CUDA available:", torch.cuda.is_available())'

ROUTER="${ROUTER:-lstm}"
case "$ROUTER" in rf|lstm|transformer) ;; *) echo "Invalid ROUTER: $ROUTER" >&2; exit 2 ;; esac
if [ "$ROUTER" != "rf" ]; then
    python -c 'import torch, sys; sys.exit(0) if torch.cuda.is_available() else sys.exit("CUDA is unavailable; neural router training requires a GPU")'
fi

LABELS="${LABELS:-outputs/router_labels_train.npz}"
VALIDATION_LABELS="${VALIDATION_LABELS-outputs/router_labels_validation.npz}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-router_${ROUTER}_baseline}"
case "$ROUTER" in
    rf) DEFAULT_OUTPUT="checkpoints/${EXPERIMENT_NAME}/router_${ROUTER}.joblib" ;;
    lstm|transformer) DEFAULT_OUTPUT="checkpoints/${EXPERIMENT_NAME}/router_${ROUTER}.pt" ;;
esac
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
if [ ! -f "$LABELS" ]; then echo "Training labels not found: $LABELS" >&2; exit 1; fi
if [ -n "$VALIDATION_LABELS" ] && [ ! -f "$VALIDATION_LABELS" ]; then
    echo "Validation labels not found: $VALIDATION_LABELS" >&2
    exit 1
fi

set -- python -m scripts.train_router \
    --router "$ROUTER" --input "${INPUT:-Input/train-001.h5}" \
    --stats "${STATS:-outputs/train_stats.json}" --labels "$LABELS" \
    --output "$OUTPUT" --epochs "${EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-4}" \
    --learning-rate "${LEARNING_RATE:-0.001}" --weight-decay "${WEIGHT_DECAY:-0.001}" \
    --device "${DEVICE:-cuda}" --seed "${SEED:-42}" \
    --n-estimators "${N_ESTIMATORS:-100}" --n-jobs "${N_JOBS:-4}" \
    --hidden-size "${HIDDEN_SIZE:-32}" --layers "${LAYERS:-1}" \
    --d-model "${D_MODEL:-32}" --heads "${HEADS:-4}" \
    --d-ff "${D_FF:-64}" --dropout "${DROPOUT:-0.1}" \
    --experiment-name "$EXPERIMENT_NAME"

if [ -n "$VALIDATION_LABELS" ]; then set -- "$@" --validation-labels "$VALIDATION_LABELS"; fi
if [ -n "${MAX_SAMPLES:-}" ]; then set -- "$@" --max-samples "$MAX_SAMPLES"; fi
if [ -n "${EXPERTS:-}" ]; then set -- "$@" --experts "$EXPERTS"; fi
"$@"
