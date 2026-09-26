#!/usr/bin/env bash
#SBATCH --job-name=moe-evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p logs outputs/experiments

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
python -c 'import torch, sys; sys.exit(0) if torch.cuda.is_available() else sys.exit("CUDA is unavailable; MoE evaluation requires a GPU")'

ROUTER="${ROUTER:-lstm}"
case "$ROUTER" in
    rf) DEFAULT_ROUTER_CHECKPOINT="checkpoints/router_rf_baseline/router_rf.joblib" ;;
    lstm|transformer) DEFAULT_ROUTER_CHECKPOINT="checkpoints/router_${ROUTER}_baseline/router_${ROUTER}.pt" ;;
    *) echo "Invalid ROUTER: $ROUTER" >&2; exit 2 ;;
esac
ROUTER_CHECKPOINT="${ROUTER_CHECKPOINT:-$DEFAULT_ROUTER_CHECKPOINT}"
LSTM_CHECKPOINT="${LSTM_CHECKPOINT:-checkpoints/lstm_baseline/lstm_best.pt}"
GRU_CHECKPOINT="${GRU_CHECKPOINT:-checkpoints/gru_baseline/gru_best.pt}"
SEQ2SEQ_CHECKPOINT="${SEQ2SEQ_CHECKPOINT:-checkpoints/seq2seq_baseline/seq2seq_attention_best.pt}"
INFORMER_CHECKPOINT="${INFORMER_CHECKPOINT:-checkpoints/informer_baseline/informer_best.pt}"
VALIDATION_LABELS="${VALIDATION_LABELS-outputs/router_labels_validation.npz}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-moe_${ROUTER}_baseline}"

for artifact in "$ROUTER_CHECKPOINT" "$LSTM_CHECKPOINT" "$GRU_CHECKPOINT" "$SEQ2SEQ_CHECKPOINT" "$INFORMER_CHECKPOINT"; do
    if [ ! -f "$artifact" ]; then echo "Required checkpoint not found: $artifact" >&2; exit 1; fi
done
if [ -n "$VALIDATION_LABELS" ] && [ ! -f "$VALIDATION_LABELS" ]; then
    echo "Validation labels not found: $VALIDATION_LABELS" >&2
    exit 1
fi

set -- python -m scripts.evaluate_moe \
    --input "${INPUT:-Input/train-001.h5}" \
    --lstm-checkpoint "$LSTM_CHECKPOINT" --gru-checkpoint "$GRU_CHECKPOINT" \
    --seq2seq-checkpoint "$SEQ2SEQ_CHECKPOINT" --informer-checkpoint "$INFORMER_CHECKPOINT" \
    --router "$ROUTER" --router-checkpoint "$ROUTER_CHECKPOINT" \
    --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-4}" \
    --device cuda --output "${OUTPUT:-outputs/${EXPERIMENT_NAME}_validation.json}" \
    --experiment-name "$EXPERIMENT_NAME"

if [ -n "$VALIDATION_LABELS" ]; then set -- "$@" --validation-labels "$VALIDATION_LABELS"; fi
"$@"
