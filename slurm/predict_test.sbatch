#!/usr/bin/env bash
#SBATCH --job-name=moe-predict-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}"
cd "$PROJECT_ROOT"
REPO_ROOT="$(pwd -P)"
if [ ! -f scripts/predict_test.py ] || [ ! -d src ]; then
    echo "Repository root not found: $REPO_ROOT; set PROJECT_ROOT explicitly" >&2
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${OUTPUT:?Set OUTPUT to a new prediction CSV path}"
if [ -e "$OUTPUT" ] || [ -L "$OUTPUT" ]; then
    echo "Output already exists; choose a new OUTPUT: $OUTPUT" >&2
    exit 1
fi

INPUT="${INPUT:-Input/test.h5}"
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
DEVICE="${DEVICE:-cuda}"

if [ ! -f "$INPUT" ]; then echo "Test input not found: $INPUT" >&2; exit 1; fi
if [ ! -f "$ROUTER_CHECKPOINT" ]; then echo "Router checkpoint not found: $ROUTER_CHECKPOINT" >&2; exit 1; fi
mkdir -p logs

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
if [[ "$DEVICE" == cuda* ]]; then
    python -c 'import torch, sys; sys.exit(0) if torch.cuda.is_available() else sys.exit("CUDA is unavailable; test prediction requested a GPU")'
fi

# Python loads only the checkpoints selected by the router artifact's mapping.
cmd=(
    python -m scripts.predict_test
    --input "$INPUT"
    --lstm-checkpoint "$LSTM_CHECKPOINT" --gru-checkpoint "$GRU_CHECKPOINT"
    --seq2seq-checkpoint "$SEQ2SEQ_CHECKPOINT" --informer-checkpoint "$INFORMER_CHECKPOINT"
    --router "$ROUTER" --router-checkpoint "$ROUTER_CHECKPOINT"
    --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-0}"
    --device "$DEVICE" --output "$OUTPUT"
)

if [ -n "${EXPERTS:-}" ]; then cmd+=(--experts "$EXPERTS"); fi
if [ -n "${MAX_BATCHES:-}" ]; then cmd+=(--max-batches "$MAX_BATCHES"); fi
"${cmd[@]}"
