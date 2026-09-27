# MoE Streamflow Forecasting

Forecast 48 hourly specific-discharge values from 336 historical hourly timesteps using a Mixture-of-Experts architecture adapted from a research paper. The model uses 12 historical input channels, four forecasting experts, and three alternative routers. Routing is sample-level and hard: one expert supplies the complete 48-hour forecast. The implementation uses PyTorch and scikit-learn and does not report model performance before experiments are run.

Run project commands from the repository root using `python -m ...`.

## 1. Data contract

| Array | Shape | Meaning |
|---|---|---|
| `X` | `[N, 336, 12]` | Historical hourly inputs |
| `y` | `[N, 48]` | Future specific-discharge target |
| `y_aux` | `[N, 48, 11]` | Optional future meteorological variables; unused by current models |

Channel 11 of `X` is historical `specific_discharge`; `y` contains future `specific_discharge`. The official train and validation assignments are read from the `split` array in `Input/train-001.h5`; no random split is created. The test file contains `X` and `basin_id`, without `y` or `y_aux`.

| Split | Samples |
|---|---:|
| Train | 254,000 |
| Validation | 18,142 |
| Test | 27,983 |

See `Input/metadata.json` for channel names and their order. Large HDF5 and CSV data files are excluded from Git.

## 2. Architecture overview

```text
X [B, 336, 12]
    |
    +--> LSTM --------------------+
    +--> GRU ---------------------+
    +--> LSTM-S2S-Attention ------+--> expert forecasts [B, N, 48]
    +--> Informer ----------------+
                                   |
                          best-expert labels
                                   |
                        RF / LSTM / Transformer
                                   |
                            selected expert
                                   v
                               [B, 48]
```

Experts train independently and are frozen before router-label generation. The router learns which expert to select for a sample. Routing selects one expert for all 48 forecast hours; it does not average forecasts. Experts and routers are trained sequentially, not jointly end to end.

## 3. Expert models

- **LSTM:** recurrent encoder followed by a direct 48-step projection.
- **GRU:** GRU encoder followed by a direct 48-step projection.
- **LSTM Seq2Seq + Attention:** LSTM encoder, additive/Bahdanau attention, and autoregressive decoder. Teacher forcing is used during training only; inference does not need future targets.
- **Informer:** adapted implementation with input projection, sinusoidal positional encoding, ProbSparse-style encoder attention, encoder distillation, causal decoder self-attention, encoder-decoder cross-attention, and a 48-step output. The encoder uses the project’s simplified ProbSparse implementation; the decoder uses full causal self-attention and full cross-attention to preserve causality. Explicit calendar timestamp embeddings are omitted because timestamps are unavailable.

## 4. Normalization

Normalization statistics are computed from the training split only. Each `X` channel is normalized independently; `y` uses one scalar mean and standard deviation. Expert predictions are inverse-transformed before physical-unit MAE, RMSE, and NSE are calculated. The Seq2Seq decoder start uses the last raw historical discharge transformed with the **target** (`y`) statistics, rather than the input channel-11 statistics.

The saved statistics are in `outputs/train_stats.json`. Generate them with:

```text
python -m scripts.compute_train_stats
```

## 5. Router labels

For sample `i` and expert `e`, the label error is the mean absolute error over the complete 48-hour horizon:

```text
error(i, e) = mean(abs(y_true[i, :] - y_pred[e, i, :]))
best_expert(i) = argmin_e error(i, e)
```

For the default four-expert configuration, the local class mapping is:

| Class | Expert |
|---:|---|
| 0 | LSTM |
| 1 | GRU |
| 2 | LSTM-S2S-Attention |
| 3 | Informer |

For expert subsets, class indices are reassigned locally and contiguously in the
requested expert order. For example, `lstm,informer` maps class 0 to LSTM and class 1
to Informer.

Generate training labels from the official training split. Validation labels are optional and are only for evaluation or validation-based router selection. Label files store original HDF5 row `Id`, `basin_id`, class, and expert errors; they do not duplicate input arrays.

## 6. Routers

- **Random Forest:** scikit-learn `RandomForestClassifier` on flattened normalized history `[336 * 12]`. Full training can require several GiB for features plus forest and worker memory. Use `--max-samples` to limit rows and reduce `--n-jobs` when memory is constrained.
- **LSTM router:** LSTM sequence encoder with an N-class output head, where N is the number of selected experts.
- **Transformer router:** input projection, positional encoding, Transformer encoder, pooled representation, and an N-class output head.

For neural routers, supplying validation labels evaluates each epoch and selects the checkpoint with the lowest validation loss. Without validation labels, the final epoch is saved and is not validation-selected.

## 7. Hard MoE inference

The selected expert subset is loaded in the ordered mapping stored in the router artifact. The experts produce `[B, N, 48]`, where N is the number of experts in the selected subset, and the router produces one class per sample `[B]`. Hard routing selects one expert per sample and returns `[B, 48]`; there is no soft averaging. The chosen forecast is inverse-transformed to physical discharge units.

## 8. Repository structure

```text
Input/                       HDF5 data, metadata, and optional CSV files
notebooks/                   Data and baseline notebook
scripts/
  compute_train_stats.py
  evaluate_persistence.py
  train_expert.py
  generate_router_labels.py
  train_router.py
  evaluate_moe.py
  predict_test.py
src/
  dataset.py
  preprocessing.py
  baseline.py
  metrics.py
  training.py
  checkpointing.py
  routing.py
  inference.py
  models/
    common.py
    experts.py
    informer.py
    routers.py
    moe.py
tests/                       Unit and synthetic integration tests
checkpoints/                 Expert and router checkpoints
outputs/                     Statistics, labels, metrics, and predictions
README.md
requirements.txt
pytest.ini
```

## 9. Installation

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project code supports CPU or CUDA devices through configuration. A PyTorch GPU environment can be configured separately for the target system.

## 10. Testing

Run the test suite from the repository root:

```text
python -m pytest -q
```

The suite covers dataset utilities, preprocessing, metrics, four experts, Informer causality, routers, best-expert labels, hard MoE selection, checkpoint save/load/resume, limited test inference, and synthetic end-to-end integration. To inspect the real HDF5 loader’s first samples, run:

```text
python -m tests.smoke_real_dataset
```

## 11. Persistence baseline

The persistence forecast repeats the last observed historical discharge for all 48 future hours. Evaluate it on the official validation split with:

```text
python -m scripts.evaluate_persistence
```

Results are written to `outputs/persistence_validation.json`.

## 12. Train an expert

Available `--model` values are `lstm`, `gru`, `seq2seq_attention`, and `informer`. Expert checkpoints are saved under `checkpoints/` by default and the best checkpoint is selected by validation loss.

LSTM:

```text
python -m scripts.train_expert --model lstm --epochs 20 --batch-size 64 --device cpu
```

Informer:

```text
python -m scripts.train_expert --model informer --epochs 20 --batch-size 64 --device cpu
```

Lightweight LSTM smoke run:

```text
python -m scripts.train_expert --model lstm --epochs 1 --batch-size 2 --max-train-batches 1 --max-validation-batches 1 --device cpu --checkpoint-dir checkpoints/smoke
```

## 13. Generate router labels

Provide the four expert checkpoints in class order. Training labels:

```text
python -m scripts.generate_router_labels --split train --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --gru-checkpoint checkpoints/gru_baseline/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_baseline/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_baseline/informer_best.pt
```

Validation labels:

```text
python -m scripts.generate_router_labels --split validation --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --gru-checkpoint checkpoints/gru_baseline/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_baseline/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_baseline/informer_best.pt
```

Default outputs are `outputs/router_labels_train.npz` and `outputs/router_labels_validation.npz`. Use these generated files with the corresponding split; label IDs are checked against the official HDF5 split.

### Expert subsets

Router labels, routers, MoE evaluation, and test prediction support ordered expert
subsets. The default remains `lstm,gru,seq2seq,informer`. Paper and best-expert
ablations use `--experts lstm,informer` and `--experts lstm,gru`, respectively.
Router class indices are local and contiguous in the requested order: for
`lstm,informer`, class 0 is LSTM and class 1 is Informer.

```text
python -m scripts.generate_router_labels --experts lstm,informer --split train --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --informer-checkpoint checkpoints/informer_baseline/informer_best.pt
python -m scripts.generate_router_labels --experts lstm,gru --split train --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --gru-checkpoint checkpoints/gru_baseline/gru_best.pt
```

The trained router artifact stores this mapping. `evaluate_moe.py` and
`predict_test.py` reconstruct it automatically; an optional `--experts` value is
checked against the stored order.

## 14. Train a router

The examples use train labels and optional validation labels. RF writes a `.joblib` artifact; neural routers write `.pt` checkpoints.

```text
python -m scripts.train_router --router rf --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz --max-samples 10000 --n-jobs 2
python -m scripts.train_router --router lstm --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz
python -m scripts.train_router --router transformer --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz
```

Validation labels are recommended for neural-router checkpoint selection. RF feature flattening can use substantial memory; `--max-samples` and a smaller `--n-jobs` help constrain it.

## 15. Evaluate the MoE

Supply the expert checkpoints selected by the router artifact and one router artifact. `--validation-labels` is optional and enables router accuracy and confusion-matrix reporting.

```text
python -m scripts.evaluate_moe --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --gru-checkpoint checkpoints/gru_baseline/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_baseline/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_baseline/informer_best.pt --router lstm --router-checkpoint checkpoints/router_lstm.pt --validation-labels outputs/router_labels_validation.npz
```

Evaluation reports physical-unit MAE, RMSE, NSE, and router class frequencies. With validation labels it also reports classification accuracy and a confusion matrix; partial label files report their matched-row coverage.

## 16. Test prediction

Test inference reads `test.h5` without targets and writes `outputs/test_predictions.csv` with columns `Id`, `q_01`, …, `q_48`:

```text
python -m scripts.predict_test --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --gru-checkpoint checkpoints/gru_baseline/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_baseline/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_baseline/informer_best.pt --router lstm --router-checkpoint checkpoints/router_lstm.pt
```

Use `--max-batches N` for a bounded smoke run. It warns that the file is partial and reports the number of rows written; keep partial smoke CSVs separate from full submissions.

## Experiment outputs

Training and evaluation commands write compact artifacts under
`outputs/experiments/<experiment_name>/`: `config.json` records the run settings,
`history.csv` records epoch history when applicable, `metrics.json` records final
validation results, and `plots/` contains presentation-ready PNG figures. Names are
deterministic when `--experiment-name` is omitted.

```text
python -m scripts.train_expert --model lstm --epochs 20 --experiment-name lstm_baseline
python -m scripts.train_router --router lstm --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz --experiment-name router_lstm_baseline
python -m scripts.evaluate_moe --lstm-checkpoint checkpoints/lstm_baseline/lstm_best.pt --gru-checkpoint checkpoints/gru_baseline/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_baseline/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_baseline/informer_best.pt --router lstm --router-checkpoint checkpoints/router_lstm.pt --experiment-name moe_lstm_baseline
```

Create `outputs/experiments/summary.csv` from completed runs with:

```text
python -m scripts.summarize_experiments
```

The summary also includes the existing default
`outputs/persistence_validation.json` artifact when it is present.

## Khipu / SLURM

### Quick start for collaborators

For execution or testing on Khipu, use `main`: clone the `main` branch, or update an
existing checkout before submitting jobs.

```sh
git checkout main
git pull

sbatch slurm/check_environment.sbatch
sbatch slurm/smoke_gpu.sbatch
```

Use the provided `slurm/*.sbatch` wrappers. Do not modify `src/models/`,
`src/training.py`, preprocessing, dataset split logic, or normalization logic merely
to run an experiment. Change experiment parameters through `sbatch --export`
variables instead. Heavy `Input/*.h5` files are transferred separately, and generated
logs, checkpoints, and outputs remain outside normal source commits.

For execution and testing, use `main`. For code development, create a feature branch
and submit a Pull Request.

Clone the repository on Khipu, enter its root, and create the planned Python 3.11
environment:

```sh
git clone --branch main https://github.com/lvilla60/moe-streamflow-forecasting.git
cd moe-streamflow-forecasting
module purge
module load gnu12/12.4.0
module load python3/3.11.11
module load cuda/12.8
python3 -m venv "$HOME/.venvs/moe-streamflow-py311-gnu12-cu128"
source "$HOME/.venvs/moe-streamflow-py311-gnu12-cu128/bin/activate"
python -m pip install --upgrade pip
```

Install a PyTorch 2.11 or newer build compatible with CUDA 12.8 using the current
Khipu-approved installation method, then install the project requirements. Because
`requirements.txt` does not pin a CUDA wheel, an already compatible PyTorch install
is retained.

```sh
python -m pip install -r requirements.txt
```

Transfer `Input/train-001.h5`, `Input/test.h5`, and `Input/metadata.json` separately;
they are intentionally outside version control. Also transfer the existing
`outputs/train_stats.json`, or generate it once from the training split with
`python -m scripts.compute_train_stats`. If test targets are provided separately, keep
them outside version control. They may be placed locally under `Input/` for final
evaluation, but must not be committed to Git.

Create the scheduler log directory before the first submission:

```sh
mkdir -p Input outputs logs
```

Check the loaded environment and run the bounded CUDA smoke job:

```sh
sbatch slurm/check_environment.sbatch
sbatch slurm/smoke_gpu.sbatch
```

Submit one expert per job. The default checkpoint directory is
`checkpoints/<experiment_name>/`.

```sh
sbatch --export=ALL,MODEL=lstm,EXPERIMENT_NAME=lstm_baseline slurm/train_expert.sbatch
sbatch --export=ALL,MODEL=gru,EXPERIMENT_NAME=gru_baseline slurm/train_expert.sbatch
sbatch --export=ALL,MODEL=seq2seq_attention,EXPERIMENT_NAME=seq2seq_baseline slurm/train_expert.sbatch
sbatch --export=ALL,MODEL=informer,EXPERIMENT_NAME=informer_baseline slurm/train_expert.sbatch
```

Generate labels for both official splits after all expert checkpoints exist:

```sh
sbatch --export=ALL,SPLIT=train,OUTPUT=outputs/router_labels_train.npz slurm/generate_router_labels.sbatch
sbatch --export=ALL,SPLIT=validation,OUTPUT=outputs/router_labels_validation.npz slurm/generate_router_labels.sbatch
```

Generate labels for the paper ablation (`EXPERTS=lstm,informer`) and best-expert
ablation (`EXPERTS=lstm,gru`) with the reusable wrapper:

```sh
sbatch --export=ALL,EXPERTS=lstm,informer,SPLIT=train,OUTPUT=outputs/router_labels_lstm_informer_train.npz slurm/generate_router_labels.sbatch
sbatch --export=ALL,EXPERTS=lstm,informer,SPLIT=validation,OUTPUT=outputs/router_labels_lstm_informer_validation.npz slurm/generate_router_labels.sbatch
sbatch --export=ALL,EXPERTS=lstm,gru,SPLIT=train,OUTPUT=outputs/router_labels_lstm_gru_train.npz slurm/generate_router_labels.sbatch
sbatch --export=ALL,EXPERTS=lstm,gru,SPLIT=validation,OUTPUT=outputs/router_labels_lstm_gru_validation.npz slurm/generate_router_labels.sbatch
```

Train routers with the generated labels:

```sh
sbatch --export=ALL,ROUTER=rf,DEVICE=cpu,EXPERIMENT_NAME=router_rf_baseline slurm/train_router.sbatch
sbatch --export=ALL,ROUTER=lstm,EXPERIMENT_NAME=router_lstm_baseline slurm/train_router.sbatch
sbatch --export=ALL,ROUTER=transformer,EXPERIMENT_NAME=router_transformer_baseline slurm/train_router.sbatch
```

The reusable router script requests one GPU because `#SBATCH` directives are parsed
before exported variables. RF itself runs on CPU; sites that require strict CPU-only
allocation can copy the resource header and remove the GPU directives.

Evaluate an MoE on the official validation split by selecting its router artifact:

```sh
sbatch --export=ALL,ROUTER=lstm,EXPERIMENT_NAME=moe_lstm_baseline slurm/evaluate_moe.sbatch
```

All wrappers may be edited for site-specific walltimes. Common settings such as
`BATCH_SIZE`, `NUM_WORKERS`, checkpoint paths, and experiment names can be overridden
with `sbatch --export=ALL,...`. Logs are written to `logs/`, checkpoints to
`checkpoints/`, and reports to `outputs/experiments/`.

## 17. Generated artifacts

`checkpoints/` stores expert and neural-router `.pt` files and Random Forest `.joblib` files. `outputs/` stores training statistics, persistence and MoE metrics, router-label archives, and test predictions. Generated artifacts and large datasets are excluded from Git.

## 18. Methodological adaptations

The implementation retains the provided official train/validation split and forecasts 48 hourly steps. It omits `y_aux`, routes each sample to one expert across the complete horizon, uses relative positional encoding for Informer without calendar timestamps, and flattens normalized histories for Random Forest. Experts and routers train sequentially without joint end-to-end optimization.
