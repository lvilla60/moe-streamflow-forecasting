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
    +--> LSTM-S2S-Attention ------+--> expert forecasts [B, 4, 48]
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

| Class | Expert |
|---:|---|
| 0 | LSTM |
| 1 | GRU |
| 2 | LSTM-S2S-Attention |
| 3 | Informer |

Generate training labels from the official training split. Validation labels are optional and are only for evaluation or validation-based router selection. Label files store original HDF5 row `Id`, `basin_id`, class, and expert errors; they do not duplicate input arrays.

## 6. Routers

- **Random Forest:** scikit-learn `RandomForestClassifier` on flattened normalized history `[336 * 12]`. Full training can require several GiB for features plus forest and worker memory. Use `--max-samples` to limit rows and reduce `--n-jobs` when memory is constrained.
- **LSTM router:** LSTM sequence encoder with a four-class output head.
- **Transformer router:** input projection, positional encoding, Transformer encoder, pooled sequence representation, and four-class output head.

For neural routers, supplying validation labels evaluates each epoch and selects the checkpoint with the lowest validation loss. Without validation labels, the final epoch is saved and is not validation-selected.

## 7. Hard MoE inference

The experts produce `[B, 4, 48]` and the router produces one class per sample `[B]`. A hard gather selects one expert forecast and returns `[B, 48]`; there is no soft averaging. The implementation may run all four experts before selecting one. The chosen forecast is inverse-transformed to physical discharge units.

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
python -m scripts.generate_router_labels --split train --lstm-checkpoint checkpoints/lstm_best.pt --gru-checkpoint checkpoints/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_best.pt
```

Validation labels:

```text
python -m scripts.generate_router_labels --split validation --lstm-checkpoint checkpoints/lstm_best.pt --gru-checkpoint checkpoints/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_best.pt
```

Default outputs are `outputs/router_labels_train.npz` and `outputs/router_labels_validation.npz`. Use these generated files with the corresponding split; label IDs are checked against the official HDF5 split.

## 14. Train a router

The examples use train labels and optional validation labels. RF writes a `.joblib` artifact; neural routers write `.pt` checkpoints.

```text
python -m scripts.train_router --router rf --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz --max-samples 10000 --n-jobs 2
python -m scripts.train_router --router lstm --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz
python -m scripts.train_router --router transformer --labels outputs/router_labels_train.npz --validation-labels outputs/router_labels_validation.npz
```

Validation labels are recommended for neural-router checkpoint selection. RF feature flattening can use substantial memory; `--max-samples` and a smaller `--n-jobs` help constrain it.

## 15. Evaluate the MoE

Supply four expert checkpoints and one router artifact. `--validation-labels` is optional and enables router accuracy and confusion-matrix reporting.

```text
python -m scripts.evaluate_moe --lstm-checkpoint checkpoints/lstm_best.pt --gru-checkpoint checkpoints/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_best.pt --router lstm --router-checkpoint checkpoints/router_lstm.pt --validation-labels outputs/router_labels_validation.npz
```

Evaluation reports physical-unit MAE, RMSE, NSE, and router class frequencies. With validation labels it also reports classification accuracy and a confusion matrix; partial label files report their matched-row coverage.

## 16. Test prediction

Test inference reads `test.h5` without targets and writes `outputs/test_predictions.csv` with columns `Id`, `q_01`, …, `q_48`:

```text
python -m scripts.predict_test --lstm-checkpoint checkpoints/lstm_best.pt --gru-checkpoint checkpoints/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_best.pt --router lstm --router-checkpoint checkpoints/router_lstm.pt
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
python -m scripts.evaluate_moe --lstm-checkpoint checkpoints/lstm_best.pt --gru-checkpoint checkpoints/gru_best.pt --seq2seq-checkpoint checkpoints/seq2seq_attention_best.pt --informer-checkpoint checkpoints/informer_best.pt --router lstm --router-checkpoint checkpoints/router_lstm.pt --experiment-name moe_lstm_baseline
```

Create `outputs/experiments/summary.csv` from completed runs with:

```text
python -m scripts.summarize_experiments
```

The summary also includes the existing default
`outputs/persistence_validation.json` artifact when it is present.

## 17. Generated artifacts

`checkpoints/` stores expert and neural-router `.pt` files and Random Forest `.joblib` files. `outputs/` stores training statistics, persistence and MoE metrics, router-label archives, and test predictions. Generated artifacts and large datasets are excluded from Git.

## 18. Methodological adaptations

The implementation retains the provided official train/validation split and forecasts 48 hourly steps. It omits `y_aux`, routes each sample to one expert across the complete horizon, uses relative positional encoding for Informer without calendar timestamps, and flattens normalized histories for Random Forest. Experts and routers train sequentially without joint end-to-end optimization.
