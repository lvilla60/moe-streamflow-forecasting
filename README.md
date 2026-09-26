# MoE Streamflow Forecasting

## Objective

Forecast the next 48 hourly streamflow/discharge values from 336 hours of historical multivariate observations using a Mixture of Experts (MoE) architecture adapted from the selected paper.

## Dataset

- Historical input: 336 hours across 12 channels
- Target: 48 future hourly specific-discharge values
- Optional `y_aux`: 48 × 11 future meteorological variables
- Train: 254,000 samples
- Validation: 18,142 samples
- Test: 27,983 samples

Large HDF5 and CSV files are intentionally not tracked by Git.

## Project structure

```text
Input/         Real HDF5 and CSV data (not tracked)
notebooks/     Exploration and analysis notebooks
src/           Dataset, preprocessing, baseline, and metric utilities
tests/         Synthetic tests and real-data smoke tests
checkpoints/   Model checkpoints
outputs/       Predictions and evaluation outputs
```

## Implemented components

- Lazy HDF5 dataset loader
- Synthetic dataset tests
- Real HDF5 smoke test
- MAE, RMSE, and NSE metrics
- Persistence baseline
- Streaming normalization statistics
- Four forecasting experts: LSTM, GRU, LSTM Seq2Seq with Attention, and Informer
- Three routers: Random Forest, LSTM, and Transformer
- Sample-level best-expert labels and hard MoE selection
- Shared training, checkpointing, evaluation, and test-inference workflows

## Architecture

```text
X (336 × 12)
   |
   +--> LSTM ---------------------+
   +--> GRU ----------------------+
   +--> LSTM-S2S-Attention -------+--> expert predictions
   +--> Informer -----------------+
                                   |
                                   +--> best-expert labels (training)
                                   |
                                   +--> Router
                                         RF / LSTM / Transformer
                                                   |
                                                   v
                                            selected expert
                                                   |
                                                   v
                                         48-hour forecast
```

Experts are trained independently. After the experts are frozen, their errors define one best-expert class per complete 48-hour sample, and the router is trained to predict that class. The initial implementation uses sample-level hard routing and does not jointly train the experts and router.

`y_aux` is omitted from the initial architecture because it is unavailable during test inference. The persistence baseline remains the reference for learned-model comparisons. See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.

## Planned modeling stages

1. **Phase 1:** Data validation, normalization, and persistence baseline
2. **Phase 2:** Independent experts: LSTM, GRU, LSTM Seq2Seq + Attention, and Informer
3. **Phase 3:** Best-expert labels and routers using Random Forest, LSTM, and Transformer models
4. **Phase 4:** MoE integration, evaluation, ablation studies, and test inference

## Environment strategy

Local Windows development uses `.venv` with CPU execution.

Khipu uses an independent project virtual environment with GNU 12.4, Python 3.11, CUDA 12.8, and PyTorch 2.11 with CUDA support.

BirdSong is only an infrastructure reference for Khipu/SLURM and is not part of this project.

## Testing

Run project modules from the repository root using `python -m ...`. Do not execute files inside `scripts/` directly.

Run tests and the real-data smoke test from the repository root:

```text
python -m pytest -q
python -m pytest -q tests/test_end_to_end_synthetic.py
python -m tests.smoke_real_dataset
python -m scripts.compute_train_stats
python -m scripts.evaluate_persistence
```

Model workflows are also executed as modules. For example:

```text
python -m scripts.train_expert --model lstm
python -m scripts.generate_router_labels --help
python -m scripts.train_router --help
python -m scripts.evaluate_moe --help
python -m scripts.predict_test --help
```

Real neural-model results are not reported until the independent experts and routers have been trained and evaluated.

Neural routers use `--validation-labels` to select the best epoch by validation loss; without it, they save the final epoch. RF training can require several GiB of RAM, so use `--max-samples` for resource-limited runs. `scripts.predict_test` accepts `--max-batches` for partial smoke output and prints the actual row count. See the architecture document for the decoder-causality correction and artifact compatibility notes.
