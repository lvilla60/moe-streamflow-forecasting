# Architecture

## Data contract

Each historical input has shape `[B, 336, 12]`. Channel 11 is historical specific discharge; channels 0–10 are meteorological or hydrometeorological variables. Experts return a normalized target forecast with shape `[B, 48]`. Training and validation use the official HDF5 split, while `test.h5` has no target fields.

`y_aux` has shape `[B, 48, 11]` but is omitted from the initial architecture because these future variables are unavailable during test inference.

## Normalization

All normalization values come from `outputs/train_stats.json`, which was computed from the training split only. Each input channel uses its own mean and population standard deviation. The discharge target uses one scalar mean and standard deviation.

Experts optimize MSE in normalized target space. Predictions are transformed back to physical specific-discharge units before MAE, RMSE, and NSE are calculated. The Seq2Seq decoder starts from the raw last historical discharge transformed with the target statistics:

```text
(X[:, -1, 11] - y_mean) / y_std
```

This is distinct from normalizing input channel 11 with its input-channel statistics.

## Expert contract

All experts accept normalized `x: [B, T, 12]` and return `[B, H]`, with `T=336` and `H=48` in real runs. Horizon and model sizes are configurable for lightweight tests.

- **LSTM:** an LSTM encoder followed by a small direct 48-step projection.
- **GRU:** the corresponding GRU encoder and direct projection.
- **LSTM Seq2Seq with Attention:** an LSTM encoder, additive attention over every encoder state, and an autoregressive LSTM decoder. It supports teacher forcing during training and requires no future targets during inference.
- **Informer:** input projection, sinusoidal relative positional encoding, simplified ProbSparse query selection, encoder distillation, a causally masked decoder, cross attention, and one-pass 48-step output projection.

Informer decoder input consists of the final `label_len` normalized historical input steps followed by 48 zero placeholders. Future meteorological values are not assumed.

The ProbSparse encoder is an approximation: it estimates query sparsity using a deterministic, evenly spaced subset of keys, evaluates the highest-scoring queries against all keys, and uses a mean-value context for other queries. The decoder uses full causal self-attention and full cross-attention. Global top-query selection in the decoder would allow later positions to change earlier outputs even with masked attention scores. This correction preserves parameter shapes but changes Informer predictions: retrain Informer and regenerate dependent labels/routers when moving from pre-audit artifacts to full experiments.

## Expert training and checkpoints

Experts are trained independently with AdamW and normalized-target MSE. Gradient clipping, teacher forcing, device, batch limits, and optimizer settings are configurable. Validation disables gradient computation and teacher forcing. Physical-unit metrics are accumulated globally without retaining the complete validation prediction array.

An expert checkpoint contains:

- model name and model configuration
- model and optimizer states
- completed epoch and best validation loss
- normalization values
- train and validation loss history

Checkpoint loading supports reconstruction and optimizer restoration for resumed training.

Resume requires the same saved expert name, complete model configuration, and normalization; optimizer settings are restored from the checkpoint. `--epochs` is the total epoch count, not an additional count. When resuming to a new checkpoint directory, the previous best checkpoint is preserved even if later epochs do not improve. Checkpoints represent the best epoch, not necessarily the last completed epoch. Random-generator/DataLoader state is not stored, so resumed runs are not bitwise continuations.

Input channels with zero standard deviation use a divisor of one. Target standard deviation must be positive; nonfinite statistics are rejected. Statistics JSON files must declare the training split. MoE inference rejects missing or incompatible router normalization metadata.

## Router-label generation

After all four experts are trained and frozen, each known-target sample is evaluated by every expert. For sample `i` and expert `e`:

```text
error(i, e) = mean(abs(y_true[i, :] - y_pred[e, i, :]))
best_expert(i) = argmin_e error(i, e)
```

The class order is fixed:

```text
0 = LSTM
1 = GRU
2 = LSTM-S2S-Attention
3 = Informer
```

Ties select the first class. Generated label artifacts store `Id`, `basin_id`, `best_expert`, and four expert errors; historical arrays are referenced by row and are not duplicated. Router training labels must use only the official training split.

`Id` is the original zero-based HDF5 row, not an index within a filtered split. Label readers validate the class order, unique integer IDs, and class bounds, and check referenced rows against the official split before router training/evaluation. Default filenames distinguish train and validation labels. Artifacts do not fingerprint HDF5 contents or exact expert weights; keep the dataset and expert versions together and regenerate labels after changing experts. Training labels are generated in-sample from trained experts, which can bias expert preferences; held-out validation remains essential.

## Routers

- **Random Forest:** scikit-learn classifier over the flattened normalized history `[B, 336 * 12]`.
- **LSTM router:** LSTM sequence encoder with a four-class linear head.
- **Transformer router:** projected inputs, sinusoidal positions, standard Transformer encoder, mean pooling, and a four-class head.

Random Forest flattening is an initial adaptation and can be memory intensive, so its training CLI supports a sample cap. Router inference uses historical inputs only and never derives classes from future targets.

For 254,000 histories, the float32 RF matrix contains 1,024,128,000 values (4.10 GB / 3.82 GiB) before forest and worker allocations. Feature preparation now allocates one matrix and normalizes small chunks in place, prints a memory estimate, and warns on large allocations. Use `--max-samples` and limit `--n-jobs` to fit the available RAM; this is still in-memory RF training. `predict_proba` always uses columns 0–3, including zero columns for unseen classes.

Neural routers supplied with `--validation-labels` calculate validation cross-entropy, accuracy, and a confusion matrix every epoch under `no_grad`. The saved checkpoint is selected by the lowest validation loss. With no validation labels, the final model is saved with `selection=final_epoch` and `best_validation_loss=None`; training loss is never reported as validation loss.

## Hard MoE inference

The initial MoE uses one router class per sample. All four experts may be run for reproducibility, producing `[B, 4, 48]`; the selected class gathers one complete forecast and returns `[B, 48]`. Soft routing and lead-time routing are outside the initial implementation.

Validation calculates physical-unit MAE, RMSE, and NSE and records router class frequencies. If validation best-expert labels are supplied, router accuracy and a confusion matrix are also reported. Test inference writes `Id` and `q_01` through `q_48` without reading `test_targets.csv`.

Partial validation label files are supported: forecast metrics cover all evaluated rows, classification metrics cover the intersection, and matched/unmatched sample counts are reported. Router accuracy is `null` if there are no matching labels. Test prediction supports `--max-batches N` for debugging, warns that the CSV may be partial, and reports the actual row count. Without the flag it processes the complete test split. Use a separate smoke-output filename to avoid confusing a partial file with a submission.

## Methodological adaptations

1. The paper's original dataset and task differ from this laboratory dataset; this project forecasts 48 hourly discharge values from 336 historical hours.
2. The official provided train/validation split is retained.
3. `y_aux` is omitted because it is unavailable during test inference.
4. Routing uses one best expert per 48-hour sample based on mean absolute error over the horizon. Horizon-level routing can be added later but is not implemented now.
5. Informer calendar embeddings are replaced by relative sinusoidal positions because explicit calendar timestamps are unavailable.
6. The Random Forest initially receives a flattened normalized historical sequence.
7. Experts and routers are trained sequentially rather than jointly end to end.

## Portability

Project modules are run from the repository root with `python -m ...`; no import-path modifications are used. The implementation targets Python 3.10 with CPU PyTorch locally and Python 3.11 with PyTorch 2.11 and CUDA 12.8 on Khipu. Devices are selected by configuration, and no CUDA-only code path is required. Seeding covers Python, NumPy, torch, and available CUDA devices as a best-effort reproducibility measure without forcing unsupported deterministic kernels.
