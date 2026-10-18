# Configuration Reference

[Documentation](README.md) / Configuration

Hydra combines `rap_wam/configs/base.yaml` with either `robodojo.yaml` or `real_robot.yaml`. The resolved configuration is authoritative. Set paths before rendering it:

```bash
export WAN_CHECKPOINT_DIR="$PWD/assets/Wan2.2-TI2V-5B"
export DATA_ROOT="$PWD/data/robodojo/zarr"
export TASK_CONTEXT_PATH="$PWD/data/robodojo/task_context.pt"
python -m rap_wam.train --config-name robodojo --cfg job --resolve
```

Hydra arguments following a training script override configuration values. Do not append `--dry-run` to `rap_wam.train`; it belongs to the launcher/scripts.

## Paths and Distributed Resources

| Environment variable | Default / meaning |
|---|---|
| `WAN_CHECKPOINT_DIR` | `./assets/Wan2.2-TI2V-5B`; original transformer/VAE directory |
| `DATA_ROOT` | Profile-specific `./data/.../zarr`; directory containing Zarr shards |
| `TASK_CONTEXT_PATH` | `./data/robodojo/task_context.pt`; RoboDojo only |
| `RUN_DIR` | Fresh timestamped `outputs/<profile>/...` if unset by the launcher |
| `PYTHON_BIN` | `python`; shell entrypoint interpreter |
| `BATCH_SIZE` | 2 per GPU; launcher emits `train_dataloader.batch_size` |
| `GRAD_ACCUM_STEPS` | 1; emits `training.gradient_accumulate_every` |
| `NPROC_PER_NODE` / `NNODES` | 1 / 1 |
| `NODE_RANK` | 0; distinct rank in `[0, NNODES)` on each node |
| `MASTER_ADDR` / `MASTER_PORT` | Required address for multi-node; port defaults to 29500 |

Explicit Hydra batch/accumulation arguments take precedence over environment defaults in the launcher summary. `RUN_DIR` must be new on rank zero, including continuation attempts. Multi-node jobs need a common location visible to all nodes. The launcher supports ordinary DDP, and rejects enabled DeepSpeed/FSDP environment flags.

## Optimization and Data

| Field | RoboDojo | Real robot | Interpretation |
|---|---:|---:|---|
| `train_dataloader.batch_size` | 2 | 2 | Local minibatch; affects difficulty normalization |
| `training.gradient_accumulate_every` | 1 | 1 | Microbatches per optimizer update |
| `training.num_epochs` | 50 | 150 | Total epoch limit, including completed epochs after resume |
| `training.preflight_full` | true | true | Scan all low-dimensional data values before model allocation |
| `optimizer.lr` | 2e-4 | 2e-4 | Peak AdamW LR |
| `optimizer.weight_decay` | 1e-4 | 1e-4 | AdamW weight decay |
| `optimizer.betas` | `[0.9,0.95]` | Same | AdamW coefficients |
| `training.optimizer_warmup_steps` | 500 | 500 | Warmup expressed in optimizer updates |
| `training.grad_norm_clip` | 0.5 | 0.5 | Gradient norm clipping |
| `training.mixed_precision` | `bf16` | `bf16` | Accelerate precision mode |
| `optimizer.dit_mode` | `full_train` | Same | Entry point requires full DiT training |
| `optimizer.obs_encoder_mode` | `freeze` | Same | Frozen VAE observation encoder |
| `training.use_ema` | false | false | Parameter EMA, distinct from spatial prior EMA |
| `task.dataset.val_split_ratio` | 0.05 | 0.0 | Episode-level holdout; fixed100 explicitly supports empty validation |
| `training.val_every` | 1 | 0 | Epoch interval for validation |
| `training.max_train_steps` | null | null | Minibatch limit **per epoch**, not total optimizer steps |

With pinned Accelerate 1.2.0, `training.lr_warmup_steps` is computed as `optimizer_warmup_steps × WORLD_SIZE` to compensate for the prepared scheduler's process-dependent stepping. Set the public optimizer-warmup field instead of manually multiplying it a second time.

```bash
BATCH_SIZE=2 GRAD_ACCUM_STEPS=2 NPROC_PER_NODE=8 \
  bash scripts/train_robodojo.sh optimizer.lr=1e-4 training.num_epochs=20
```

This example has global batch 32 on one node. It is a configuration example, not the paper's recipe. Accumulation changes auxiliary calls per optimizer update because their counters advance on local minibatches.

## Predictive Interface and CAP

| Policy field | Default | Meaning |
|---|---:|---|
| `num_latent_queries` | 16 | Learned future-supervised query slots |
| `align_dim` / `align_loss_weight` | 128 / 0.1 | Alignment projection width and loss coefficient |
| `target_temporal_pool` | 1 | Pool recorded future over time before a spatial 4 × 4 target grid |
| `decouple_action_from_video` | true | Suppress action-query rows reading video keys |
| `decouple_action_attends_state` | true | Keep state reads; indirect visual paths remain |
| `coarse_head_hidden_mult` / `coarse_zero_init` | 0.25 / true | CAP compact MLP width and zero output initialization |
| `num_inference_steps` | 16 | Ordinary sampling default; rehearsal overrides its own step count |

CAP branches predict the same velocity target through their sum. `coarse_*` and `residual_*` are implementation parameter names, not separately supervised physical action components. Model tensor names are kept stable.

## Rehearsal and Scheduled State

| Policy field | RoboDojo | Real robot | Meaning |
|---|---:|---:|---|
| `ccrt_lambda_max` | 1.0 | 1.0 | Maximum next-chunk flow-loss coefficient |
| `ccrt_p_max` | 0.25 | 0.25 | Maximum probability of a rehearsal call |
| `ccrt_warmup_steps` | 18000 | 6000 | Local calls for linear probability/coefficient ramp |
| `ccrt_sample_steps` | 4 | 4 | Gradient-free first-chunk denoising steps |
| `opsr_p_max` | 0.5 | 0.5 | Maximum base probability of predicted-state substitution |
| `opsr_warmup_steps` | 18000 | 6000 | Substitution schedule warmup |
| `opsr_difficulty_gamma` | 1.0 | 1.0 | Optional correction from local-batch normalized last-action error |

At local batch 1, normalized difficulty is zero. Scheduled substitution remains, while this correction contributes no across-sample variation. A predicted state requires matching absolute-joint state/action layouts; this is not simulator interaction or expert relabeling.

## Response Credit and Spatial Prior

| Policy field | Default | Paper role |
|---|---:|---|
| `puga_weight` / `puga_act_weight` | 0.5 / 0.3 | Extra normalized weighted video/action loss coefficients |
| `puga_p` | 0.5 | Auxiliary probability |
| `puga_warmup_steps` | 18000 / 6000 | Auxiliary coefficient ramp, RoboDojo / real robot |
| `puga_w_unc` / `puga_w_lev` / `puga_w_diff` | 1 / 1 / 0.5 | Prior P / response S / current error E coefficients |
| `puga_act_delta` | 0.1 | Gaussian perturbation std on normalized noisy actions |
| `puga_n_samples` | 1 | Must be 1; other values are rejected at training entry |
| `uprime_grid` / `uprime_ema_decay` | 14 / 0.9 | Persistent rehearsal-error prior grid and EMA decay |
| `uprime_use_gap` | false | Retain the paper's spatial-prior construction |
| `latent_cache` | false | Public recipes encode online; no offline builder is shipped |
| `strict_auxiliary_errors` | true | Raise auxiliary exceptions with the original cause |

Internal `puga_*` and `uprime_*` identifiers are mapped to paper terminology in [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md). They do not introduce separate policies or a scoring network. The internal normalization floor is 1e-6. Every visual/action location retains a positive loss coefficient.

## Logging and Saving

`logging.mode=disabled` disables the optional remote tracker; it does not disable local JSON logs. Checkpoint selection uses `checkpoint.topk.monitor_key=train_loss`, minimum mode and k=3, plus `latest.ckpt`. `training.checkpoint_every=1` saves each epoch. Sampling dumps are disabled by default.

See [checkpoint contents](CHECKPOINTS.md) and [auxiliary diagnostics](PAPER_ALIGNMENT.md#auxiliary-participation-and-failures). Changes to split, batch, optimizer timing or branch schedules should be recorded with the resulting checkpoints.

The launcher reserves a new output directory atomically and rejects `hydra.run.dir` overrides; set `RUN_DIR` so the saved launch summary and actual Hydra output agree. Dry-run composes the selected profile and checks supported options without reading data or allocating model tensors. Sample-based top-k selection requires sampling to be enabled; selection is deferred until that metric exists.

Device allocation follows `CUDA_VISIBLE_DEVICES` and torchrun local rank; `training.device` is retained only for the optional prompt-preparation path and does not select a training GPU. The entry rejects CPU execution with the production Wan encoder before model allocation. Deprecated nonempty optimizer start paths are rejected; use the explicit training resume fields.
