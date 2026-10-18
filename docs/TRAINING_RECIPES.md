# Training Recipes and Reference Differences

See [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for the paper-to-code map and Table 2 comparison. Public defaults are a portable training recipe, not an exact-paper reproduction preset.

## Public Defaults

| Setting | RoboDojo | Real robot |
|---|---|---|
| Wan model | Wan2.2-TI2V-5B | same |
| DiT training / LoRA | full train / off | same |
| Observation encoder | frozen | same |
| Local batch / accumulation | 2 / 1 | 2 / 1 |
| Action/video horizon | 16 / 16 | 16 / 16 |
| Input | T-shaped RGB 384×320, state 14D, language | head RGB 224×224, state 28D |
| Epochs | 50 | 150 |
| Learning rate | 2e-4 | 2e-4 |
| Optimizer-step warmup | 500 | 500 |
| Rehearsal / state / credit warmup | 18000 | 6000 |
| Validation split | 5% of episodes | exactly zero |
| Checkpoint | each epoch, latest + top 3 train-loss | same |
| Latent cache | off | off |

The full public configuration is the authority; inspect it without constructing a model:

```bash
python -m rap_wam.train --config-name robodojo --cfg job --resolve
python -m rap_wam.train --config-name real_robot --cfg job --resolve
```

The inherited module warmup counts preserve the implementation's own counters. They are not presented as an exactly matched sample budget across different world sizes/batches.

## Why the Default Batch Is 2

The DAOPSR implementation computes `(difficulty-min)/(max-min+epsilon)` **within each local minibatch**. At batch 1 this becomes zero; scheduled state exposure remains, but difficulty adaptation has no across-sample variation. Accumulating multiple batch-1 gradients does not alter this fact. `BATCH_SIZE=1` is available for historical comparisons, and the launcher prints a note when selected.

Historical RoboDojo reference: 12 nodes × 8 GPUs × local batch 1, 50 epochs, 14D state/action, per-task language. It used consolidated VAE caches, an asynchronous checkpoint workspace, a different checkpoint selection schedule and a nominal validation ratio 0 that still held out one episode in its original loader. This public recipe uses online VAE encoding, synchronous checkpointing and a 5% episode holdout. It preserves the model/loss implementation, not the original wall time or identical training sequence/checkpoint choice. No historical benchmark success rate is attributed to the new recipe without rerunning it.

Historical real-robot fixed100 reference: 15 nodes × 8 GPUs × local batch 1, accumulation 1, global batch 120, 150 epochs. The phone-packing data card recorded 100 episodes / 69,796 frames / 66,596 two-chunk windows. This package preserves the strict 100-episode, zero-validation loader, but accepts other frame counts because private demonstrations are not shipped. Batch 1 and the original distributed resource layout can be selected explicitly.

## Multi-Node Invocation

Example for two nodes, eight GPUs each. On every node, install the same source/dependencies and mount the same data, model and output locations. Set node zero's reachable address; port 29500 must be available. The launcher uses ordinary PyTorch DDP, not DeepSpeed/FSDP.

```bash
export NNODES=2 NPROC_PER_NODE=8 BATCH_SIZE=2 GRAD_ACCUM_STEPS=1
export MASTER_ADDR="node-zero.example.org" MASTER_PORT=29500
export RUN_DIR=/shared/outputs/robodojo_run_001
export DATA_ROOT=/shared/data/robodojo/zarr
export WAN_CHECKPOINT_DIR=/shared/assets/Wan2.2-TI2V-5B
export TASK_CONTEXT_PATH=/shared/data/robodojo/task_context.pt
# Node 0:
NODE_RANK=0 bash scripts/train_robodojo.sh
# Node 1, in its own terminal:
NODE_RANK=1 bash scripts/train_robodojo.sh
```

This example's global batch is `2 × 8 × 2 × 1 = 32`. Only rank zero may claim a fresh output directory. Reusing an old output directory is rejected by the launcher. Cloud-specific task creation and device allocation are intentionally left to the hosting platform.

`training.optimizer_warmup_steps` expresses optimizer updates. The packaged Hydra resolver multiplies it by `WORLD_SIZE` to compensate for Accelerate 1.2.0's prepared scheduler stepping once per process. Other scheduler behavior is inherited. Keep the dependency pins when interpreting this field.

Override examples:

```bash
NPROC_PER_NODE=8 bash scripts/train_real_robot.sh \
  training.num_epochs=150 optimizer.lr=2e-4 training.optimizer_warmup_steps=500
NPROC_PER_NODE=1 BATCH_SIZE=1 bash scripts/train_robodojo.sh --dry-run
```

`training.max_train_steps` in the inherited workspace limits batches **per epoch**; it is not a total optimizer-step budget. Use it only with an explicitly small epoch count for smoke testing. The release does not promise bit-exact mid-epoch resume, nor simulator/physical-robot evaluation.

Starting with 0.2.0, normalization statistics use only complete episodes represented in the training split, excluding validation episodes. Earlier releases scanned every shard when fitting statistics despite their documentation. New runs therefore differ from those earlier releases. Resumed checkpoints retain their saved normalization rather than silently refitting it. The actual limited dataloader length is used for accumulation flushes and scheduler budgeting, including a final partial accumulation group.
