# RoboDojo Training Walkthrough

[Documentation](README.md) / RoboDojo

This workflow trains the language-conditioned 14D ARX X5 policy from episode demonstrations. Complete [installation](INSTALLATION.md) and set `WAN_CHECKPOINT_DIR` before generating text embeddings or constructing the model.

## 1. Check the Source Demonstrations

Obtain data through the [official RoboDojo download guide](https://robodojo-benchmark.com/doc/usage/install-and-download/) or its [dataset repository](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo). Select the matching ARX X5 demonstration export. The supported source is **LeRobot v2.1 episode-video format**, not an arbitrary newer LeRobot export. Each episode needs:

- One parquet file containing 14D `action` and `observation.state` rows.
- Head, left-wrist and right-wrist RGB videos with matching frame counts.
- An entry in `meta/episodes.jsonl` containing exactly one known instruction string.
- At least 33 synchronized frames, so a current observation and two 16-step chunks fit.

See the [exact source layout and key names](DATA_FORMAT.md#robodojo-input). No download credentials, private storage mounts or dataset files are embedded in the repository. Obtain the demonstrations through their original distribution and applicable terms.

## 2. Convert to Zarr

Run from the repository root. Conversion refuses an existing output directory and validates the staged output before publishing it.

```bash
python -m rap_wam.prepare_data robodojo \
  --source /path/to/robodojo_lerobot_v21 \
  --out data/robodojo/zarr \
  --episodes-per-shard 50
```

The three camera frames become one T-shaped RGB image: head 320 × 256 above two 160 × 128 wrists. The result has height 384, width 320. The command preserves 14D joint state/actions and writes one task index per episode.

```text
data/robodojo/zarr/
├── ... .zarr/              # data arrays and episode/task metadata
├── task_strings.json      # ordered vocabulary used for this conversion
└── manifest.json
```

Use `--limit 10` with a separate new output directory for a deliberate small conversion check. Frame-count mismatches fail by default. `--truncate-to-shortest` permits explicit shortest-stream truncation; it does not synchronize timestamps or correct dropped frames.

## 3. Prepare the Ordered Language Context

The packaged vocabulary contains 35 archived instruction strings. Conversion matches strings exactly, without fuzzy matching or a fallback to task zero. Always encode the vocabulary emitted beside the converted shards:

```bash
python -m rap_wam.task_context \
  --tasks data/robodojo/zarr/task_strings.json \
  --wan-checkpoint-dir "$WAN_CHECKPOINT_DIR" \
  --out data/robodojo/task_context.pt
```

This step uses the UMT5 encoder on `cuda:0` by default. Select another device with `--device cuda:1`, or provide a different local tokenizer with `--tokenizer /path/to/umt5_tokenizer`. It refuses to overwrite an existing context file. Each context row is padded to `[512,4096]`; the saved file contains both the ordered strings and `[num_tasks,512,4096]` embedding table.

For a custom vocabulary, provide a JSON list to conversion via `--tasks /path/to/task_strings.json`, generate a fresh matching table, then set `task.dataset.num_tasks` to its exact length at training time. Do not reorder an existing vocabulary or reuse its table after changing labels. Training validates table shape, task order and finite values.

## 4. Validate and Inspect

```bash
python -m rap_wam.preflight robodojo \
  --data-root data/robodojo/zarr --num-tasks 35 --full \
  --task-context data/robodojo/task_context.pt \
  --wan-checkpoint-dir "$WAN_CHECKPOINT_DIR"

export DATA_ROOT="$PWD/data/robodojo/zarr"
export TASK_CONTEXT_PATH="$PWD/data/robodojo/task_context.pt"
python -m rap_wam.train --config-name robodojo --cfg job --resolve
NPROC_PER_NODE=1 bash scripts/train_robodojo.sh --dry-run
```

Standalone preflight scans **data**, reporting shards, episodes, frames and valid two-chunk windows before the split. With `--task-context` and `--wan-checkpoint-dir`, standalone preflight also checks context order and Wan asset files. Training performs these checks and scans all low-dimensional values by default before constructing the 5B policy. A successful `--dry-run` validates launch arguments, Hydra overrides and supported training modes; it does not read model tensors, check memory capacity or initialize DDP communication.

## 5. Launch a Training Job

```bash
export RUN_DIR="$PWD/outputs/robodojo/run_001"
BATCH_SIZE=2 GRAD_ACCUM_STEPS=1 NPROC_PER_NODE=8 \
  bash scripts/train_robodojo.sh
```

Set the GPU count and batch to your actual allocation. Defaults are 50 epochs, learning rate 2e-4, optimizer warmup 500 updates and auxiliary warmups 18,000 local training calls. The public recipe holds out 5% of episodes, with at least one held-out episode when more than one exists. Split membership is deterministic for the supplied episode order and seed; windows never cross episode boundaries.

For a deliberately short **GPU integration probe**, use a fresh directory and explicitly enable the expensive branches from the first update:

```bash
export RUN_DIR="$PWD/outputs/robodojo/integration_probe_001"
NPROC_PER_NODE=1 bash scripts/train_robodojo.sh \
  training.num_epochs=1 training.max_train_steps=2 \
  policy.ccrt_warmup_steps=0 policy.ccrt_p_max=1.0 \
  policy.puga_warmup_steps=0 policy.puga_p=1.0
```

This requires enough memory for the real model and is an integration probe, not the production training schedule or a convergence experiment. `training.max_train_steps` limits minibatches **per epoch**. Restore the default schedules for a research training run.

## What to Retain

Keep run configuration/logs/checkpoints, conversion report, source data identity, ordered vocabulary, task-context table, dependency versions, seed and resource layout. The embedding table is not persistent checkpoint state. See [checkpoints](CHECKPOINTS.md) and [reproducibility](REPRODUCIBILITY.md).

The training vocabulary's 35 labels are not the benchmark's 42 evaluation task units. No simulator evaluation or task-policy repertoire is launched by this workflow.

## Explicit Language Labels During Sampling

The language-conditioned model requires `task_index` as an integer tensor of shape `[B]` in training batches and top-level observation dictionaries passed to `predict_action` or `predict_visual`. Indices refer to the retained vocabulary order. Missing labels are errors; the policy does not silently sample with empty text. Rehearsal inherits the active training batch labels, and optional workspace sampling forwards them explicitly. This is the model interface; simulator/hardware serving is still outside this release.
