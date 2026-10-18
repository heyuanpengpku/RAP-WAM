# Checkpoints and Run Outputs

[Documentation](README.md) / Checkpoints

## A Training Run on Disk

The launcher chooses a fresh timestamped `RUN_DIR`, or uses the value you supply. The ordinary epoch-checkpoint configuration produces:

```text
outputs/robodojo/run_001/
├── .hydra/                  # Hydra configuration and overrides
├── launch.json              # Launch resources, batch accounting and run directory
├── logs.json.txt            # JSON training/validation metric records
└── checkpoints/
    ├── latest.ckpt
    └── step=...-loss=....ckpt
```

Additional logs or optional visual metrics may appear when explicitly enabled. The defaults save `latest.ckpt` each epoch plus up to three selected checkpoints ranked by **minimum training loss**. File names are not robot-success measurements. The fixed100 real-robot profile has no validation examples.

## What Is Saved

The checkpoint payload has `cfg`, `state_dicts` and `pickles` sections. `state_dicts` includes the model and optimizer, with normalization state carried by the model. The workspace also stores selected progress fields such as epoch and global step. This is a Python/PyTorch checkpoint using dill, not a standalone tensor-only model export.

The learned parameters and the spatial rehearsal-error prior are part of model state. The RoboDojo language table is a **non-persistent** buffer, and is not embedded in it. New checkpoints also save auxiliary schedule counters, prior readiness and process-local participation counts in model extra state. Random-number state and dataloader position are not part of an exact continuation protocol. Legacy checkpoints lack some of these fields and cannot reconstruct their historical schedule counters.

Keep these companion artifacts:

| Artifact | Why it matters |
|---|---|
| Source release identity and dependency versions | Resolve Hydra targets and reproduce implementation behavior |
| Resolved configuration and launch summary | Architecture, optimization, resource layout and overrides |
| Wan checkpoint identity | Original backbone/VAE assets needed by the constructor |
| Dataset identity and conversion `manifest.json` | Frame, episode and window accounting |
| Joint ordering, units and source conventions | Correct interpretation of normalized state/actions |
| RoboDojo `task_strings.json` and `task_context.pt` | Exact label-to-embedding mapping |
| Train/validation split seed and source ordering | Reconstruct episode membership |

Treat checkpoints as trusted Python artifacts when inspecting/loading them; dill-based loading can execute Python objects. The metadata and weight files are not a substitute for a deployment adapter.

## Inspect Metric Records

Run logs contain JSON records; useful fields include `global_step`, `epoch`, `train_loss`, `lr`, optional `val_loss`, and policy loss components. For example:

```python
import json
from pathlib import Path

records = [json.loads(line) for line in
           Path("outputs/robodojo/run_001/logs.json.txt").read_text().splitlines()
           if line.strip()]
latest = records[-1]
print({key: latest[key] for key in ("global_step", "epoch", "train_loss")
       if key in latest})
```

`train_loss` is the gathered loss reported by the workspace. Per-update auxiliary fields come from local policy diagnostics. With accumulation, the logged component can be the last microbatch rather than an average over every microbatch; cumulative counters expose a separate process-local view. They are not aggregated DDP-wide participation statistics.

A sampled but failed/empty auxiliary is different from a skipped probability gate. The [paper guide](PAPER_ALIGNMENT.md#auxiliary-participation-and-failures) defines attempted/succeeded/applied/failed counters. A small number of early updates can legitimately have no auxiliary contribution while warmup/probability schedules are low.

## Continue from a Saved Checkpoint

The training entry supports an explicit resume path. Supply compatible data, model assets and the same profile/architecture; use a new output directory because the launcher rejects any existing one.

```bash
export RUN_DIR="$PWD/outputs/robodojo/continued_001"
NPROC_PER_NODE=1 bash scripts/train_robodojo.sh \
  training.resume=true \
  training.resume_path=/path/to/prior_run/checkpoints/latest.ckpt
```

This uses the workspace's checkpoint loader. It is **not** a promise of bit-exact mid-epoch continuation: dataloader position and randomness may differ. New epoch checkpoints store the next epoch to run and retain auxiliary schedules/prior readiness; older checkpoints may replay their last epoch and restart missing schedule state. The epoch budget is the configured total epoch limit, not a count of additional epochs. Resuming without an existing explicit checkpoint path is rejected by the training entry. Loading preserves the current output directory, including when reading a legacy checkpoint that stored an old directory. Verify actual saved progress and the resolved config before treating a continuation as part of a controlled comparison.

If you change architecture, vocabulary, action dimension or normalization schema, resolve the incompatibility explicitly instead of suppressing state-dictionary errors. Historical checkpoints with private module targets require a deliberate configuration migration; this release ships only the public `rap_wam` namespace.

## Evaluation and Export

Model classes provide `predict_action` and joint visual sampling methods, but this package does not ship a full simulator/hardware runner or a tensor-only export command. The paper's evaluation smoothing/replanning protocol belongs to that separate integration. Record the actual integration and sampling configuration with any new results rather than assigning the paper's reported scores to a checkpoint filename.

Checkpoint writes use a temporary file and atomic replacement so a failed save does not truncate the previous file. The new checkpoint format stores `format_version=2`. An epoch-end checkpoint saves the next epoch; a step checkpoint can restart the current epoch and does not promise restoration of the exact dataloader offset. Model extra state is backward compatible with prior state dictionaries, but missing legacy counters cannot be reconstructed.
