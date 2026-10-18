# RAP-WAM Examples

These examples make file and launch interfaces concrete. They do not provide demonstrations, trained policies, hardware acquisition or a minimum-memory training configuration.

## Real-Robot NPZ Export

Import `export_episode` from [export_real_robot.py](export_real_robot.py) and pass synchronized RGB plus every named joint array in `DIMS`. The helper checks dimensions/finite values and writes without overwriting an existing file. It performs no coordinate or timing conversion.

The executable example deliberately generates **synthetic zero-valued data**:

```bash
python examples/export_real_robot.py --synthetic \
  --out data/schema_example/episode_npz/episode_000000.npz
python -m rap_wam.prepare_data real_robot \
  --source data/schema_example/episode_npz --out data/schema_example/zarr \
  --expected-episodes 1
python -m rap_wam.preflight real_robot \
  --data-root data/schema_example/zarr --expected-episodes 1 --full
```

A 40-frame synthetic episode yields eight two-chunk windows. This checks conversion and schema only; do not train a meaningful policy from it. The fixed100 training adapter still expects 100 episodes by default.

## Environment Templates

From the repository root:

```bash
source examples/robodojo.env.example
bash scripts/train_robodojo.sh --dry-run
```

Use [real_robot.env.example](real_robot.env.example) with the other training script. Edit paths/resources and choose a new run directory for each actual job. A single-process template is illustrative; full model memory capacity must be established on the intended hardware. Multi-node configuration is described in [training recipes](../docs/TRAINING_RECIPES.md#multi-node-invocation).
