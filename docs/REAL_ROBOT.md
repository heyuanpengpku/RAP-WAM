# Real-Robot Training Walkthrough

[Documentation](README.md) / Real robot

This workflow covers one physical-robot interface: head RGB at 224 × 224, a 28D absolute-joint state/action layout, and 100 demonstrations of the phone-packing task. The model/loss core is shared with RoboDojo, without language conditioning. Hardware collection, control drivers, timing alignment and the paper's separate 32D interface are outside this release.

## 1. Export Synchronized Demonstrations

Create one `.npz` per episode. All arrays in an episode must share frame count T; T must be at least 33. RGB is uint8 `[T,224,224,3]`. State and action arrays are finite float32, with names/dimensions fixed by [DATA_FORMAT.md](DATA_FORMAT.md#real-robot-input).

| Concatenation order | State fields | Action fields |
|---|---|---|
| Waist, 2D | `waist_joints_states` | `action_waist_joints` |
| Left arm, 7D | `left_arm_states` | First seven columns of `action_arm_joints` |
| Right arm, 7D | `right_arm_states` | Last seven columns of `action_arm_joints` |
| Left hand, 6D | `left_hand_states` | `action_left_hand_joints` |
| Right hand, 6D | `right_hand_states` | `action_right_hand_joints` |

Actions must be absolute joint positions in a schema compatible with state, because rehearsal can substitute the final predicted action as the next state. Do not put Cartesian deltas, velocity commands or differently ordered joints into these fields. Preserve the source waist convention and units. The loader cannot infer or calibrate them.

[examples/export_real_robot.py](../examples/export_real_robot.py) provides an importable named-array export function and a synthetic example. Its generated sample is for verifying file structure only; it is not training data or a hardware recorder.

## 2. Convert the 100 Episodes

Place the episode files directly under a source directory; lexical filename order is used. Zero-padded names make that order explicit.

```text
episode_npz/
├── episode_000000.npz
├── episode_000001.npz
└── ... episode_000099.npz
```

```bash
python -m rap_wam.prepare_data real_robot \
  --source /path/to/episode_npz \
  --out data/real_robot/zarr \
  --expected-episodes 100 --episodes-per-shard 50
python -m rap_wam.preflight real_robot \
  --data-root data/real_robot/zarr --expected-episodes 100 --full
```

Conversion refuses an existing output directory. Invalid/nonfinite values, short episodes or inconsistent counts cause failure before the staged dataset is published. Existing Zarr with the same contract can bypass conversion and proceed through preflight.

## 3. Inspect the Configuration and Launch

```bash
export WAN_CHECKPOINT_DIR="$PWD/assets/Wan2.2-TI2V-5B"
export DATA_ROOT="$PWD/data/real_robot/zarr"
export RUN_DIR="$PWD/outputs/real_robot/run_001"

python -m rap_wam.train --config-name real_robot --cfg job --resolve
NPROC_PER_NODE=1 bash scripts/train_real_robot.sh --dry-run
NPROC_PER_NODE=1 bash scripts/train_real_robot.sh
```

Choose resources to fit full DiT training and auxiliary branches. Public defaults are 150 epochs, learning rate 2e-4, local batch 2, accumulation 1 and auxiliary warmups 6,000 local calls. All 100 demonstrations train the policy; validation is intentionally empty, and checkpoint selection uses training loss. No validation success or robot performance is inferred from that loss.

## 4. Using a Different Demonstration Count

Change the conversion/preflight expectation and the training adapter expectation together. This example uses a separate 20-episode dataset:

```bash
python -m rap_wam.prepare_data real_robot \
  --source /path/to/twenty_episode_npz --out data/real_robot_20/zarr \
  --expected-episodes 20
python -m rap_wam.preflight real_robot \
  --data-root data/real_robot_20/zarr --expected-episodes 20 --full

DATA_ROOT="$PWD/data/real_robot_20/zarr" NPROC_PER_NODE=1 \
  bash scripts/train_real_robot.sh task.dataset.expected_episode_count=20
```

It is no longer the fixed100 recipe. Keep the 28D ordering/image contracts and record your change. Changing only the expected episode count does not generalize the loader to other robots, dimensions or command spaces.

## 5. From Training to Deployment

The policy contains joint sampling methods, but this release does not provide a hardware-serving or closed-loop controller. Deployment needs a compatible observation-to-tensor adapter, normalization, correct checkpoint/configuration, joint mapping/units, action timing, replanning and platform integration. Save the normalization state with the model; it is part of the trained policy interface.

Use [CHECKPOINTS.md](CHECKPOINTS.md) for artifact handling and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the boundary between this workflow and the paper's physical-robot results.
