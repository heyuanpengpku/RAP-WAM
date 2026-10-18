# Data Contracts

## Shared Training Windows

Training reads `<DATA_ROOT>/*.zarr`. Each shard contains concatenated frame arrays under `data/` and strictly increasing, exclusive episode ends under `meta/episode_ends` (int64). Every episode must contain at least **33 frames**. One sample uses a current observation plus two consecutive 16-step futures; there are `episode_length - 32` valid windows before the split. Windows never cross episode boundaries.

An item exposes `obs`, `action`, `obs2`, `action_2`, and a scalar `task_index` for RoboDojo. Action tensors are `[16, action_dim]`; the loader uses actions at offsets 1..16 and 17..32. State tensors contain the current/second-chunk states. Training fits normalization statistics from training episodes.

## RoboDojo Input

Use LeRobot-v2.1 episode-video layout:

```text
meta/info.json                         # chunks_size, normally 1000
meta/episodes.jsonl                    # episode_index and one task string in tasks

data/chunk-000/episode_000000.parquet  # action [14], observation.state [14] per row
videos/chunk-000/observation.images.cam_high/episode_000000.mp4
videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
```

The converter decodes RGB with FFmpeg. It resizes head to 320×256 (width×height), wrists to 160×128 each, using OpenCV AREA interpolation; head is on top and wrists below. The final array is `[T,384,320,3]`. Action and state must be `[T,14]`. Mismatching frame counts fail by default; `--truncate-to-shortest` explicitly opts into the historical converter's shortest-stream behavior.

The task strings are matched exactly against the supplied ordered vocabulary. The default `rap_wam/task_strings.json` is the archived 35-task vocabulary. The converter writes this order into `task_strings.json` alongside the shards, and stores one integer per episode in `meta/task_index`. Never reorder a vocabulary after conversion.

RoboDojo Zarr keys:

| Key | Dtype / shape |
|---|---|
| `data/rgb_head` | uint8 `[T,384,320,3]` |
| `data/action_joint` | float32 `[T,14]` |
| `data/robot_state_joint` | float32 `[T,14]` |
| `meta/episode_ends` | int64 `[E]` |
| `meta/task_index` | int64 `[E]`, in `[0,num_tasks)` |

Task-context files contain `embedding_table: float32 [num_tasks,512,4096]` and the same ordered `tasks` list. Context rows are true UMT5 embeddings padded with zeros. Training rejects missing labels, out-of-range labels, mismatched task order, wrong dimensions and nonfinite context values.

Conversion is sequential and shard-based, keeping approximately one decoded episode in RAM. It publishes a new output directory only after validation. Existing outputs are refused; automatic conversion resume is not implemented. `--limit N` supports explicit subsets; no incomplete episodes are silently skipped.

## Real-Robot Input

Provide one `.npz` per demonstration, with **synchronized** arrays below. `np.load(..., allow_pickle=False)` is used. Files are processed in lexical filename order; the fixed100 recipe requires exactly 100 episodes.

| NPZ key / resulting `data/` key | Dtype / shape |
|---|---|
| `rgb_head` | uint8 `[T,224,224,3]` |
| `waist_joints_states` | float32 `[T,2]` |
| `left_arm_states` / `right_arm_states` | float32 `[T,7]` each |
| `left_hand_states` / `right_hand_states` | float32 `[T,6]` each |
| `action_waist_joints` | float32 `[T,2]` |
| `action_arm_joints` | float32 `[T,14]`, left then right arm |
| `action_left_hand_joints` / `action_right_hand_joints` | float32 `[T,6]` each |

The model concatenates state as waist → left arm → right arm → left hand → right hand, and action in the analogous order. Commands must be **absolute joint positions**, with the same joint ordering and units as state. Preserve the source dataset's waist coordinate convention; this training release performs no SDK conversion or joint reordering. The raw interface requires arrays under these named keys, not an unspecified 28D vector.

A minimal export pattern, after collecting and aligning the named arrays:

```python
import numpy as np
np.savez_compressed("episode_000000.npz", rgb_head=rgb_uint8,
                    waist_joints_states=waist, left_arm_states=left_arm,
                    right_arm_states=right_arm, left_hand_states=left_hand,
                    right_hand_states=right_hand,
                    action_waist_joints=waist_command,
                    action_arm_joints=both_arm_commands,
                    action_left_hand_joints=left_hand_command,
                    action_right_hand_joints=right_hand_command)
```

Existing Zarr using this exact contract can bypass conversion. Other episode counts can be used intentionally by changing both `--expected-episodes` during conversion/preflight and `task.dataset.expected_episode_count` during training. Such a run is no longer the fixed100 reference.

Standalone preflight checks schema and the first 4096 low-dimensional frames per shard by default; `--full` scans every low-dimensional value. Training defaults to `training.preflight_full=true`, scanning every low-dimensional value before model allocation. This is data validation, not a hardware calibration check.

## Temporal Alignment at a Glance

For a sample starting at source frame s, with one observation and 16-step horizons:

| Tensor | Source offsets relative to s |
|---|---|
| Current image / state | 0 |
| `obs.rgb_head_future` / `action` | 1 through 16 |
| Recorded `obs2.robot_state` | 16 |
| `obs2.rgb_head_future` / `action_2` | 17 through 32 |

Rehearsal replaces the second chunk's image condition with the last generated first-chunk frame. Its future frames/actions still come from the recorded offsets above. Optional state substitution uses the last predicted first-chunk absolute-joint action. No window crosses an episode boundary.

Conversion stores report fields (`profile`, `shards`, `episodes`, `frames`, `two_chunk_windows_before_split`, `low_dim_scan`) in `manifest.json` beside the shards. This is a conversion/data summary, distinct from the release's file-hash manifest under `provenance/`.

Continue with the [RoboDojo Training](ROBODOJO.md) or [real-robot walkthrough](REAL_ROBOT.md).
