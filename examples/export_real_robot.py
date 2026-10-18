"""Export already synchronized arrays; the CLI emits a synthetic schema example."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


DIMS = {
    "waist_joints_states": 2,
    "left_arm_states": 7,
    "right_arm_states": 7,
    "left_hand_states": 6,
    "right_hand_states": 6,
    "action_waist_joints": 2,
    "action_arm_joints": 14,
    "action_left_hand_joints": 6,
    "action_right_hand_joints": 6,
}


def export_episode(path, *, rgb_head, **joint_arrays):
    """Save one NPZ without overwriting; do not infer ordering, units or timing.

    Pass every named field in DIMS as a keyword. RGB must be uint8 [T,224,224,3];
    joint arrays are converted to finite float32 with the documented dimensions.
    Input arrays must already describe synchronized absolute joint positions.
    """
    rgb = np.asarray(rgb_head)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (224, 224, 3):
        raise ValueError("rgb_head must be uint8 [T,224,224,3]")
    frames = len(rgb)
    if frames < 33:
        raise ValueError("Each two-chunk episode needs at least 33 frames")
    if set(joint_arrays) != set(DIMS):
        raise ValueError("Provide exactly the named waist, arm and hand arrays in DIMS")
    arrays = {"rgb_head": rgb}
    for key, width in DIMS.items():
        array = np.asarray(joint_arrays[key], dtype=np.float32)
        if array.shape != (frames, width) or not np.isfinite(array).all():
            raise ValueError(f"{key} must be finite float32 [{frames},{width}]")
        arrays[key] = array
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True,
                        help="Explicitly generate zero-valued example arrays, not robot data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    if args.frames < 33:
        parser.error("--frames must be at least 33")
    fields = {key: np.zeros((args.frames, width), np.float32) for key, width in DIMS.items()}
    export_episode(args.out, rgb_head=np.zeros((args.frames, 224, 224, 3), np.uint8), **fields)
    print(f"Wrote synthetic schema example: {args.out}")


if __name__ == "__main__":
    main()
