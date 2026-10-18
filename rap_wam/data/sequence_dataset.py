"""Shared Zarr sequence dataset for RAP-WAM training.

Reads concatenated modality arrays and episode boundaries, preserves
episode-level splits, and fits normalizers to training episodes only."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import tqdm
import zarr

from rap_wam.core.common.normalize_util import get_image_range_normalizer
from rap_wam.core.dataset.normalizer_stats import StreamingArrayStats
from rap_wam.core.dataset.normalizer_stats import pop_progress_kwargs
from rap_wam.core.dataset.preprocessors import build_obs_preprocessors
from rap_wam.core.dataset.preprocessors.utils import normalize_image_size
from rap_wam.core.model.common.normalizer import LinearNormalizer


_DEFAULT_STATE_KEYS = (
    "waist_joints_states",
    "left_arm_states",
    "right_arm_states",
    "left_hand_states",
    "right_hand_states",
)

_DEFAULT_ACTION_KEYS = (
    "action_waist_joints",
    "action_arm_joints",
    "action_left_hand_joints",
    "action_right_hand_joints",
)


class SequenceDataset(torch.utils.data.Dataset):
    """Reads multiple Diffusion-Policy-style zarr files.

    Each zarr at ``<data_root>/*.zarr`` contains:
      - ``data/<modality>``  arrays of shape ``[N_total_frames, ...]``
        concatenated across all episodes in that zarr
      - ``meta/episode_ends`` int64 1D array containing episode end indices

    Each item exposes the shared image/action training contract:

        {
          "obs": {
            "<output_condition_key>": [n_obs_steps, 3, H, W'] float in [0, 1],
            "<output_future_key>":    [video_future_horizon, 3, H, W'] float in [0, 1],
            "<output_last_key>":      [1, 3, H, W'] float in [0, 1],
            "<output_state_key>":     [n_obs_steps, state_dim] float,   # if state_keys
          },
          "action": [action_horizon, action_dim] float,                  # if include_action
        }

    For multi-view mode (e.g. views=["rgb_head","rgb_left_hand","rgb_right_hand"])
    the views are horizontally concatenated into a single wide tensor (W' = W * len(views))
    and exposed under output_condition_key. Wan2.2 VAE accepts arbitrary H,W as long
    as both are divisible by 16, so 224 x (224 * 3) = 224 x 672 mosaic works.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        zarr_files: Sequence[str] | None = None,
        views: Sequence[str] = ("rgb_head",),
        output_condition_key: str = "rgb_head",
        output_future_key: str = "rgb_head_future",
        output_last_key: str = "rgb_head_last",
        state_keys: Sequence[str] | None = None,
        output_state_key: str = "robot_state",
        action_keys: Sequence[str] | None = None,
        include_action: bool = True,
        n_obs_steps: int = 1,
        n_action_steps: int = 16,
        video_future_horizon: int = 16,
        image_size: Tuple[int, int] | None = None,
        preprocessors: Any | None = None,
        val_split_ratio: float = 0.1,
        split_seed: int = 42,
        split: str = "train",
        pad_before: int = 0,
        pad_after: int = 0,
    ) -> None:
        super().__init__()
        del pad_before, pad_after

        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        self._views = tuple(str(v).strip() for v in views if str(v).strip())
        if not self._views:
            raise ValueError("views must not be empty")

        self._state_keys = tuple(str(k).strip() for k in (state_keys if state_keys is not None else _DEFAULT_STATE_KEYS))
        self._state_keys = tuple(k for k in self._state_keys if k)
        self._action_keys = tuple(str(k).strip() for k in (action_keys if action_keys is not None else _DEFAULT_ACTION_KEYS))
        self._action_keys = tuple(k for k in self._action_keys if k)

        self.output_condition_key = str(output_condition_key).strip() or "rgb_head"
        self.output_future_key = str(output_future_key).strip() or "rgb_head_future"
        self.output_last_key = str(output_last_key).strip() or "rgb_head_last"
        self.output_state_key = str(output_state_key).strip() or "robot_state"
        self.include_action = bool(include_action)

        self.n_obs_steps = int(n_obs_steps)
        self.video_future_horizon = int(video_future_horizon)
        self.action_output_horizon = int(n_action_steps) if self.include_action else 0
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be > 0")
        if self.video_future_horizon <= 0:
            raise ValueError("video_future_horizon must be > 0")

        self.sequence_horizon = self.n_obs_steps + max(
            self.video_future_horizon,
            self.action_output_horizon,
        )

        self.split = str(split).strip().lower()
        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train/val/all, got {self.split}")
        self.val_split_ratio = float(val_split_ratio)
        if not 0.0 <= self.val_split_ratio < 1.0:
            raise ValueError("val_split_ratio must be finite and in [0, 1)")
        if self.include_action and self.action_output_horizon <= 0:
            raise ValueError("n_action_steps must be positive when actions are enabled")
        self.split_seed = int(split_seed)

        # discover zarr files
        if zarr_files is None or len(zarr_files) == 0:
            discovered = sorted(
                p for p in self.data_root.iterdir() if p.suffix == ".zarr" and p.is_dir()
            )
            if not discovered:
                raise FileNotFoundError(f"no .zarr directories found under {self.data_root}")
            self._zarr_paths = tuple(str(p) for p in discovered)
        else:
            paths = []
            for z in zarr_files:
                z_path = Path(z)
                if not z_path.is_absolute():
                    z_path = self.data_root / z
                if not z_path.exists():
                    raise FileNotFoundError(f"zarr not found: {z_path}")
                paths.append(str(z_path.resolve()))
            self._zarr_paths = tuple(paths)

        # peek metadata
        self._meta = self._load_zarr_meta(self._zarr_paths)
        self._validate_meta_compat()

        # peek frame dims from first view in first zarr
        first_root = zarr.open(self._zarr_paths[0], mode="r")
        head_arr = first_root[f"data/{self._views[0]}"]
        if head_arr.ndim != 4:
            raise ValueError(
                f"expected RGB array {self._views[0]} to be 4D [T,H,W,C], got {head_arr.shape}"
            )
        _, H, W, C = head_arr.shape
        if C != 3:
            raise ValueError(f"expected 3 RGB channels, got C={C}")
        self.frame_h = int(H)
        self.frame_w_per_view = int(W)
        self.frame_w = int(W * len(self._views))

        # state / action dims
        self.state_dims: Dict[str, int] = {}
        self.state_shapes: Dict[str, Tuple[int, ...]] = {}
        for key in self._state_keys:
            arr = first_root[f"data/{key}"]
            if arr.ndim == 1:
                self.state_dims[key] = 1
                self.state_shapes[key] = (1,)
            else:
                self.state_dims[key] = int(np.prod(arr.shape[1:]))
                self.state_shapes[key] = tuple(int(v) for v in arr.shape[1:])
        self.state_dim = int(sum(self.state_dims.values()))

        self.action_dims: Dict[str, int] = {}
        self.action_shapes: Dict[str, Tuple[int, ...]] = {}
        for key in self._action_keys:
            if f"data/{key}" not in first_root:
                if self.include_action:
                    raise KeyError(
                        f"action key {key} missing in {self._zarr_paths[0]}"
                    )
                continue
            arr = first_root[f"data/{key}"]
            if arr.ndim == 1:
                self.action_dims[key] = 1
                self.action_shapes[key] = (1,)
            else:
                self.action_dims[key] = int(np.prod(arr.shape[1:]))
                self.action_shapes[key] = tuple(int(v) for v in arr.shape[1:])
        self.action_dim = int(sum(self.action_dims.values()))

        # build sample index for this split
        self._samples = self._build_sample_index()

        # obs_shapes / obs_keys
        if image_size is None:
            self.image_size = (self.frame_h, self.frame_w)
        else:
            self.image_size = normalize_image_size(
                image_size, default_image_size=(self.frame_h, self.frame_w)
            )
        output_image_shape = (3, int(self.image_size[0]), int(self.image_size[1]))

        self.obs_shapes: Dict[str, Tuple[int, ...]] = {
            self.output_condition_key: output_image_shape,
            self.output_future_key: output_image_shape,
            self.output_last_key: output_image_shape,
        }
        if self._state_keys:
            self.obs_shapes[self.output_state_key] = (self.state_dim,)
        self._obs_keys = [
            self.output_condition_key,
            self.output_future_key,
            self.output_last_key,
        ]
        if self._state_keys:
            self._obs_keys.append(self.output_state_key)

        self._obs_preprocessors = build_obs_preprocessors(
            preprocessors,
            obs_keys=self._obs_keys,
            default_image_size=self.image_size,
        )

        # per-process zarr handle cache (for DataLoader workers)
        self._zarr_cache: dict[int, zarr.Group] = {}
        self._zarr_lock = threading.Lock()

    # ---------------- properties ----------------

    @property
    def obs_keys(self) -> Sequence[str]:
        return list(self._obs_keys)

    @property
    def action_keys(self) -> Sequence[str]:
        return list(self._action_keys)

    @property
    def source_lengths(self) -> Tuple[int, ...]:
        return (len(self._samples),)

    @property
    def source_offsets(self) -> Tuple[int, ...]:
        return (0,)

    def __len__(self) -> int:
        return len(self._samples)

    # ---------------- index ----------------

    def _build_sample_index(self) -> list[tuple[int, int, int]]:
        ordered_episodes = [
            (zarr_idx, ep_idx)
            for zarr_idx, meta in enumerate(self._meta)
            for ep_idx in range(int(meta["n_episodes"]))
        ]
        rng = np.random.default_rng(self.split_seed)
        n_total = len(ordered_episodes)
        if n_total == 0:
            return []
        perm = rng.permutation(n_total).astype(np.int64)
        n_val = max(1, int(round(self.val_split_ratio * n_total))) if n_total > 1 else 0
        val_set = set(int(i) for i in perm[:n_val].tolist())

        samples: list[tuple[int, int, int]] = []
        for flat_idx, (zarr_idx, ep_idx) in enumerate(ordered_episodes):
            in_val = flat_idx in val_set
            if self.split == "train" and in_val:
                continue
            if self.split == "val" and not in_val:
                continue

            meta = self._meta[zarr_idx]
            ends = meta["episode_ends"]
            ep_start_global = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
            ep_end_global = int(ends[ep_idx])
            ep_length = ep_end_global - ep_start_global
            if ep_length < self.sequence_horizon:
                continue
            for offset in range(0, ep_length - self.sequence_horizon + 1):
                samples.append((zarr_idx, ep_idx, offset))
        return samples

    def get_sample_meta(self, idx: int) -> Dict[str, Any]:
        zarr_idx, ep_idx, frame_offset = self._samples[int(idx)]
        meta = self._meta[zarr_idx]
        ends = meta["episode_ends"]
        ep_start_global = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
        ep_end_global = int(ends[ep_idx])
        return {
            "source_kind": "zarr",
            "zarr_idx": int(zarr_idx),
            "zarr_path": self._zarr_paths[zarr_idx],
            "episode_idx": int(ep_idx),
            "episode_start_global": ep_start_global,
            "episode_end_global": ep_end_global,
            "frame_offset": int(frame_offset),
            "frame_start_global": ep_start_global + int(frame_offset),
            "n_obs_steps": int(self.n_obs_steps),
            "video_future_horizon": int(self.video_future_horizon),
            "action_output_horizon": int(self.action_output_horizon),
        }

    # ---------------- getitem ----------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        zarr_idx, ep_idx, frame_offset = self._samples[int(idx)]
        meta = self._meta[zarr_idx]
        ends = meta["episode_ends"]
        ep_start_global = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
        frame_start_global = ep_start_global + int(frame_offset)

        condition_raw = self._read_rgb_window(
            zarr_idx=zarr_idx, frame_start=frame_start_global, n_frames=self.n_obs_steps
        )
        target_full = self._read_rgb_window(
            zarr_idx=zarr_idx, frame_start=frame_start_global, n_frames=self.sequence_horizon
        )
        future_raw = target_full[self.n_obs_steps : self.n_obs_steps + self.video_future_horizon]
        if future_raw.shape[0] != self.video_future_horizon:
            raise ValueError(
                f"future horizon mismatch: expected {self.video_future_horizon}, got {future_raw.shape[0]}"
            )
        last_raw = future_raw[-1:]

        obs: Dict[str, torch.Tensor] = {
            self.output_condition_key: self.process_obs_array(
                self.output_condition_key, condition_raw
            ),
            self.output_future_key: self.process_obs_array(
                self.output_future_key, future_raw
            ),
            self.output_last_key: self.process_obs_array(
                self.output_last_key, last_raw
            ),
        }
        if self._state_keys:
            state_chunks = []
            for key in self._state_keys:
                arr = self._read_low_dim(
                    zarr_idx, key, frame_start_global, self.n_obs_steps
                )
                state_chunks.append(arr)
            state_concat = np.concatenate(state_chunks, axis=1).astype(
                np.float32, copy=False
            )
            obs[self.output_state_key] = self.process_obs_array(
                self.output_state_key, state_concat
            )

        batch: Dict[str, Any] = {"obs": obs}

        if self.include_action and self._action_keys:
            action_chunks = []
            for key in self._action_keys:
                arr = self._read_low_dim(
                    zarr_idx, key, frame_start_global, self.sequence_horizon
                )
                action_slice = arr[
                    self.n_obs_steps : self.n_obs_steps + self.action_output_horizon
                ]
                action_chunks.append(action_slice)
            action_concat = np.concatenate(action_chunks, axis=1).astype(
                np.float32, copy=False
            )
            batch["action"] = torch.from_numpy(action_concat)

        return batch

    # ---------------- I/O ----------------

    def _open_zarr(self, zarr_idx: int) -> zarr.Group:
        cached = self._zarr_cache.get(zarr_idx)
        if cached is not None:
            return cached
        with self._zarr_lock:
            cached = self._zarr_cache.get(zarr_idx)
            if cached is None:
                cached = zarr.open(self._zarr_paths[zarr_idx], mode="r")
                self._zarr_cache[zarr_idx] = cached
        return cached

    def _read_rgb_window(
        self, *, zarr_idx: int, frame_start: int, n_frames: int
    ) -> np.ndarray:
        root = self._open_zarr(zarr_idx)
        parts = []
        for view in self._views:
            arr = root[f"data/{view}"]
            chunk = np.asarray(arr[frame_start : frame_start + n_frames])
            if chunk.shape[0] != n_frames:
                raise ValueError(
                    f"RGB window short read on {view}: expected {n_frames}, got {chunk.shape[0]}"
                )
            parts.append(chunk)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=2)

    def _read_low_dim(
        self, zarr_idx: int, key: str, frame_start: int, n_frames: int
    ) -> np.ndarray:
        root = self._open_zarr(zarr_idx)
        arr = root[f"data/{key}"]
        chunk = np.asarray(arr[frame_start : frame_start + n_frames])
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        elif chunk.ndim > 2:
            chunk = chunk.reshape(chunk.shape[0], -1)
        return chunk.astype(np.float32, copy=False)

    def process_obs_array(self, key: str, value: np.ndarray) -> torch.Tensor:
        preprocessor = self._obs_preprocessors[key]
        return preprocessor.process(value, key=key, obs_shape=self.obs_shapes[key])

    @staticmethod
    def _load_zarr_meta(paths: Sequence[str]) -> list[Dict[str, Any]]:
        meta_list = []
        for p in paths:
            root = zarr.open(p, mode="r")
            raw_ends = root["meta/episode_ends"]
            if raw_ends.ndim != 1 or raw_ends.dtype.kind not in "iu":
                raise ValueError(f"episode_ends must be a 1D integer array: {p}")
            ends = np.asarray(raw_ends[:], dtype=np.int64)
            if not len(ends) or np.any(np.diff(np.concatenate([[0], ends])) <= 0):
                raise ValueError(f"episode_ends must be nonempty and strictly increasing: {p}")
            meta_list.append(
                {
                    "path": p,
                    "episode_ends": ends,
                    "n_episodes": int(len(ends)),
                    "n_frames": int(ends[-1]) if len(ends) > 0 else 0,
                }
            )
        return meta_list

    def _validate_meta_compat(self) -> None:
        required = list(self._views)
        if self._state_keys:
            required += list(self._state_keys)
        if self.include_action and self._action_keys:
            required += list(self._action_keys)
        shapes = {}
        for path, meta in zip(self._zarr_paths, self._meta):
            root = zarr.open(path, mode="r")
            missing = [k for k in required if f"data/{k}" not in root]
            if missing:
                raise KeyError(f"{path} missing required keys: {missing}")
            for key in required:
                array = root[f"data/{key}"]
                if not array.ndim or array.shape[0] != int(meta["n_frames"]):
                    raise ValueError(f"{path}: {key} length must match the final episode end")
                shape = tuple(array.shape[1:])
                if key in shapes and shapes[key] != shape:
                    raise ValueError(f"{key} shape differs across shards: {path}")
                shapes[key] = shape
                if key in self._views and (array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8):
                    raise ValueError(f"{path}: {key} must be RGB uint8 [T,H,W,3]")
                if key not in self._views and array.dtype != np.float32:
                    raise ValueError(f"{path}: {key} must be float32")

    # ---------------- workspace interface ----------------

    def build_dataloader_sampler(
        self, *, shuffle: bool, start_epoch: int = 0
    ):
        del shuffle, start_epoch
        return None

    def describe_dataloader_sampling(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "mode": "rap_wam_zarr",
            "num_zarrs": len(self._zarr_paths),
            "num_samples": len(self._samples),
            "num_views": len(self._views),
            "frame_size": (self.frame_h, self.frame_w),
        }

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        fit_kwargs = dict(kwargs)
        # ignore fixed_preset for our dataset; we always scan
        fit_kwargs.pop("fixed_preset", None)
        last_n_dims = int(fit_kwargs.pop("last_n_dims", 1))
        show_progress, progress_desc = pop_progress_kwargs(
            fit_kwargs, default_desc=f"Loading rap_wam normalizer [{self.split}]"
        )

        action_stats = (
            StreamingArrayStats(last_n_dims=last_n_dims)
            if (self.include_action and self._action_keys)
            else None
        )
        state_stats = (
            StreamingArrayStats(last_n_dims=last_n_dims) if self._state_keys else None
        )

        if self.split != "train":
            raise ValueError("Fit normalization on the training split and reuse it for validation")
        ranges = list(self._selected_episode_ranges())
        if not ranges:
            raise ValueError("Cannot fit normalization on an empty training split")
        progress_bar = tqdm.tqdm(total=len(ranges), desc=progress_desc, unit="episode",
                                 leave=show_progress, disable=not show_progress)
        try:
            for zarr_idx, start, end in ranges:
                for offset in range(start, end, 4096):
                    count = min(4096, end - offset)
                    if action_stats is not None:
                        action_stats.update(np.concatenate([
                            self._read_low_dim(zarr_idx, key, offset, count)
                            for key in self._action_keys], axis=1))
                    if state_stats is not None:
                        state_stats.update(np.concatenate([
                            self._read_low_dim(zarr_idx, key, offset, count)
                            for key in self._state_keys], axis=1))
                progress_bar.update(1)
        finally:
            progress_bar.close()

        if action_stats is not None and action_stats.count > 0:
            normalizer["action"] = action_stats.build_normalizer(
                mode=mode, **fit_kwargs
            )
        if state_stats is not None and state_stats.count > 0:
            normalizer[self.output_state_key] = state_stats.build_normalizer(
                mode=mode, **fit_kwargs
            )
        for key in (
            self.output_condition_key,
            self.output_future_key,
            self.output_last_key,
        ):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def _selected_episode_ranges(self):
        """Yield complete frame ranges only for episodes represented in this split."""
        selected = {(int(sample[0]), int(sample[1])) for sample in self._samples}
        for zarr_idx, meta in enumerate(self._meta):
            start = 0
            for episode_idx, end in enumerate(meta["episode_ends"]):
                end = int(end)
                if (zarr_idx, episode_idx) in selected:
                    yield zarr_idx, start, end
                start = end

    def get_all_actions(self) -> torch.Tensor:
        if not self.include_action or not self._action_keys:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        values = [np.concatenate([
            self._read_low_dim(zarr_idx, key, start, end - start)
            for key in self._action_keys], axis=1)
            for zarr_idx, start, end in self._selected_episode_ranges()]
        if not values:
            return torch.empty((0, self.action_dim), dtype=torch.float32)
        return torch.from_numpy(np.concatenate(values, axis=0).astype(np.float32, copy=False))

    def export_model_spec(self) -> Dict[str, Any]:
        return {
            "obs_shapes": dict(self.obs_shapes),
            "action_dim": int(self.action_dim) if self.include_action else 0,
            "action_horizon": int(self.action_output_horizon),
            "obs_keys": list(self._obs_keys),
            "obs_horizon": int(self.n_obs_steps),
            "future_horizon": int(self.video_future_horizon),
            "state_dim": int(self.state_dim) if self._state_keys else 0,
            "state_horizon": int(self.n_obs_steps) if self._state_keys else 0,
            "state_key": self.output_state_key if self._state_keys else None,
        }


__all__ = ["SequenceDataset"]
