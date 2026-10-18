"""Optional three-chunk extension of the shared sequence dataset.

This utility is retained for data compatibility; the public RoboDojo and
real-robot recipes use the two-chunk dataset."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from rap_wam.data.two_chunk_dataset import TwoChunkDataset


class ThreeChunkDataset(TwoChunkDataset):
    """2-chunk dataset + a genuine consecutive 3rd chunk (obs3 / action_3)."""

    # ---- sample index: require room for 3 consecutive chunks ----
    def _build_sample_index(self) -> List[Tuple[int, int, int]]:
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

        # chunk-2 window = [offset + 2*horizon, offset + 2*horizon + sequence_horizon)
        three_chunk_span = 2 * int(self.video_future_horizon) + int(self.sequence_horizon)

        samples: List[Tuple[int, int, int]] = []
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
            if ep_length < three_chunk_span:
                continue
            for offset in range(0, ep_length - three_chunk_span + 1):
                samples.append((zarr_idx, ep_idx, offset))
        return samples

    # ---- getitem: 2-chunk parent (chunk0+chunk1) + a consecutive chunk-2 ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = super().__getitem__(idx)  # chunk-0 + obs2/action_2 (chunk-1)

        zarr_idx, ep_idx, frame_offset = self._samples[int(idx)]
        ends = self._meta[zarr_idx]["episode_ends"]
        ep_start_global = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
        # chunk-2 window starts two horizons later
        fs3 = ep_start_global + int(frame_offset) + 2 * int(self.video_future_horizon)

        target_full3 = self._read_rgb_window(
            zarr_idx=zarr_idx, frame_start=fs3, n_frames=self.sequence_horizon
        )
        future3 = target_full3[self.n_obs_steps : self.n_obs_steps + self.video_future_horizon]
        if future3.shape[0] != self.video_future_horizon:
            raise ValueError(
                f"chunk-2 future horizon mismatch: expected {self.video_future_horizon}, got {future3.shape[0]}"
            )

        obs3: Dict[str, torch.Tensor] = {
            self.output_future_key: self.process_obs_array(self.output_future_key, future3),
        }
        if self._state_keys:
            state_chunks = [
                self._read_low_dim(zarr_idx, key, fs3, self.n_obs_steps)
                for key in self._state_keys
            ]
            state_concat = np.concatenate(state_chunks, axis=1).astype(np.float32, copy=False)
            obs3[self.output_state_key] = self.process_obs_array(self.output_state_key, state_concat)
        batch["obs3"] = obs3

        if self.include_action and self._action_keys:
            action_chunks = []
            for key in self._action_keys:
                arr = self._read_low_dim(zarr_idx, key, fs3, self.sequence_horizon)
                action_chunks.append(
                    arr[self.n_obs_steps : self.n_obs_steps + self.action_output_horizon]
                )
            action_concat = np.concatenate(action_chunks, axis=1).astype(np.float32, copy=False)
            batch["action_3"] = torch.from_numpy(action_concat)

        return batch


__all__ = ["ThreeChunkDataset"]
