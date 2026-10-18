"""Two adjacent demonstration chunks for predictive rehearsal.

The first chunk supplies ordinary training targets. The following chunk
supplies recorded image/action targets and its initial state. Both windows
belong to one episode and share the parent dataset split."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from rap_wam.data.sequence_dataset import SequenceDataset


class TwoChunkDataset(SequenceDataset):
    """Consecutive second-chunk observations and actions for rehearsal."""

    # ---- sample index: require room for 2 consecutive chunks ----
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

        # chunk-1 window = [offset + horizon, offset + horizon + sequence_horizon)
        two_chunk_span = int(self.video_future_horizon) + int(self.sequence_horizon)

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
            if ep_length < two_chunk_span:
                continue
            for offset in range(0, ep_length - two_chunk_span + 1):
                samples.append((zarr_idx, ep_idx, offset))
        return samples

    # ---- getitem: parent chunk-0 + a consecutive chunk-1 ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = super().__getitem__(idx)  # chunk-0, identical to parent

        zarr_idx, ep_idx, frame_offset = self._samples[int(idx)]
        ends = self._meta[zarr_idx]["episode_ends"]
        ep_start_global = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
        # chunk-1 window starts one horizon later (matches infer gt_pos += horizon)
        fs2 = ep_start_global + int(frame_offset) + int(self.video_future_horizon)

        target_full2 = self._read_rgb_window(
            zarr_idx=zarr_idx, frame_start=fs2, n_frames=self.sequence_horizon
        )
        future2 = target_full2[self.n_obs_steps : self.n_obs_steps + self.video_future_horizon]
        if future2.shape[0] != self.video_future_horizon:
            raise ValueError(
                f"chunk-1 future horizon mismatch: expected {self.video_future_horizon}, got {future2.shape[0]}"
            )

        obs2: Dict[str, torch.Tensor] = {
            self.output_future_key: self.process_obs_array(self.output_future_key, future2),
        }
        if self._state_keys:
            state_chunks = [
                self._read_low_dim(zarr_idx, key, fs2, self.n_obs_steps)
                for key in self._state_keys
            ]
            state_concat = np.concatenate(state_chunks, axis=1).astype(np.float32, copy=False)
            obs2[self.output_state_key] = self.process_obs_array(self.output_state_key, state_concat)
        batch["obs2"] = obs2

        if self.include_action and self._action_keys:
            action_chunks = []
            for key in self._action_keys:
                arr = self._read_low_dim(zarr_idx, key, fs2, self.sequence_horizon)
                action_chunks.append(
                    arr[self.n_obs_steps : self.n_obs_steps + self.action_output_horizon]
                )
            action_concat = np.concatenate(action_chunks, axis=1).astype(np.float32, copy=False)
            batch["action_2"] = torch.from_numpy(action_concat)

        return batch


__all__ = ["TwoChunkDataset"]
