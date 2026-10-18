"""Strict fixed100 two-chunk dataset for the 28D real-robot workflow.

This is an additive wrapper around the shared RAP-WAM dataset.  Its only
behavioural change is that ``val_split_ratio=0`` really means zero held-out
episodes: all 100 selected demonstrations contribute training windows.  The
upstream loader otherwise reserves one episode even when the ratio is zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rap_wam.data.two_chunk_dataset import TwoChunkDataset


class Fixed100Dataset(TwoChunkDataset):
    """Two-chunk dataset with strict data-card and zero-validation checks."""

    def __init__(
        self,
        *args: Any,
        expected_episode_count: int = 100,
        expected_total_frames: int | None = None,
        expected_train_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._fixed100_expected_episode_count = int(expected_episode_count)
        self._fixed100_expected_total_frames = (
            None if expected_total_frames is None else int(expected_total_frames)
        )
        self._fixed100_expected_train_samples = (
            None if expected_train_samples is None else int(expected_train_samples)
        )
        super().__init__(*args, **kwargs)

        episodes = sum(int(meta["n_episodes"]) for meta in self._meta)
        frames = sum(int(meta["n_frames"]) for meta in self._meta)
        if episodes != self._fixed100_expected_episode_count:
            raise ValueError(
                f"fixed100 episode contract mismatch: {episodes} != "
                f"{self._fixed100_expected_episode_count}"
            )
        if (
            self._fixed100_expected_total_frames is not None
            and frames != self._fixed100_expected_total_frames
        ):
            raise ValueError(
                f"fixed100 frame contract mismatch: {frames} != "
                f"{self._fixed100_expected_total_frames}"
            )
        if (
            self.split == "train"
            and self._fixed100_expected_train_samples is not None
            and len(self._samples) != self._fixed100_expected_train_samples
        ):
            raise ValueError(
                f"fixed100 sample contract mismatch: {len(self._samples)} != "
                f"{self._fixed100_expected_train_samples}"
            )
        if self.val_split_ratio <= 0.0 and self.split == "val" and self._samples:
            raise ValueError("fixed100 zero-validation contract produced validation samples")

    def _build_sample_index(self) -> List[Tuple[int, int, int]]:
        if self.val_split_ratio > 0.0:
            return super()._build_sample_index()

        if self.split == "val":
            return []

        two_chunk_span = int(self.video_future_horizon) + int(self.sequence_horizon)
        samples: List[Tuple[int, int, int]] = []
        for zarr_idx, meta in enumerate(self._meta):
            ends = meta["episode_ends"]
            for ep_idx in range(int(meta["n_episodes"])):
                ep_start = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
                ep_end = int(ends[ep_idx])
                ep_length = ep_end - ep_start
                if ep_length < two_chunk_span:
                    continue
                for offset in range(ep_length - two_chunk_span + 1):
                    samples.append((zarr_idx, ep_idx, offset))
        return samples

    def describe_dataloader_sampling(self) -> Dict[str, Any]:
        result = dict(super().describe_dataloader_sampling())
        result.update(
            {
                "mode": "fixed100_full_demo_2chunk",
                "selected_source_episodes": self._fixed100_expected_episode_count,
                "zero_validation_is_exact": self.val_split_ratio <= 0.0,
            }
        )
        return result


__all__ = ["Fixed100Dataset"]
