"""LanguageDataset -- 2chunk dataset + per-task LANGUAGE index.

NEW FILE, purely additive: zero change to the parent 2chunk dataset, the base
SequenceDataset, or any rap_wam.core code.

What it adds
------------
Every sample carries a scalar ``task_index`` (int in [0, 34]) identifying which of
the 35 RoboDojo tasks the episode belongs to.  The language *embedding* itself is
NOT loaded here (T5/umt5 is never loaded during training); the policy holds a
precomputed [35, 512, 4096] table and looks it up by this index.  So this class
only has to answer "which task is sample idx?" -- a single int per sample that
collates (default_collate) to a clean [B] LongTensor (no custom collate_fn).

Source of truth
---------------
``meta/task_index`` int64 length == n_episodes was written into every shard by the
offline zarr-augment step (mirrors how ``meta/episode_ends`` is stored).  We read it
once per zarr (same pattern the parent uses to read episode_ends) and cache a
per-zarr array; ``__getitem__`` then indexes it by the episode index that the parent
already resolves in ``self._samples[idx]``.

Robustness: if a shard is missing ``meta/task_index`` (e.g. an un-augmented zarr),
that zarr's episodes fall back to task_index 0 with a one-time warning, so training
never crashes on a partially-prepared dataset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import zarr

from rap_wam.data.two_chunk_dataset import TwoChunkDataset


class LanguageDataset(TwoChunkDataset):
    """2chunk dataset + a scalar ``task_index`` per sample (language conditioning)."""

    _TASK_INDEX_KEY = "meta/task_index"

    # ---- lazily-built per-zarr task_index cache (mirrors episode_ends storage) ----
    def _get_task_index_table(self) -> List[Optional[np.ndarray]]:
        cache = getattr(self, "_task_index_by_zarr", None)
        if cache is not None:
            return cache
        table: List[Optional[np.ndarray]] = []
        warned = False
        for zpath in self._zarr_paths:
            arr: Optional[np.ndarray] = None
            try:
                root = zarr.open(zpath, mode="r")
                if self._TASK_INDEX_KEY in root:
                    arr = np.asarray(root[self._TASK_INDEX_KEY][:], dtype=np.int64)
            except Exception:
                arr = None
            if arr is None and not warned:
                print(
                    f"[lang_dataset] WARNING: {self._TASK_INDEX_KEY} missing in "
                    f"{zpath} (and possibly others) -> those episodes use task_index 0"
                )
                warned = True
            table.append(arr)
        self._task_index_by_zarr = table
        return table

    def _task_index_for(self, zarr_idx: int, ep_idx: int) -> int:
        table = self._get_task_index_table()
        arr = table[int(zarr_idx)]
        if arr is None:
            return 0
        j = int(ep_idx)
        if j < 0 or j >= arr.shape[0]:
            return 0
        return int(arr[j])

    # ---- getitem: parent output + scalar task_index ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = super().__getitem__(idx)  # chunk-0 + chunk-1, identical to 2chunk parent
        zarr_idx, ep_idx = self._samples[int(idx)][:2]
        ti = self._task_index_for(zarr_idx, ep_idx)
        batch["task_index"] = torch.tensor(int(ti), dtype=torch.long)
        return batch


__all__ = ["LanguageDataset"]
