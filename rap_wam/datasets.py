"""Public data adapters for the two supported training workflows."""
from __future__ import annotations
import numpy as np
import zarr
from rap_wam.data.language_dataset import LanguageDataset
from rap_wam.fixed100_dataset import Fixed100Dataset


class RoboDojoDataset(LanguageDataset):
    """Reject missing language labels instead of silently assigning task zero."""
    def __init__(self, *args, num_tasks: int = 35, **kwargs):
        self.num_tasks = int(num_tasks)
        if self.num_tasks < 1:
            raise ValueError('num_tasks must be positive')
        super().__init__(*args, **kwargs)
        table = []
        for path, meta in zip(self._zarr_paths, self._meta):
            root = zarr.open(path, mode='r')
            if 'meta/task_index' not in root:
                raise ValueError(f'Missing meta/task_index: {path}')
            raw = root['meta/task_index']
            if raw.dtype.kind not in 'iu':
                raise ValueError(f'task_index must be an integer array: {path}')
            labels = np.asarray(raw[:], dtype=np.int64)
            if labels.shape != (int(meta['n_episodes']),):
                raise ValueError(f'task_index must contain one label per episode: {path}')
            if np.any(labels < 0) or np.any(labels >= self.num_tasks):
                raise ValueError(f'task_index must be in [0, {self.num_tasks}): {path}')
            table.append(labels)
        self._task_index_by_zarr = table
        if self.state_dim != 14 or self.action_dim != 14:
            raise ValueError('This RoboDojo recipe requires 14D joint state/action')


class RealRobotDataset(Fixed100Dataset):
    """Head camera and 28D absolute joint-position training data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.state_dim != 28 or self.action_dim != 28:
            raise ValueError('This real-robot recipe requires 28D joint state/action')
