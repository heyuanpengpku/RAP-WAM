"""RoboDojo task conditioning with explicit labels for training and sampling."""
from pathlib import Path
from typing import List
import torch
from rap_wam.policy.rap_policy import RAPPolicy
from rap_wam.task_vocabulary import validate_task_strings


class LanguageConditionedRAPPolicy(RAPPolicy):
    def __init__(self, *args, task_ctx_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(task_ctx_path).expanduser()
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError('Task context must contain tasks and embedding_table')
        tasks = validate_task_strings(loaded.get('tasks'))
        table = loaded.get('embedding_table')
        expected = (len(tasks), self.text_len, self.text_dim)
        if not isinstance(table, torch.Tensor) or tuple(table.shape) != expected:
            raise ValueError(f'Task context must have shape {expected}')
        if not table.is_floating_point() or not torch.isfinite(table).all():
            raise ValueError('Task context must contain finite floating-point embeddings')
        self.register_buffer('_task_ctx', table.float(), persistent=False)
        self._cur_task_index = None

    def _checked_task_index(self, value, batch_size):
        if value is None:
            raise ValueError('Language-conditioned RoboDojo calls require task_index [B]')
        value = torch.as_tensor(value)
        if value.ndim != 1 or len(value) != batch_size or value.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise ValueError('task_index must be a 1D integer array with one label per sample')
        if torch.any(value < 0) or torch.any(value >= self._task_ctx.shape[0]):
            raise ValueError('task_index is outside the ordered vocabulary')
        return value.to(device=self._task_ctx.device, dtype=torch.long)

    def compute_loss(self, batch: dict, training: bool = True):
        previous = self._cur_task_index
        self._cur_task_index = self._checked_task_index(batch.get('task_index'), batch['action'].shape[0])
        try:
            return super().compute_loss(batch, training=training)
        finally:
            self._cur_task_index = previous

    def _sample_joint_predictions(self, *, obs_dict, **kwargs):
        previous = self._cur_task_index
        key = self.obs_encoder.image_key
        self._cur_task_index = self._checked_task_index(
            obs_dict.get('task_index', previous), obs_dict[key].shape[0])
        try:
            observations = {key: value for key, value in obs_dict.items() if key != 'task_index'}
            return super()._sample_joint_predictions(obs_dict=observations, **kwargs)
        finally:
            self._cur_task_index = previous

    def _build_context_list(self, batch_size: int) -> List[torch.Tensor]:
        indices = self._checked_task_index(self._cur_task_index, batch_size)
        dtype = self.model.text_embedding[0].weight.dtype
        selected = self._task_ctx.index_select(0, indices).to(device=self.device, dtype=dtype)
        return list(selected.unbind(0))


__all__ = ['LanguageConditionedRAPPolicy']
