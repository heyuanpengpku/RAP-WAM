"""Visual-metric tracking for training checkpoint selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rap_wam.core.workspace.wan_train_workspace import WanTrainWorkspace


class MetricTrainingWorkspace(WanTrainWorkspace):
    """Training workspace that tracks sampled visual MSE for checkpoint selection."""

    def __init__(self, cfg: Any, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir=output_dir)
        
        self._latest_sample_visual_mse: float | None = None
        self._latest_sample_action_mse: float | None = None
        self._latest_sample_step: int = -1

    @torch.no_grad()
    def _run_sampling_evaluation(self, **kwargs: Any):
        metrics = super()._run_sampling_evaluation(**kwargs)
        if isinstance(metrics, dict):
            visual = metrics.get("sample_visual_mse")
            if visual is not None:
                self._latest_sample_visual_mse = float(visual)
                self._latest_sample_step = int(self.global_step)
            action = metrics.get("sample_action_mse_error")
            if action is not None:
                self._latest_sample_action_mse = float(action)
        return metrics

    def _augment_checkpoint_metrics(self, monitor_payload):
        augmented = dict(monitor_payload)
        if self._latest_sample_visual_mse is not None:
            augmented["sample_visual_mse"] = float(self._latest_sample_visual_mse)
        if self._latest_sample_action_mse is not None:
            augmented["sample_action_mse_error"] = float(self._latest_sample_action_mse)
        return augmented

    def _save_checkpoint_bundle(self, *, accelerator, ckpt_dir, topk_manager, monitor_payload):
        return super()._save_checkpoint_bundle(
            accelerator=accelerator, ckpt_dir=ckpt_dir, topk_manager=topk_manager,
            monitor_payload=self._augment_checkpoint_metrics(monitor_payload),
        )


__all__ = ["MetricTrainingWorkspace"]
