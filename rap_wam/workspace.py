"""DDP/CPU-only training workspace without optional DeepSpeed unwrapping imports."""
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from rap_wam.training.metric_workspace import MetricTrainingWorkspace


def unwrap_ddp(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


class TrainWorkspace(MetricTrainingWorkspace):
    def _get_eval_policy(self, accelerator):
        return self.ema_model if self.ema_model is not None else unwrap_ddp(self.model)

    def _save_checkpoint_bundle(self, *, accelerator, ckpt_dir: Path, topk_manager, monitor_payload):
        if not accelerator.is_main_process:
            return
        wrapped = self.model
        self.model = unwrap_ddp(wrapped)
        try:
            self.save_checkpoint(ckpt_dir / 'latest.ckpt')
            augmented = self._augment_checkpoint_metrics(monitor_payload)
            if topk_manager.monitor_key in augmented:
                path = topk_manager.get_ckpt_path(augmented)
                if path:
                    self.save_checkpoint(path)
        finally:
            self.model = wrapped
