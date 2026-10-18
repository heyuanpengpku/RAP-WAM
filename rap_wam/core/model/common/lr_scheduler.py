from __future__ import annotations

import math

import torch


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int = 0,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """Create a learning-rate scheduler from a supported schedule name."""
    schedule = str(name).strip().lower()
    warmup = max(0, int(num_warmup_steps))
    total = max(1, int(num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup > 0 and current_step < warmup:
            return float(current_step) / float(max(1, warmup))

        progress = float(current_step - warmup) / float(max(1, total - warmup))
        progress = min(max(progress, 0.0), 1.0)

        if schedule in {"constant", "constant_with_warmup"}:
            return 1.0
        if schedule in {"linear", "linear_with_warmup"}:
            return max(0.0, 1.0 - progress)
        if schedule in {"cosine", "cosine_with_warmup"}:
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        raise ValueError(f"unsupported lr scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
