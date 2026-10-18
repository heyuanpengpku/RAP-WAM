from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from rap_wam.core.dataset.preprocessors.base import BaseObsPreprocessor


class PassthroughPreprocessor(BaseObsPreprocessor):
    """Preserve input tensor values without additional preprocessing."""

    def process(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        del key, obs_shape
        return torch.from_numpy(np.asarray(value))


__all__ = ["PassthroughPreprocessor"]
