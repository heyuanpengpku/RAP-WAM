from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rap_wam.core.common.normalize_util import get_image_range_normalizer
from rap_wam.core.dataset.preprocessors.base import BaseObsPreprocessor
from rap_wam.core.dataset.preprocessors.utils import normalize_image_size


class RGBPreprocessor(BaseObsPreprocessor):
    """Convert temporal RGB observations to float tensors of shape [T,C,H,W]."""

    def __init__(
        self,
        *,
        image_size: int | Sequence[int] | None = None,
        default_image_size: Sequence[int] | None = None,
    ) -> None:
        self.image_size = normalize_image_size(
            image_size,
            default_image_size=default_image_size,
        )

    def process(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        del obs_shape
        rgb = torch.from_numpy(np.asarray(value))
        if rgb.ndim != 4:
            raise ValueError(f"{key} must be a 4D RGB tensor [T,H,W,C] or [T,C,H,W]; got {tuple(rgb.shape)}")

        if rgb.shape[-1] in (1, 3, 4):
            rgb = rgb.permute(0, 3, 1, 2)
        elif rgb.shape[1] in (1, 3, 4):
            pass
        else:
            raise ValueError(f"{key} cannot infer the channel dimension; shape={tuple(rgb.shape)}")

        is_uint8 = rgb.dtype == torch.uint8
        rgb = rgb[:, :3].float()
        if is_uint8 or (rgb.numel() > 0 and rgb.max() > 1.0):
            rgb = rgb / 255.0

        return F.interpolate(
            rgb,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

    def get_normalizer(self, *, key: str):
        del key
        return get_image_range_normalizer()


__all__ = ["RGBPreprocessor"]
