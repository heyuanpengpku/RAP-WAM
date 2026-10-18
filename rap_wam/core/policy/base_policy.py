from __future__ import annotations

from typing import Dict

import torch

from rap_wam.core.model.common.module_attr_mixin import ModuleAttrMixin
from rap_wam.core.model.common.normalizer import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    """Base interface for image-conditioned action policies."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def reset(self):
        return None

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError
