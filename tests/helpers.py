from pathlib import Path
import numpy as np
import torch
from rap_wam.prepare_data import STATE_KEYS, ACTION_KEYS
from rap_wam.core.model.common.normalizer import LinearNormalizer


def episode(profile, n=40, offset=0):
    size = (384, 320) if profile == 'robodojo' else (224, 224)
    rgb = np.broadcast_to(np.zeros((1, *size, 3), np.uint8), (n, *size, 3))
    dims = {'action_joint': 14, 'robot_state_joint': 14} if profile == 'robodojo' else {**STATE_KEYS, **ACTION_KEYS}
    arrays = {'rgb_head': rgb}
    for key, dim in dims.items():
        arrays[key] = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None] + offset, (n, dim)).copy()
    return arrays


class TinyObsEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class TinyPolicy(torch.nn.Module):
    """Only exercises the trainer/optimizer/checkpoint path, not Wan model quality."""
    def __init__(self, obs_encoder, action_dim, **kwargs):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.bias = torch.nn.Parameter(torch.zeros(action_dim))
        self.normalizer = LinearNormalizer()

    def get_optimizer(self, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=float(kwargs['lr']))

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, batch, training=True):
        target = batch['action'].float().mean(dim=1)
        return ((self.bias[None, :] - target) ** 2).mean()
