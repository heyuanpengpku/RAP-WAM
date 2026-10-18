"""Small-model contracts for the real policies; no pretrained weights or GPU."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from rap_wam.core.model.wan.runtime import WanModel
from rap_wam.core.model.common.module_attr_mixin import ModuleAttrMixin
from rap_wam.policy.rap_policy import RAPPolicy
from rap_wam.policy.language_policy import LanguageConditionedRAPPolicy


class SyntheticLatentEncoder(ModuleAttrMixin):
    """Deterministic stand-in for the external 5B VAE, used only in this test."""
    def __init__(self, dim):
        super().__init__()
        self.obs_shapes = {'rgb_head': (3, 32, 32), 'robot_state': (dim,)}
        self.target_horizon = 16; self.total_video_frames = 17; self.obs_horizon = 1
        self.image_key = 'rgb_head'; self.target_key = 'rgb_head_future'
        self.fallback_target_key = 'rgb_head_last'

    def _encode(self, frames):
        x = frames[:, ::4].permute(0, 2, 1, 3, 4).float()
        x = F.adaptive_avg_pool3d(x, (x.shape[2], 2, 2))
        return x.repeat(1, 16, 1, 1, 1)

    def prepare_condition_only(self, obs):
        prefix = self._encode(obs[self.image_key])
        b = prefix.shape[0]
        condition = torch.zeros(b, 48, 5, 2, 2)
        condition[:, :, :1] = prefix
        mask = torch.ones_like(condition); mask[:, :, :1] = 0
        return {'condition_latents': condition, 'latent_condition_mask': mask,
                'latent_shape': condition.shape, 'num_condition_latent_steps': 1}

    def prepare_batch(self, obs, training=True):
        values = self.prepare_condition_only(obs)
        values['clean_latents'] = self._encode(torch.cat([obs[self.image_key], obs[self.target_key]], dim=1))
        return values

    def decode_full_latents(self, value):
        x = F.interpolate(value[:, :3].float(), size=(17, 32, 32), mode='trilinear', align_corners=False)
        return x.permute(0, 2, 1, 3, 4).sigmoid()


def cpu_attention(q, k, v, **kwargs):
    """CPU substitute for the CUDA-only Wan cross-attention kernel."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2)).transpose(1, 2)


class PolicyContractTests(unittest.TestCase):
    def run_policy(self, dim, language):
        torch.manual_seed(11)
        backbone = WanModel(model_type='ti2v', patch_size=(1, 2, 2), text_len=4,
                            in_dim=48, out_dim=48, dim=32, ffn_dim=64,
                            freq_dim=16, text_dim=8, num_heads=4, num_layers=1).bfloat16()
        # Wan's production head expects FP32 conditioning from a CUDA-only
        # precision guard; make that interface explicit in this CPU-only test.
        backbone.head.register_forward_pre_hook(lambda module, args: (args[0], args[1].float()))
        options = dict(obs_encoder=SyntheticLatentEncoder(dim), wan_checkpoint_dir='.',
                       wan_dtype='bf16', action_dim=dim, action_horizon=16,
                       num_train_timesteps=8, num_inference_steps=1,
                       state_hidden_dim=32, action_hidden_dim=32,
                       num_latent_queries=4, align_dim=8,
                       decouple_action_from_video=True, ccrt_lambda_max=1.0,
                       ccrt_warmup_steps=0, ccrt_p_max=1.0, ccrt_sample_steps=1,
                       opsr_p_max=.5, opsr_warmup_steps=0, opsr_difficulty_gamma=1.,
                       puga_weight=.5, puga_act_weight=.3, puga_p=1.0,
                       puga_warmup_steps=0, latent_cache=False)
        with tempfile.TemporaryDirectory(prefix='rap_wam_policy_') as tmp:
            cls = RAPPolicy
            if language:
                path = Path(tmp) / 'context.pt'
                torch.save({'tasks': ['a', 'b'], 'embedding_table': torch.randn(2, 4, 8)}, path)
                options['task_ctx_path'] = str(path)
                cls = LanguageConditionedRAPPolicy
            (Path(tmp) / 'config.json').write_text('{}')
            options['wan_checkpoint_dir'] = tmp
            with patch.object(WanModel, 'from_pretrained', return_value=backbone), \
                 patch('wan.modules.model.flash_attention', cpu_attention):
                policy = cls(**options)
                # CPU LayerNorm requires FP32 affine parameters for FP32 residuals.
                for layer in policy.modules():
                    if isinstance(layer, torch.nn.LayerNorm):
                        layer.float()
                batch = {'obs': {'rgb_head': torch.rand(2, 1, 3, 32, 32),
                                  'rgb_head_future': torch.rand(2, 16, 3, 32, 32),
                                  'robot_state': torch.randn(2, 1, dim)},
                         'obs2': {'rgb_head_future': torch.rand(2, 16, 3, 32, 32),
                                  'robot_state': torch.randn(2, 1, dim)},
                         'action': torch.randn(2, 16, dim), 'action_2': torch.randn(2, 16, dim),
                         'task_index': torch.tensor([0, 1])}
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = policy(batch, training=True)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(policy.get_last_loss_components()['ccrt_applied'], 1.)
                self.assertIn('puga_loss', policy.get_last_loss_components())
                loss.backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                    and p.grad.abs().sum() > 0 for p in policy.model.parameters()))

    def test_real_robot_policy_contract(self):
        self.run_policy(28, False)

    def test_robodojo_policy_contract(self):
        self.run_policy(14, True)


if __name__ == '__main__':
    unittest.main()
