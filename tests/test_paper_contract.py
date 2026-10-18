"""Numerical paper contracts and explicit auxiliary failure handling."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from rap_wam.core.model.common.module_attr_mixin import ModuleAttrMixin
from rap_wam.policy.rap_policy import RAPPolicy
from rap_wam.policy.residual_policy import ResidualActionPolicy


def small_policy():
    """Construct only the loss/diagnostic shell; no backbone or model weights."""
    policy = RAPPolicy.__new__(RAPPolicy)
    ModuleAttrMixin.__init__(policy)
    policy.strict_auxiliary_errors = True
    policy._auxiliary_counts = {name: dict.fromkeys(
        ('attempted', 'succeeded', 'applied', 'failed'), 0)
        for name in ('rehearsal', 'credit')}
    policy._last_loss_components = {}
    policy.ccrt_lambda_max = 1.; policy.ccrt_p_max = 1.
    policy.ccrt_warmup_steps = 0; policy._ccrt_step = 0
    policy.puga_weight = .5; policy.puga_act_weight = .3
    policy.puga_warmup_steps = 0; policy.puga_p = 1.; policy._puga_step = 0
    policy._latcache_init = False; policy._latent_cache_default = False
    policy.puga_eps = 1e-6
    return policy


def base_loss(policy, batch, training=True):
    policy._last_loss_components = {'total_loss': 1.}
    return torch.tensor(1., requires_grad=True)


class PaperContractTests(unittest.TestCase):
    def test_visual_temporal_credit_and_detached_probe(self):
        policy = small_policy()
        policy.use_future_only_loss = True
        policy.puga_w_unc = 1.; policy.puga_w_lev = 1.; policy.puga_w_diff = .5
        policy.puga_act_delta = .1; policy.uprime_use_gap = False
        policy.train_timesteps = torch.tensor([0.])
        policy.normalizer = SimpleNamespace(params_dict={})
        policy._normalize_non_image_obs = lambda value: value
        clean = torch.zeros(1, 1, 5, 1, 2)
        policy.obs_encoder = SimpleNamespace(prepare_batch=lambda *args, **kwargs: {
            'clean_latents': clean, 'condition_latents': clean.clone(),
            'latent_condition_mask': torch.ones_like(clean),
            'num_condition_latent_steps': 1})
        policy.register_buffer('_uprime_ema', torch.tensor([[0., 2.]]))
        reference = torch.arange(1., 9.).reshape(1, 1, 4, 1, 2).requires_grad_()
        prefix = torch.zeros(1, 1, 1, 1, 2)
        response = torch.tensor([[1., 3.], [2., 6.], [5., 7.], [4., 8.]])
        perturbed = (reference.detach() + response.sqrt().reshape_as(reference)).requires_grad_()
        actions = torch.arange(1., 17.).reshape(1, 16, 1).requires_grad_()
        noisy_actions = torch.zeros_like(actions)
        calls = []
        def forward(video, action, timestep, obs, batch_size):
            calls.append((video.clone(), action.clone(), timestep.clone(), torch.is_grad_enabled()))
            if len(calls) == 1:
                return torch.cat([prefix, reference], dim=2), actions, None
            return torch.cat([prefix, perturbed], dim=2), actions.detach(), None
        noise = (clean.clone(), noisy_actions, torch.zeros_like(clean), torch.zeros_like(actions))
        with patch.object(policy, '_puga_noise', return_value=noise), \
             patch.object(policy, '_puga_forward_video', side_effect=forward):
            loss = policy._puga_loss({'obs': {}, 'action': torch.zeros_like(actions)})
        errors = torch.arange(1., 9.).square().reshape(4, 2)
        visual = 1 + torch.tensor([0., 1.]) + (response - 1) / 7 + .5 * (errors - 1) / 63
        # Mean raw responses are 2, 4, 6, 6. Each temporal coefficient spans 4 actions.
        temporal = torch.tensor([1., 1.5, 2., 2.]).repeat_interleave(4)
        expected = .5 * (visual * errors).sum() / visual.sum()
        expected += .3 * (temporal * torch.arange(1., 17.).square()).sum() / temporal.sum()
        torch.testing.assert_close(loss, expected)
        self.assertEqual([call[3] for call in calls], [True, False])
        torch.testing.assert_close(calls[0][0], calls[1][0])
        torch.testing.assert_close(calls[0][2], calls[1][2])
        self.assertFalse(torch.equal(calls[0][1], calls[1][1]))
        loss.backward()
        expected_grad = (reference.detach().reshape(4, 2) * visual / visual.sum()).reshape_as(reference)
        torch.testing.assert_close(reference.grad, expected_grad)
        self.assertIsNone(perturbed.grad)
        self.assertIsNotNone(actions.grad)

    def test_prior_initialization_ema_and_constant_normalization(self):
        policy = small_policy(); policy.uprime_ema_decay = .9; policy._uprime_ready = False
        policy.register_buffer('_uprime_ema', torch.zeros(2, 2))
        policy._update_uprime(torch.full((2, 4, 2, 2), 2.))
        torch.testing.assert_close(policy._uprime_ema, torch.full((2, 2), 2.))
        policy._update_uprime(torch.full((2, 4, 2, 2), 4.))
        torch.testing.assert_close(policy._uprime_ema, torch.full((2, 2), 2.2))
        torch.testing.assert_close(policy._puga_norm01(torch.ones(2, 4)), torch.zeros(2, 4))
        # A time-constant spatial prior cannot contribute temporal variation.
        prior = torch.tensor([[0., 2.], [3., 1.]])[None, None].expand(2, 4, -1, -1)
        torch.testing.assert_close(policy._puga_norm01(prior.mean((2, 3))), torch.zeros(2, 4))

    def test_auxiliary_exceptions_stop_training_by_default(self):
        for branch in ('rehearsal', 'credit'):
            policy = small_policy()
            if branch == 'credit':policy.ccrt_lambda_max = 0.
            method = '_ccrt_chunk1_loss' if branch == 'rehearsal' else '_puga_loss'
            original = ValueError('invalid auxiliary input')
            with patch.object(ResidualActionPolicy, 'compute_loss', base_loss), \
                 patch.object(policy, method, side_effect=original):
                with self.assertRaisesRegex(RuntimeError, branch + ' auxiliary failed') as caught:
                    policy.compute_loss({'obs2': {}}, training=True)
            self.assertIs(caught.exception.__cause__, original)
            self.assertEqual(policy._auxiliary_counts[branch]['failed'], 1)

    def test_explicit_tolerant_mode_warns_and_logs_failed_participation(self):
        policy = small_policy(); policy.strict_auxiliary_errors = False
        with patch.object(ResidualActionPolicy, 'compute_loss', base_loss), \
             patch.object(policy, '_ccrt_chunk1_loss', side_effect=ValueError('bad rehearsal')), \
             patch.object(policy, '_puga_loss', side_effect=ValueError('bad credit')):
            with self.assertWarnsRegex(RuntimeWarning, 'auxiliary failed'):
                result = policy.compute_loss({'obs2': {}}, training=True)
        self.assertEqual(result.item(), 1.)
        for branch in ('rehearsal', 'credit'):
            self.assertEqual(policy._last_loss_components[f'{branch}_attempted'], 1.)
            self.assertEqual(policy._last_loss_components[f'{branch}_failed_total'], 1.)
            self.assertEqual(policy._last_loss_components[f'{branch}_applied_total'], 0.)

    def test_applied_loss_total_and_probability_skip_reset(self):
        policy = small_policy()
        with patch.object(ResidualActionPolicy, 'compute_loss', base_loss), \
             patch.object(policy, '_ccrt_chunk1_loss', return_value=torch.tensor(2.)), \
             patch.object(policy, '_puga_loss', return_value=torch.tensor(3.)):
            result = policy.compute_loss({'obs2': {}}, training=True)
            self.assertEqual(result.item(), 6.)
            self.assertEqual(policy._last_loss_components['total_loss'], 6.)
            for branch in ('rehearsal', 'credit'):
                self.assertEqual(policy._last_loss_components[f'{branch}_applied'], 1.)
                self.assertEqual(policy._last_loss_components[f'{branch}_applied_total'], 1.)
            policy.ccrt_p_max = 0.; policy.puga_p = 0.
            result = policy.compute_loss({'obs2': {}}, training=True)
            self.assertEqual(result.item(), 1.)
            for branch in ('rehearsal', 'credit'):
                self.assertEqual(policy._last_loss_components[f'{branch}_attempted'], 0.)
                self.assertEqual(policy._last_loss_components[f'{branch}_applied_total'], 1.)
