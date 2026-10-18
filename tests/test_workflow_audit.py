"""Regression tests for complete public workflows and failure boundaries."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dill
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from rap_wam.datasets import RoboDojoDataset, RealRobotDataset
from rap_wam.prepare_data import write_shards
from rap_wam.preflight import validate_wan_assets, validate_config
from rap_wam.train import validate_training_options
from rap_wam.task_vocabulary import validate_task_strings
from rap_wam.core.workspace.base_workspace import BaseWorkspace
from rap_wam.workspace import TrainWorkspace
from rap_wam.policy.language_policy import LanguageConditionedRAPPolicy
from rap_wam.policy.rap_policy import RAPPolicy
from rap_wam.core.policy.wan_joint_action_policy import WanJointActionPolicy
from rap_wam.core.model.common.module_attr_mixin import ModuleAttrMixin
from tests.helpers import episode
from tests.test_release import config


class WorkflowAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rap_wam_audit_')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, profile, count=2, **extra):
        path = self.root / profile
        tasks = ['a', 'b'] if profile == 'robodojo' else None
        write_shards([(episode(profile, offset=1000 * index), index if tasks else None)
                      for index in range(count)], path, profile, tasks=tasks)
        cfg = config(profile)
        fields = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        fields.pop('_target_'); fields.update(data_root=str(path), **extra)
        if profile == 'robodojo':fields['num_tasks'] = 2
        else:fields['expected_episode_count'] = count
        return path, fields

    def trainer_config(self, epochs=1, accumulation=1):
        path, _ = self.dataset('real_robot')
        cfg = config('real_robot'); cfg.task.dataset.data_root = str(path)
        cfg.task.dataset.expected_episode_count = 2
        cfg.policy._target_ = 'tests.helpers.TinyPolicy'
        cfg.policy.obs_encoder._target_ = 'tests.helpers.TinyObsEncoder'
        cfg.training.mixed_precision = 'no'; cfg.training.num_epochs = epochs
        cfg.training.gradient_accumulate_every = accumulation
        cfg.training.max_train_steps = 1; cfg.training.optimizer_warmup_steps = 0
        cfg.train_dataloader.num_workers = 0; cfg.val_dataloader.num_workers = 0
        cfg.optimizer.lr = .1
        OmegaConf.resolve(cfg)
        return cfg

    def test_normalizer_and_action_scan_exclude_held_out_episodes(self):
        _, fields = self.dataset('robodojo', val_split_ratio=.5)
        train, val = RoboDojoDataset(**fields, split='train'), RoboDojoDataset(**fields, split='val')
        normalizer = train.get_normalizer(show_progress=False)
        values = train.get_all_actions()
        self.assertEqual(len(values), 40)
        torch.testing.assert_close(normalizer.get_input_stats()['action']['max'], values.max(dim=0).values)
        torch.testing.assert_close(normalizer.get_input_stats()['robot_state']['min'], values.min(dim=0).values)
        self.assertTrue(abs(values.mean() - val.get_all_actions().mean()) > 900)
        with self.assertRaisesRegex(ValueError, 'training split'):
            val.get_normalizer(show_progress=False)

    def test_metadata_rejects_incomplete_arrays_and_invalid_split(self):
        path, fields = self.dataset('real_robot')
        with self.assertRaisesRegex(ValueError, 'val_split_ratio'):
            RealRobotDataset(**{**fields, 'val_split_ratio': 1.1})
        shard = next(path.glob('*.zarr')); group = zarr.open(shard, mode='a')
        group['data/action_arm_joints'].resize(79, 14)
        with self.assertRaisesRegex(ValueError, 'length'):
            RealRobotDataset(**fields)

    def test_vocabulary_requires_a_unique_nonempty_string_list(self):
        for value in [{'a': 0}, 'abc', [], ['a', 'a'], ['a', None], [' ']]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_task_strings(value)
        self.assertEqual(validate_task_strings(['b', 'a']), ['b', 'a'])

    def test_fractional_label_does_not_publish_data(self):
        output = self.root / 'bad'
        with self.assertRaises(ValueError):
            write_shards([(episode('robodojo'), .5)], output, 'robodojo', tasks=['a'])
        self.assertFalse(output.exists())

    def test_resume_missing_checkpoint_fails_before_dataset_creation(self):
        cfg = config('real_robot'); cfg.training.resume = True
        cfg.training.resume_path = str(self.root / 'missing.ckpt')
        with patch.object(TrainWorkspace, '_build_dataset', side_effect=AssertionError('too late')):
            with self.assertRaises(FileNotFoundError):
                TrainWorkspace(cfg, output_dir=str(self.root / 'new'))

    def test_checkpoint_failure_preserves_previous_file_and_output_location(self):
        cfg = OmegaConf.create({})
        old = self.root / 'old'; old.mkdir()
        workspace = BaseWorkspace(cfg, str(old)); workspace.model = torch.nn.Linear(1, 1)
        target = old / 'latest.ckpt'; workspace.save_checkpoint(target)
        original = target.read_bytes()
        with patch('torch.save', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):workspace.save_checkpoint(target)
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(list(old.glob('*.tmp')))
        new = self.root / 'new'; new.mkdir()
        payload = {'state_dicts': {}, 'pickles': {'_output_dir': dill.dumps(str(old))}}
        restored = BaseWorkspace(cfg, str(new)); restored.load_payload(payload)
        self.assertEqual(restored.output_dir, str(new))

    def test_short_loader_flushes_accumulation_and_resume_uses_next_epoch(self):
        cfg = self.trainer_config(accumulation=2)
        run = self.root / 'run'; run.mkdir()
        workspace = TrainWorkspace(cfg, output_dir=str(run)); workspace.run()
        self.assertEqual(workspace.global_step, 1)
        self.assertTrue(workspace.model.bias.detach().abs().sum() > 0)
        checkpoint = run / 'checkpoints/latest.ckpt'
        next_cfg = copy.deepcopy(cfg); next_cfg.training.resume = True
        next_cfg.training.resume_path = str(checkpoint); next_cfg.training.num_epochs = 2
        new = self.root / 'continued'; new.mkdir()
        resumed = TrainWorkspace(next_cfg, output_dir=str(new))
        self.assertEqual(resumed.epoch, 1)
        self.assertEqual(resumed.output_dir, str(new))
        resumed.run()
        self.assertEqual(resumed.global_step, 2)
        self.assertTrue((new / 'checkpoints/latest.ckpt').is_file())

    def test_nonfinite_loss_stops_before_checkpoint_publication(self):
        cfg = self.trainer_config(); run = self.root / 'nan_run'; run.mkdir()
        workspace = TrainWorkspace(cfg, output_dir=str(run))
        with patch.object(workspace.model, 'forward', return_value=torch.tensor(float('nan'), requires_grad=True)):
            with self.assertRaises(FloatingPointError):workspace.run()
        self.assertFalse((run / 'checkpoints/latest.ckpt').exists())

    def test_launcher_rejects_mismatched_output_and_unknown_switches(self):
        from rap_wam.launch import build_command
        for override in ['hydra.run.dir=elsewhere', '--config-name=other', '--typo']:
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_command('real_robot', [override], {})

    def test_context_publication_is_atomic_and_exclusive(self):
        from rap_wam.task_context import save_context
        output = self.root / 'context.pt'
        with patch('torch.save', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):save_context({'tasks': ['a']}, output)
        self.assertFalse(output.exists())
        save_context({'tasks': ['a']}, output); before = output.read_bytes()
        with self.assertRaises(FileExistsError):save_context({'tasks': ['b']}, output)
        self.assertEqual(output.read_bytes(), before)

    def test_dark_uint8_images_keep_their_intensity(self):
        from rap_wam.core.dataset.preprocessors.rgb import RGBPreprocessor
        processor = RGBPreprocessor(image_size=(2, 2))
        value = np.ones((1, 2, 2, 3), dtype=np.uint8)
        result = processor.process(value, key='rgb_head', obs_shape=(3, 2, 2))
        torch.testing.assert_close(result, torch.full((1, 3, 2, 2), 1 / 255.))
        result = processor.process(value.astype(np.float32), key='rgb_head', obs_shape=(3, 2, 2))
        torch.testing.assert_close(result, torch.ones((1, 3, 2, 2)))

    def test_sampling_metrics_forward_the_language_label(self):
        from rap_wam.core.workspace.wan_train_workspace import WanTrainWorkspace
        class Policy(torch.nn.Module):
            def predict_visual(self, observations):
                torch.testing.assert_close(observations['task_index'], torch.tensor([1]))
                return {'visual_pred': torch.zeros(1)}
            def decode_visual_tokens(self, value):return torch.zeros(1, 16, 3, 2, 2)
        shell = WanTrainWorkspace.__new__(WanTrainWorkspace)
        batch = {'obs': {'rgb_head_future': torch.zeros(1, 16, 3, 2, 2)}, 'task_index': torch.tensor([1])}
        result = shell._run_sampling_evaluation(batch=batch, policy=Policy(), device=torch.device('cpu'))
        self.assertEqual(result['sample_visual_mse'], 0.)

    def test_source_check_rejects_unexpected_binaries_and_links(self):
        from rap_wam.check_release import check
        root = self.root / 'source'; root.mkdir()
        (root / 'pyproject.toml').write_text('[project]\nname = "rap_wam"\n')
        (root / 'unexpected.bin').write_bytes(bytes([255, 254, 253]))
        (root / 'linked.py').symlink_to(root / 'unexpected.bin')
        result = check(root)
        self.assertTrue(any('Unexpected binary' in error for error in result['errors']))
        self.assertTrue(any('Symbolic links' in error for error in result['errors']))

    def test_public_workspace_preserves_sample_checkpoint_metrics(self):
        workspace = TrainWorkspace.__new__(TrainWorkspace)
        BaseWorkspace.__init__(workspace, OmegaConf.create({}), str(self.root))
        workspace.model = torch.nn.Linear(1, 1)
        workspace._latest_sample_visual_mse = .25
        workspace._latest_sample_action_mse = None
        captured = {}
        class TopK:
            monitor_key = 'sample_visual_mse'
            def get_ckpt_path(self, values):captured.update(values)
        accelerator = type('Accelerator', (), {'is_main_process': True})()
        workspace._save_checkpoint_bundle(accelerator=accelerator, ckpt_dir=self.root,
                                          topk_manager=TopK(), monitor_payload={'train_loss': 1.})
        self.assertEqual(captured['sample_visual_mse'], .25)

    def test_production_entry_rejects_cpu_before_model_or_data_allocation(self):
        from rap_wam.train import main
        cfg = config('real_robot')
        with patch('torch.cuda.is_available', return_value=False), \
             patch('rap_wam.preflight.validate_config', side_effect=AssertionError('too late')):
            with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):main.__wrapped__(cfg)

    def test_invalid_training_options_are_rejected_without_weights(self):
        cfg = config('real_robot')
        for key, value in [('training.num_epochs', 0), ('training.max_train_steps', 0),
                           ('train_dataloader.batch_size', 0), ('policy.puga_p', 1.2)]:
            bad = copy.deepcopy(cfg); OmegaConf.update(bad, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):validate_training_options(bad)

    def test_asset_preflight_uses_loader_directory_and_checks_all_shards(self):
        root = self.root / 'assets'; transformer = root / 'transformer'; transformer.mkdir(parents=True)
        (root / 'Wan2.2_VAE.pth').write_bytes(b'fixture')
        (root / 'config.json').write_text('{"model_type":"wrong"}')
        (transformer / 'config.json').write_text('{"model_type":"ti2v"}')
        (transformer / 'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(
            {'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
        (transformer / 'part1.safetensors').write_bytes(b'fixture')
        with self.assertRaises(FileNotFoundError):validate_wan_assets(root)
        (transformer / 'part2.safetensors').write_bytes(b'fixture')
        self.assertEqual(validate_wan_assets(root)['transformer_weight_files'], 2)

    def language_shell(self):
        policy = LanguageConditionedRAPPolicy.__new__(LanguageConditionedRAPPolicy)
        ModuleAttrMixin.__init__(policy)
        policy.model = torch.nn.Module(); policy.model.text_embedding = torch.nn.Sequential(torch.nn.Linear(8, 8))
        policy.obs_encoder = type('Encoder', (), {'image_key': 'rgb_head'})()
        policy.register_buffer('_task_ctx', torch.arange(64).reshape(2, 4, 8).float(), persistent=False)
        policy._cur_task_index = None
        return policy

    def test_language_sampling_requires_and_uses_explicit_labels(self):
        policy = self.language_shell()
        obs = {'rgb_head': torch.zeros(2, 1, 3, 2, 2)}
        def sample(instance, *, obs_dict, **kwargs):
            contexts = instance._build_context_list(2)
            self.assertNotIn('task_index', obs_dict)
            torch.testing.assert_close(contexts[0], instance._task_ctx[1])
            return torch.zeros(1), torch.zeros(2, 16, 14)
        with self.assertRaisesRegex(ValueError, 'task_index'):policy.predict_action(obs)
        with patch.object(WanJointActionPolicy, '_sample_joint_predictions', sample):
            result = policy.predict_action({**obs, 'task_index': torch.tensor([1, 0])})
        self.assertEqual(result['action'].shape, (2, 16, 14))
        self.assertIsNone(policy._cur_task_index)
        for labels in [torch.tensor([-1, 0]), torch.tensor([0., 1.]), torch.tensor([0])]:
            with self.assertRaises(ValueError):policy._checked_task_index(labels, 2)

    def test_auxiliary_schedules_and_prior_readiness_survive_state_restore(self):
        def shell():
            policy = RAPPolicy.__new__(RAPPolicy); ModuleAttrMixin.__init__(policy)
            policy._ccrt_step = 0; policy._puga_step = 0; policy._uprime_ready = False
            policy._auxiliary_counts = {'credit': {'applied': 0}}
            policy.register_buffer('_uprime_ema', torch.zeros(2, 2))
            return policy
        old = shell(); old._ccrt_step = 37; old._puga_step = 39; old._uprime_ready = True
        old._uprime_ema.fill_(2.); old._auxiliary_counts['credit']['applied'] = 6
        restored = shell(); restored.load_state_dict(old.state_dict())
        self.assertEqual(restored._ccrt_step, 37); self.assertEqual(restored._puga_step, 39)
        self.assertTrue(restored._uprime_ready)
        self.assertEqual(restored._auxiliary_counts['credit']['applied'], 6)
        legacy = old.state_dict(); legacy.pop('_extra_state')
        restored = shell(); restored.load_state_dict(legacy)
        self.assertTrue(restored._uprime_ready)


if __name__ == '__main__':unittest.main()
