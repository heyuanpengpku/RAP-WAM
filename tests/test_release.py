import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from rap_wam.train import register_resolvers
from rap_wam.datasets import RoboDojoDataset, RealRobotDataset
from rap_wam.prepare_data import write_shards, robodojo_episodes
from rap_wam.preflight import validate_shards
from rap_wam.launch import build_command
from tests.helpers import episode

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    register_resolvers()
    with initialize_config_module(config_module='rap_wam.configs', version_base=None):
        return compose(config_name=name)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='rap wam test ')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_release_check_ignores_local_runtime_artifacts(self):
        from rap_wam.check_release import check
        release = self.root / 'release'
        release.mkdir()
        (release / 'pyproject.toml').write_text('[project]\nname = "rap_wam"\n')
        for folder in ['.venv', 'assets', 'data', 'outputs', '.cache']:
            target = release / folder
            target.mkdir()
            (target / 'generated.py').write_text('This is not Python syntax!')
        self.assertEqual(check(release)['errors'], [])
        (release / 'broken.py').write_text('This is not Python syntax!')
        self.assertTrue(check(release)['errors'])

    def test_all_project_modules_import_from_this_release(self):
        import pkgutil
        import rap_wam
        for info in pkgutil.walk_packages(rap_wam.__path__, prefix='rap_wam.'):
            with self.subTest(module=info.name):
                module = importlib.import_module(info.name)
                Path(module.__file__).resolve().relative_to(ROOT.resolve())
        from rap_wam.core.model.wan import runtime
        self.assertEqual(runtime._repo_root(), ROOT)

    def test_all_configuration_targets_resolve(self):
        def targets(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == '_target_':
                        yield child
                    yield from targets(child)
            elif isinstance(value, list):
                for child in value:
                    yield from targets(child)
        import hydra
        for profile in ['robodojo', 'real_robot']:
            for target in targets(OmegaConf.to_container(config(profile), resolve=False)):
                with self.subTest(profile=profile, target=target):
                    hydra.utils.get_class(target)

    def test_config_modes_and_warmup(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '96'}):
            for profile in ['robodojo', 'real_robot']:
                cfg = config(profile)
                OmegaConf.resolve(cfg)
                self.assertEqual(cfg.optimizer.dit_mode, 'full_train')
                self.assertEqual(cfg.train_dataloader.batch_size, 2)
                self.assertEqual(cfg.training.gradient_accumulate_every, 1)
                self.assertEqual(cfg.training.lr_warmup_steps, 500 * 96)
                self.assertFalse(cfg.policy.latent_cache)
                self.assertTrue(cfg.policy.strict_auxiliary_errors)
                self.assertEqual(cfg.policy.uprime_grid, 14)
                self.assertFalse(cfg.policy.uprime_use_gap)

    def test_imports_do_not_initialize_cuda(self):
        code = "import torch; from rap_wam.workspace import TrainWorkspace; from rap_wam.policy.language_policy import LanguageConditionedRAPPolicy; assert not torch.cuda.is_initialized(); print('CPU_IMPORT_OK')"
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
        x = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(x.returncode, 0, x.stderr)
        self.assertIn('CPU_IMPORT_OK', x.stdout)

    def test_real_robot_windows_and_no_validation(self):
        path = self.root / 'real'
        write_shards([(episode('real_robot'), None), (episode('real_robot', offset=100), None)],
                     path, 'real_robot', episodes_per_shard=1, expected_episodes=2)
        cfg = config('real_robot')
        kwargs = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        kwargs.pop('_target_'); kwargs.update(data_root=str(path), expected_episode_count=2,
                                             expected_total_frames=80, expected_train_samples=16)
        train = RealRobotDataset(**kwargs, split='train')
        val = RealRobotDataset(**kwargs, split='val')
        self.assertEqual(len(train), 16)
        self.assertEqual(len(val), 0)
        batch = train[0]
        self.assertEqual(tuple(batch['action'].shape), (16, 28))
        self.assertEqual(tuple(batch['action_2'].shape), (16, 28))
        self.assertEqual(tuple(batch['obs']['robot_state'].shape), (1, 28))
        self.assertEqual(float(batch['action'][0, 0]), 1.)
        self.assertEqual(float(batch['action_2'][0, 0]), 17.)

    def test_robodojo_split_and_language(self):
        path = self.root / 'robo'
        write_shards([(episode('robodojo'), 0), (episode('robodojo'), 1)], path,
                     'robodojo', expected_episodes=2, tasks=['a', 'b'])
        cfg = config('robodojo')
        kwargs = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        kwargs.pop('_target_'); kwargs.update(data_root=str(path), num_tasks=2, val_split_ratio=.5)
        train, val = RoboDojoDataset(**kwargs, split='train'), RoboDojoDataset(**kwargs, split='val')
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 8)
        self.assertNotEqual(train._samples[0][:2], val._samples[0][:2])
        batch = train[0]
        self.assertEqual(tuple(batch['action_2'].shape), (16, 14))
        self.assertIn(int(batch['task_index']), [0, 1])
        shard = next(path.glob('*.zarr'))
        z = zarr.open(str(shard), 'a'); del z['meta/task_index']
        with self.assertRaises((ValueError, KeyError)):
            RoboDojoDataset(**kwargs, split='train')

    def test_context_order_is_checked(self):
        from rap_wam.preflight import validate_config
        path = self.root / 'language_data'
        write_shards([(episode('robodojo'), 0), (episode('robodojo'), 1)], path,
                     'robodojo', tasks=['a', 'b'])
        cfg = config('robodojo')
        cfg.task.dataset.data_root = str(path); cfg.task.dataset.num_tasks = 2
        cfg.policy.task_ctx_path = str(self.root / 'context.pt')
        table = torch.zeros(2, 512, 4096)
        torch.save({'tasks': ['a', 'b'], 'embedding_table': table}, cfg.policy.task_ctx_path)
        validate_config(cfg, check_weights=False)
        torch.save({'tasks': ['b', 'a'], 'embedding_table': table}, cfg.policy.task_ctx_path)
        with self.assertRaises(ValueError):
            validate_config(cfg, check_weights=False)

    def test_context_lookup_uses_the_batch_labels(self):
        from rap_wam.policy.language_policy import LanguageConditionedRAPPolicy
        policy = LanguageConditionedRAPPolicy.__new__(LanguageConditionedRAPPolicy)
        torch.nn.Module.__init__(policy)
        policy.model = torch.nn.Module()
        policy.model.text_embedding = torch.nn.Sequential(torch.nn.Linear(8, 8))
        table = torch.arange(64).reshape(2, 4, 8).float()
        policy.register_buffer('_task_ctx', table, persistent=False)
        policy._cur_task_index = torch.tensor([1, 0])
        contexts = policy._build_context_list(2)
        self.assertTrue(torch.equal(contexts[0], table[1]))
        self.assertTrue(torch.equal(contexts[1], table[0]))

    def test_bad_data_is_not_published(self):
        path = self.root / 'invalid'
        bad = episode('real_robot')
        bad['action_arm_joints'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            write_shards([(bad, None)], path, 'real_robot')
        self.assertFalse(path.exists())
        with self.assertRaises(ValueError):
            write_shards([(episode('real_robot'), None)], path, 'real_robot', expected_episodes=100)
        self.assertFalse(path.exists())

    def test_launch_paths_and_batch_accounting(self):
        cmd, result = build_command('robodojo', ['train_dataloader.batch_size=1'],
                                    {'NNODES': '2', 'NPROC_PER_NODE': '8', 'NODE_RANK': '1',
                                     'MASTER_ADDR': '127.0.0.1', 'RUN_DIR': str(self.root / 'run')})
        self.assertEqual(result['effective_global_batch'], 16)
        self.assertIn('--node_rank=1', cmd)
        with self.assertRaises(ValueError):
            build_command('real_robot', [], {'NNODES': '2'})
        env = os.environ.copy(); env['PYTHON_BIN'] = sys.executable
        x = subprocess.run(['bash', str(ROOT / 'scripts/train_real_robot.sh'), '--dry-run'],
                           cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(x.returncode, 0, x.stderr)
        self.assertIn('effective_global_batch', x.stdout)
        self.assertFalse((self.root / 'outputs').exists())

    def test_real_trainer_update_and_checkpoint(self):
        from rap_wam.workspace import TrainWorkspace
        path = self.root / 'train_data'
        write_shards([(episode('real_robot'), None), (episode('real_robot', offset=100), None)],
                     path, 'real_robot', expected_episodes=2)
        cfg = config('real_robot')
        cfg.task.dataset.data_root = str(path); cfg.task.dataset.expected_episode_count = 2
        cfg.policy._target_ = 'tests.helpers.TinyPolicy'
        cfg.policy.obs_encoder._target_ = 'tests.helpers.TinyObsEncoder'
        cfg.training.mixed_precision = 'no'; cfg.training.num_epochs = 1
        cfg.training.max_train_steps = 1; cfg.training.optimizer_warmup_steps = 0
        cfg.train_dataloader.num_workers = 0; cfg.val_dataloader.num_workers = 0
        cfg.optimizer.lr = .1
        OmegaConf.resolve(cfg)
        run = self.root / 'run'; run.mkdir()
        workspace = TrainWorkspace(cfg, output_dir=str(run))
        before = workspace.model.bias.detach().clone()
        workspace.run()
        self.assertEqual(workspace.global_step, 1)
        self.assertFalse(torch.equal(workspace.model.bias.detach(), before))
        ckpt = run / 'checkpoints/latest.ckpt'
        self.assertTrue(ckpt.is_file())
        import dill
        saved = torch.load(ckpt, map_location='cpu', pickle_module=dill)
        self.assertIn('optimizer', saved['state_dicts'])
        self.assertIn('bias', saved['state_dicts']['model'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools unavailable')
    def test_lerobot_video_parquet_conversion(self):
        import pandas as pd
        source = self.root / 'lerobot'; (source / 'meta').mkdir(parents=True)
        (source / 'meta/info.json').write_text(json.dumps({'chunks_size': 1000}))
        (source / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'tasks': ['b']}) + '\n')
        data = source / 'data/chunk-000'; data.mkdir(parents=True)
        pd.DataFrame({'action': [np.zeros(14, np.float32)] * 40,
                      'observation.state': [np.zeros(14, np.float32)] * 40}).to_parquet(data / 'episode_000000.parquet')
        from rap_wam.prepare_data import CAMERAS
        for cam in CAMERAS:
            folder = source / 'videos/chunk-000' / cam; folder.mkdir(parents=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=red:s=16x16:r=10',
                            '-frames:v', '40', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(folder / 'episode_000000.mp4')], check=True)
        output = self.root / 'converted'
        report = write_shards(robodojo_episodes(source, ['a', 'b']), output, 'robodojo', tasks=['a', 'b'])
        self.assertEqual(report['frames'], 40)
        z = zarr.open(str(next(output.glob('*.zarr'))), 'r')
        self.assertEqual(int(z['meta/task_index'][0]), 1)
        self.assertEqual(tuple(z['data/rgb_head'].shape), (40, 384, 320, 3))


if __name__ == '__main__':
    unittest.main()
