"""Validate data, language context and optional Wan assets before model allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import zarr
from rap_wam.assets import transformer_directory
from rap_wam.task_vocabulary import load_task_strings


def validate_shards(data_root, profile, expected_episodes=None, num_tasks=35, full=False):
    if profile not in {'robodojo', 'real_robot'}:
        raise ValueError('Unknown training profile')
    if int(num_tasks) < 1 or (expected_episodes is not None and int(expected_episodes) < 1):
        raise ValueError('Task and expected episode counts must be positive')
    root = Path(data_root).expanduser().resolve()
    paths = sorted(p for p in root.glob('*.zarr') if p.is_dir())
    if not paths:
        raise ValueError(f'No Zarr shards found in {root}')
    if profile == 'robodojo':
        state = {'robot_state_joint': 14}
        action = {'action_joint': 14}
        size = (384, 320)
    else:
        state = {'waist_joints_states': 2, 'left_arm_states': 7, 'right_arm_states': 7,
                 'left_hand_states': 6, 'right_hand_states': 6}
        action = {'action_waist_joints': 2, 'action_arm_joints': 14,
                  'action_left_hand_joints': 6, 'action_right_hand_joints': 6}
        size = (224, 224)
    episodes = frames = windows = 0
    for path in paths:
        z = zarr.open(path, mode='r')
        rgb = z['data/rgb_head']
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (*size, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'RGB must be uint8 [T,{size[0]},{size[1]},3]: {path}')
        total = len(rgb)
        raw_ends = z['meta/episode_ends']
        if raw_ends.ndim != 1 or raw_ends.dtype.kind not in 'iu':
            raise ValueError(f'episode_ends must be a 1D integer array: {path}')
        ends = np.asarray(raw_ends[:], dtype=np.int64)
        lengths = np.diff(np.concatenate([[0], ends]))
        if not len(ends) or ends[-1] != total or np.any(lengths < 33):
            raise ValueError(f'Episodes must cover all frames and each contain at least 33 frames: {path}')
        for name, dim in {**state, **action}.items():
            array = z['data/' + name]
            if array.shape != (total, dim) or array.dtype != np.float32:
                raise ValueError(f'{name} must be float32 [{total},{dim}]: {path}')
            for start in range(0, total, 4096 if full else max(1, total)):
                if not np.isfinite(array[start:min(total, start + 4096)]).all():
                    raise ValueError(f'Nonfinite {name}: {path}')
        if profile == 'robodojo':
            raw_labels = z['meta/task_index']
            labels = np.asarray(raw_labels[:])
            if raw_labels.dtype.kind not in 'iu' or labels.shape != ends.shape:
                raise ValueError(f'task_index must be one integer per episode: {path}')
            if np.any(labels < 0) or np.any(labels >= int(num_tasks)):
                raise ValueError(f'task_index outside vocabulary: {path}')
        frames += total
        episodes += len(ends)
        windows += int((lengths - 32).sum())
    if expected_episodes is not None and episodes != int(expected_episodes):
        raise ValueError(f'Expected {expected_episodes} episodes, found {episodes}')
    return {'profile': profile, 'shards': len(paths), 'episodes': episodes,
            'frames': frames, 'two_chunk_windows_before_split': windows,
            'low_dim_scan': 'full' if full else 'first_4096_frames_per_shard'}


def validate_context(context_path, tasks_path, num_tasks, text_len=512, text_dim=4096):
    import torch
    tasks = load_task_strings(tasks_path)
    payload = torch.load(Path(context_path).expanduser(), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Task context must be a dictionary containing tasks and embedding_table')
    table = payload.get('embedding_table')
    if not isinstance(table, torch.Tensor) or tuple(table.shape) != (int(num_tasks), text_len, text_dim):
        raise ValueError(f'Task context must have shape [{num_tasks},{text_len},{text_dim}]')
    if not table.is_floating_point() or not torch.isfinite(table).all():
        raise ValueError('Task context must contain finite floating-point embeddings')
    if payload.get('tasks') != tasks or len(tasks) != int(num_tasks):
        raise ValueError('Task-context task order must match the conversion vocabulary exactly')
    return {'task_context': 'validated', 'task_count': len(tasks)}


def validate_wan_assets(checkpoint_dir, subfolder=None):
    root = Path(checkpoint_dir).expanduser().resolve()
    vae = root / 'Wan2.2_VAE.pth'
    if not vae.is_file() or vae.stat().st_size == 0:
        raise FileNotFoundError('A nonempty Wan2.2_VAE.pth is required')
    directory = transformer_directory(root, subfolder)
    config = json.loads((directory / 'config.json').read_text())
    if not isinstance(config, dict) or str(config.get('model_type', 'ti2v')).lower() != 'ti2v':
        raise ValueError('The public recipes require a TI2V-5B transformer configuration')
    indexes = list(directory.glob('diffusion_pytorch_model*.index.json'))
    if indexes:
        files = set()
        for index in indexes:
            mapping = json.loads(index.read_text()).get('weight_map')
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('Transformer weight index must contain a nonempty weight_map')
            files.update(mapping.values())
        for name in files:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('Transformer shard names must be local filenames')
        weights = [directory / name for name in sorted(files)]
    else:
        weights = [directory / name for name in
                   ('diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.bin')
                   if (directory / name).is_file()]
    if not weights or any(not path.is_file() or path.stat().st_size == 0 for path in weights):
        raise FileNotFoundError('Transformer weights or referenced shards are missing/empty')
    return {'wan_assets': 'config/vae/weight files present; tensor deserialization occurs at model construction',
            'transformer_weight_files': len(weights)}


def validate_config(cfg, check_weights=True):
    ds = cfg.task.dataset
    report = validate_shards(ds.data_root, str(cfg.task.name),
                             getattr(ds, 'expected_episode_count', None),
                             getattr(ds, 'num_tasks', 35),
                             full=bool(getattr(cfg.training, 'preflight_full', True)))
    if cfg.task.name == 'robodojo':
        report.update(validate_context(cfg.policy.task_ctx_path, cfg.task.task_strings_path, ds.num_tasks))
    if check_weights:
        report.update(validate_wan_assets(cfg.wan_checkpoint_dir,
                                          getattr(cfg.policy, 'transformer_subfolder', None)))
    resume = bool(getattr(cfg.training, 'resume', False))
    if resume:
        path = str(getattr(cfg.training, 'resume_path', '')).strip()
        if not path or not Path(path).expanduser().is_file():
            raise FileNotFoundError('Resuming requires an existing training.resume_path checkpoint')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=['robodojo', 'real_robot'])
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--expected-episodes', type=int)
    parser.add_argument('--num-tasks', type=int, default=35)
    parser.add_argument('--full', action='store_true', help='Scan all low-dimensional array values')
    parser.add_argument('--task-context', help='Additionally validate the RoboDojo context table')
    parser.add_argument('--tasks', help='Ordered vocabulary; defaults to DATA_ROOT/task_strings.json')
    parser.add_argument('--wan-checkpoint-dir', help='Additionally check model config, VAE and weight files')
    args = parser.parse_args()
    if (args.task_context or args.tasks) and args.profile != 'robodojo':
        parser.error('Language-context options apply only to RoboDojo')
    report = validate_shards(args.data_root, args.profile, args.expected_episodes, args.num_tasks, args.full)
    if args.task_context:
        report.update(validate_context(args.task_context,
                      args.tasks or Path(args.data_root).expanduser() / 'task_strings.json', args.num_tasks))
    elif args.tasks:
        tasks = load_task_strings(args.tasks)
        if len(tasks) != args.num_tasks:
            parser.error('Vocabulary length does not match --num-tasks')
    if args.wan_checkpoint_dir:
        report.update(validate_wan_assets(args.wan_checkpoint_dir))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
