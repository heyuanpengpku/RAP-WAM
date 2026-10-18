"""Prepare sharded training Zarr from RoboDojo LeRobot-v2.1 or real-robot NPZ."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import numpy as np
from rap_wam.task_vocabulary import load_task_strings, validate_task_strings

STATE_KEYS = {'waist_joints_states': 2, 'left_arm_states': 7, 'right_arm_states': 7,
              'left_hand_states': 6, 'right_hand_states': 6}
ACTION_KEYS = {'action_waist_joints': 2, 'action_arm_joints': 14,
               'action_left_hand_joints': 6, 'action_right_hand_joints': 6}
CAMERAS = ['observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist']


def build_tshape(head, left, right):
    import cv2
    top = cv2.resize(head, (320, 256), interpolation=cv2.INTER_AREA)
    left = cv2.resize(left, (160, 128), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (160, 128), interpolation=cv2.INTER_AREA)
    return np.concatenate([top, np.concatenate([left, right], axis=1)], axis=0)


def read_video(path):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                       '-show_entries', 'stream=width,height', '-of', 'json', str(path)]))
    stream = info['streams'][0]
    h, w = int(stream['height']), int(stream['width'])
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-pix_fmt', 'rgb24',
                             '-f', 'rawvideo', '-'], capture_output=True)
    if result.returncode or not result.stdout or len(result.stdout) % (h * w * 3):
        raise ValueError(f'Video decode failed or partial frame: {path}')
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, h, w, 3)


def robodojo_episodes(source, tasks, limit=0, truncate=False):
    import pandas as pd
    source = Path(source)
    tasks = validate_task_strings(tasks)
    lookup = {task: i for i, task in enumerate(tasks)}
    info = json.loads((source / 'meta/info.json').read_text())
    chunks = int(info.get('chunks_size', 1000))
    episodes = [json.loads(line) for line in (source / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
    if any(not isinstance(record.get('episode_index'), int) or isinstance(record.get('episode_index'), bool) for record in episodes):
        raise ValueError('Episode indices must be integers')
    episodes.sort(key=lambda x: int(x['episode_index']))
    indices = [int(record['episode_index']) for record in episodes]
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError('Episode indices must be unique nonnegative integers')
    if chunks < 1:
        raise ValueError('chunks_size must be positive')
    if limit:
        episodes = episodes[:limit]
    for record in episodes:
        ep = int(record['episode_index'])
        instructions = record.get('tasks', [])
        if len(instructions) != 1 or instructions[0] not in lookup:
            raise ValueError(f'Episode {ep} must have one known task string')
        chunk = f'chunk-{ep // chunks:03d}'
        frame = pd.read_parquet(source / 'data' / chunk / f'episode_{ep:06d}.parquet')
        action = np.stack(frame['action'].values).astype(np.float32)
        state = np.stack(frame['observation.state'].values).astype(np.float32)
        videos = [read_video(source / 'videos' / chunk / cam / f'episode_{ep:06d}.mp4') for cam in CAMERAS]
        lengths = [len(action), len(state)] + [len(v) for v in videos]
        if len(set(lengths)) != 1 and not truncate:
            raise ValueError(f'Frame-count mismatch in episode {ep}: {lengths}; explicit --truncate-to-shortest is available')
        n = min(lengths)
        rgb = np.stack([build_tshape(*(v[t] for v in videos)) for t in range(n)])
        yield {'rgb_head': rgb, 'action_joint': action[:n], 'robot_state_joint': state[:n]}, lookup[instructions[0]]


def real_robot_episodes(source, limit=0):
    paths = sorted(Path(source).glob('*.npz'))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError('No episode NPZ files found')
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            keys = ['rgb_head', *STATE_KEYS, *ACTION_KEYS]
            yield {key: np.array(episode[key]) for key in keys}, None


def write_shards(episodes, output, profile, episodes_per_shard=50, expected_episodes=None, tasks=None):
    if profile not in {'robodojo', 'real_robot'}:
        raise ValueError('Unknown training profile')
    if expected_episodes is not None and int(expected_episodes) < 1:
        raise ValueError('expected_episodes must be positive')
    if profile == 'robodojo':
        tasks = validate_task_strings(tasks)
    import zarr
    from numcodecs import Blosc
    from rap_wam.preflight import validate_shards
    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output must be a new directory: {output}')
    if episodes_per_shard < 1:
        raise ValueError('episodes_per_shard must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name('.' + output.name + '.partial-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        group = None
        count = 0
        ends, labels = [], []
        current_frames = 0
        for arrays, label in episodes:
            required = {"rgb_head", "action_joint", "robot_state_joint"} if profile == "robodojo" else {"rgb_head", *STATE_KEYS, *ACTION_KEYS}
            if set(arrays) != required:
                raise ValueError("Episode arrays must contain exactly the documented profile keys")
            rgb = arrays['rgb_head']
            size = (384, 320) if profile == 'robodojo' else (224, 224)
            n = len(rgb)
            if n < 33 or rgb.dtype != np.uint8 or rgb.shape != (n, *size, 3):
                raise ValueError('Every episode needs at least 33 RGB uint8 frames at the recipe resolution')
            dims = {'action_joint': 14, 'robot_state_joint': 14} if profile == 'robodojo' else {**STATE_KEYS, **ACTION_KEYS}
            for key, dim in dims.items():
                if arrays[key].shape != (n, dim) or not np.isfinite(arrays[key]).all():
                    raise ValueError(f'Invalid episode {key}: expected finite [{n},{dim}]')
            if count % episodes_per_shard == 0:
                group = zarr.open(str(stage / f'shard_{count // episodes_per_shard:04d}.zarr'), mode='w')
                group.require_group('data'); group.require_group('meta')
                ends, labels, current_frames = [], [], 0
                for key, arr in arrays.items():
                    dtype = 'u1' if key == 'rgb_head' else 'f4'
                    group.create_dataset('data/' + key, shape=(0, *arr.shape[1:]),
                                         chunks=(32 if key == 'rgb_head' else 4096, *arr.shape[1:]),
                                         dtype=dtype, compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE))
            for key, arr in arrays.items():
                group['data/' + key].append(arr.astype(group['data/' + key].dtype, copy=False), axis=0)
            current_frames += n
            ends.append(current_frames)
            group.create_dataset('meta/episode_ends', data=np.asarray(ends, np.int64), overwrite=True)
            if profile == 'robodojo':
                if not isinstance(label, (int, np.integer)) or not 0 <= int(label) < len(tasks):
                    raise ValueError('RoboDojo task index must be an integer in the vocabulary range')
                labels.append(int(label))
                group.create_dataset('meta/task_index', data=np.asarray(labels, np.int64), overwrite=True)
            count += 1
        report = validate_shards(stage, profile, expected_episodes, len(tasks) if tasks else 35, full=True)
        if tasks is not None:
            (stage / 'task_strings.json').write_text(json.dumps(tasks, indent=2))
        (stage / 'manifest.json').write_text(json.dumps(report, indent=2))
        os.rename(stage, output)
        return report
    except Exception:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=['robodojo', 'real_robot'])
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes-per-shard', type=int, default=50)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--expected-episodes', type=int)
    parser.add_argument('--tasks', help='JSON vocabulary for RoboDojo; defaults to the archived 35 task strings')
    parser.add_argument('--truncate-to-shortest', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    tasks = None
    if args.profile == 'robodojo':
        tasks = load_task_strings(args.tasks or Path(__file__).with_name('task_strings.json'))
        episodes = robodojo_episodes(args.source, tasks, args.limit, args.truncate_to_shortest)
    else:
        episodes = real_robot_episodes(args.source, args.limit)
    expected = args.expected_episodes
    if args.profile == 'real_robot' and expected is None:
        expected = 100
    report = write_shards(episodes, args.out, args.profile, args.episodes_per_shard, expected, tasks)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
