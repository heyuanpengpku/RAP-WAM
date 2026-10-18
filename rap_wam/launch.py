"""Build and run portable single-node or multi-node torchrun commands."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(profile, overrides, env, python=sys.executable):
    if profile not in {'robodojo', 'real_robot'}:
        raise ValueError('Unknown training profile')
    nodes = int(env.get('NNODES', '1'))
    gpus = int(env.get('NPROC_PER_NODE', '1'))
    rank = int(env.get('NODE_RANK', '0'))
    if nodes < 1 or gpus < 1 or not 0 <= rank < nodes:
        raise ValueError('Require NNODES >= 1, NPROC_PER_NODE >= 1 and 0 <= NODE_RANK < NNODES')
    batch = int(env.get('BATCH_SIZE', '2'))
    accum = int(env.get('GRAD_ACCUM_STEPS', '1'))
    for value in overrides:
        if value.startswith('-') or '=' not in value:
            raise ValueError('Launcher arguments must be Hydra key=value overrides; use --dry-run to inspect')
        key, text = value.split('=', 1)
        if key.lstrip('+~') == 'hydra.run.dir':
            raise ValueError('Set RUN_DIR instead of overriding hydra.run.dir')
        if key == 'train_dataloader.batch_size':
            batch = int(text)
        elif key == 'training.gradient_accumulate_every':
            accum = int(text)
    if batch < 1 or accum < 1:
        raise ValueError('Batch size and gradient accumulation must be positive')
    cmd = [python, '-m', 'torch.distributed.run', f'--nproc_per_node={gpus}']
    if nodes == 1:
        cmd += ['--standalone']
    else:
        master = env.get('MASTER_ADDR')
        if not master or not env.get('RUN_DIR'):
            raise ValueError('Multi-node training requires common MASTER_ADDR and RUN_DIR on every node')
        port = int(env.get('MASTER_PORT', '29500'))
        if not 1 <= port <= 65535:
            raise ValueError('MASTER_PORT must be in 1..65535')
        cmd += [f'--nnodes={nodes}', f'--node_rank={rank}', f'--master_addr={master}', f'--master_port={port}']
    cmd += ['-m', 'rap_wam.train', '--config-name', profile,
            f'train_dataloader.batch_size={batch}', f'training.gradient_accumulate_every={accum}'] + list(overrides)
    return cmd, {'nodes': nodes, 'gpus_per_node': gpus, 'node_rank': rank,
                 'batch_per_gpu': batch, 'gradient_accumulation': accum,
                 'effective_global_batch': batch * accum * nodes * gpus}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=['robodojo', 'real_robot'])
    parser.add_argument('--dry-run', action='store_true')
    args, overrides = parser.parse_known_args()
    try:
        cmd, summary = build_command(args.profile, overrides, os.environ)
        for key in ['ACCELERATE_USE_DEEPSPEED', 'ACCELERATE_USE_FSDP']:
            if os.environ.get(key, '').lower() in ['1', 'true']:
                raise ValueError('This release supports ordinary DDP; unset ' + key)
    except ValueError as error:
        parser.error(str(error))
    # Validate Hydra keys and supported modes without reading data or allocating a model.
    try:
        from hydra import compose, initialize_config_module
        from rap_wam.train import register_resolvers, validate_training_options
        register_resolvers()
        composed = [f'train_dataloader.batch_size={summary["batch_per_gpu"]}',
                    f'training.gradient_accumulate_every={summary["gradient_accumulation"]}'] + overrides
        with initialize_config_module(config_module='rap_wam.configs', version_base=None):
            configuration = compose(config_name=args.profile, overrides=composed)
        validate_training_options(configuration)
    except Exception as error:
        parser.error(str(error))
    env = os.environ.copy()
    if not env.get('RUN_DIR'):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        env['RUN_DIR'] = str(Path('outputs', args.profile, stamp).resolve())
    else:
        env['RUN_DIR'] = str(Path(env['RUN_DIR']).expanduser().resolve())
    env.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    env.setdefault('HYDRA_FULL_ERROR', '1')
    summary['run_dir'] = env['RUN_DIR']
    print(json.dumps(summary, indent=2), flush=True)
    print(shlex.join(cmd), flush=True)
    if summary['batch_per_gpu'] == 1:
        print('NOTE: local batch 1 disables across-sample difficulty variation.', flush=True)
    if args.dry_run:
        return 0
    if summary['node_rank'] == 0:
        run_dir = Path(env['RUN_DIR'])
        # mkdir is atomic: two independent jobs cannot claim the same directory.
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
        (run_dir / 'launch.json').write_text(json.dumps(summary, indent=2))
    return subprocess.call(cmd, env=env)


if __name__ == '__main__':
    sys.exit(main())
