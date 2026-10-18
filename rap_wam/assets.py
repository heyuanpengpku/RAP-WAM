"""Resolve the same Wan transformer directory for loading and preflight."""
from pathlib import Path


def transformer_directory(checkpoint_dir, subfolder=None):
    root = Path(checkpoint_dir).expanduser().resolve()
    if subfolder is not None:
        target = root / subfolder
        if not (target / 'config.json').is_file():
            raise FileNotFoundError(f'Wan transformer config.json is missing in {target}')
        return target
    for target in (root / 'transformer', root / 'model', root):
        if (target / 'config.json').is_file():
            return target
    raise FileNotFoundError(f'Wan transformer config.json is missing under {root}')
