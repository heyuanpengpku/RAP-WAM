"""Shared validation for ordered instruction vocabularies."""
from __future__ import annotations

import json
from pathlib import Path


def validate_task_strings(value):
    if not isinstance(value, list) or not value:
        raise ValueError('Task vocabulary must be a nonempty JSON list of strings')
    if any(not isinstance(task, str) or not task.strip() for task in value):
        raise ValueError('Every task instruction must be a nonempty string')
    if len(set(value)) != len(value):
        raise ValueError('Task instructions must be unique and ordered')
    return list(value)


def load_task_strings(path):
    return validate_task_strings(json.loads(Path(path).expanduser().read_text()))
