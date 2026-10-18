from __future__ import annotations

import copy
import json
import numbers
from typing import Any, Callable, Optional


class JsonLogger:
    def __init__(self, path: str, filter_fn: Optional[Callable[[str, Any], bool]] = None):
        if filter_fn is None:
            filter_fn = lambda k, v: isinstance(v, numbers.Number)
        self.path = path
        self.filter_fn = filter_fn
        self.file = None
        self.last_log = None

    def __enter__(self):
        self.file = open(self.path, "a", buffering=1, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()
            self.file = None

    def log(self, data: dict):
        filtered_data = dict(filter(lambda x: self.filter_fn(*x), data.items()))
        self.last_log = filtered_data
        for k, v in filtered_data.items():
            if isinstance(v, numbers.Integral):
                filtered_data[k] = int(v)
            elif isinstance(v, numbers.Number):
                filtered_data[k] = float(v)
        buf = json.dumps(filtered_data, ensure_ascii=False)
        self.file.write(buf + "\n")

    def get_last_log(self):
        return copy.deepcopy(self.last_log)
