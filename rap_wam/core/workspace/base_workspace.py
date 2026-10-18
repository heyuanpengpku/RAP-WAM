from __future__ import annotations

from typing import Any, Optional

import dill
import os
import tempfile
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir

    @property
    def output_dir(self) -> str:
        if self._output_dir is not None:
            return self._output_dir
        try:
            return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        except Exception:
            return str(Path.cwd())

    def get_checkpoint_path(self, tag: str = "latest") -> Path:
        return Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"

    def get_checkpoint_cfg(self) -> OmegaConf:
        """Return the configuration stored in a checkpoint; subclasses may provide an inference configuration."""
        return self.cfg

    def save_checkpoint(self, path: str | Path | None = None, tag: str = "latest"):
        if path is None:
            ckpt_path = self.get_checkpoint_path(tag=tag)
        else:
            ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        include_keys = tuple(self.include_keys)
        exclude_keys = tuple(self.exclude_keys)
        payload = {"format_version": 2, "cfg": self.get_checkpoint_cfg(), "state_dicts": {}, "pickles": {}}
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)

        descriptor, temporary = tempfile.mkstemp(prefix="." + ckpt_path.name + ".", suffix=".tmp", dir=ckpt_path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(payload, handle, pickle_module=dill)
            os.replace(temporary, ckpt_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return str(ckpt_path)

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a dictionary")
        state_dicts = payload.get("state_dicts")
        pickles = payload.get("pickles")
        if not isinstance(state_dicts, dict):
            raise ValueError("checkpoint.state_dicts must be a dictionary")
        if not isinstance(pickles, dict):
            raise ValueError("checkpoint.pickles must be a dictionary")

        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = tuple(key for key in pickles if key != "_output_dir")

        for key, value in state_dicts.items():
            if key in exclude_keys:
                continue
            if key not in self.__dict__:
                print(f"[checkpoint] skipping {key} (the current workspace has no such attribute)")
                continue
            target = self.__dict__[key]
            if target is None or not hasattr(target, "load_state_dict"):
                continue
            if kwargs:
                target.load_state_dict(value, **kwargs)
            else:
                target.load_state_dict(value)

        for key in include_keys:
            if key in pickles:
                self.__dict__[key] = dill.loads(pickles[key])

    def load_checkpoint(
        self,
        path: str | Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        map_location: str = "cpu",
    ):
        if path is None:
            ckpt_path = self.get_checkpoint_path(tag=tag)
        else:
            ckpt_path = Path(path)

        with ckpt_path.open("rb") as f:
            payload = torch.load(f, map_location=map_location, pickle_module=dill)
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
        )
        return payload

    def run(self):
        raise NotImplementedError
