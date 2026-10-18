from __future__ import annotations

import copy
import inspect
from typing import Any, Mapping, Sequence

import hydra

from rap_wam.core.dataset.preprocessors.base import BaseObsPreprocessor
from rap_wam.core.dataset.preprocessors.passthrough import PassthroughPreprocessor


def build_obs_preprocessors(
    cfg: Any,
    *,
    obs_keys: Sequence[str],
    default_image_size: Sequence[int] | None = None,
) -> dict[str, BaseObsPreprocessor]:
    known_keys = _clean_keys(obs_keys)
    raw_cfg = _as_mapping_or_empty(cfg, field_name="dataset.preprocessors")

    unknown = sorted(key for key in raw_cfg.keys() if str(key).strip() not in set(known_keys))
    if unknown:
        raise ValueError("dataset.preprocessors contains unknown observation keys: " + ", ".join(unknown))

    preprocessors: dict[str, BaseObsPreprocessor] = {}
    for key in known_keys:
        raw_item = raw_cfg.get(key)
        if raw_item is None:
            preprocessors[key] = PassthroughPreprocessor()
            continue
        preprocessors[key] = _instantiate_preprocessor(
            raw_item,
            default_image_size=default_image_size,
            field_name=f"dataset.preprocessors[{key}]",
        )
    return preprocessors


def remap_preprocessor_cfgs(
    cfg: Any,
    *,
    selected_obs_keys: Sequence[str],
    alias_map: Mapping[str, str],
) -> dict[str, Any]:
    raw_cfg = _as_mapping_or_empty(cfg, field_name="dataset.preprocessors")
    remapped: dict[str, Any] = {}
    for canonical_key in _clean_keys(selected_obs_keys):
        if canonical_key not in raw_cfg:
            continue
        source_key = str(alias_map.get(canonical_key, canonical_key)).strip()
        if not source_key:
            raise ValueError(f"dataset.preprocessors[{canonical_key}] resolves to an empty source key")
        remapped[source_key] = copy.deepcopy(raw_cfg[canonical_key])
    return remapped


def _instantiate_preprocessor(
    raw_item: Any,
    *,
    default_image_size: Sequence[int] | None,
    field_name: str,
) -> BaseObsPreprocessor:
    if isinstance(raw_item, BaseObsPreprocessor):
        return raw_item

    extra = {}
    # Only image preprocessors which accept this argument should receive it.
    # Hydra-recursively-created instances are handled above; direct dict input
    # must work identically for low-dimensional preprocessors with no __init__.
    if default_image_size is not None and isinstance(raw_item, Mapping):
        target = raw_item.get("_target_")
        if isinstance(target, str):
            target = hydra.utils.get_class(target)
        if callable(target):
            parameters = inspect.signature(target).parameters
            if "default_image_size" in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            ):
                extra["default_image_size"] = tuple(default_image_size)
    instance = hydra.utils.instantiate(raw_item, **extra)
    if not isinstance(instance, BaseObsPreprocessor):
        raise TypeError(f"{field_name} invalid preprocessor type: {type(instance)}")
    return instance


def _as_mapping_or_empty(raw: Any, *, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(key).strip(): value for key, value in raw.items()}
    if hasattr(raw, "items") and callable(getattr(raw, "items")):
        try:
            return {str(key).strip(): value for key, value in raw.items()}  # type: ignore[call-arg]
        except Exception as exc:
            raise TypeError(f"{field_name} must be a mapping") from exc
    raise TypeError(f"{field_name} must be a mapping")


def _clean_keys(values: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for raw in values:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


__all__ = ["build_obs_preprocessors", "remap_preprocessor_cfgs"]
