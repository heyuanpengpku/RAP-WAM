from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from rap_wam.core.model.wan.runtime import T5EncoderModel


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def ensure_wan_prompt_embedding(
    *,
    prompt_text: str,
    view_id: str,
    wan_checkpoint_dir: str,
    output_dir: str | Path,
    device: str,
    dtype: str = "bf16",
    text_len: int = 512,
    tokenizer_path: str = "google/umt5-xxl",
    allow_create: bool = True,
    wait_timeout_sec: float = 900.0,
    poll_interval_sec: float = 1.0,
) -> Path:
    prompt_text = str(prompt_text).strip()
    view_id = str(view_id).strip()
    if not prompt_text:
        raise ValueError("prompt_text cannot be empty")
    if not view_id:
        raise ValueError("view_id cannot be empty")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(wan_checkpoint_dir).expanduser().resolve()
    t5_checkpoint_path = checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    if not t5_checkpoint_path.exists():
        raise FileNotFoundError(f"T5 checkpoint not found: {t5_checkpoint_path}")

    cache_spec = _build_prompt_embedding_cache_spec(
        prompt_text=prompt_text,
        checkpoint_dir=checkpoint_dir,
        dtype=dtype,
        text_len=text_len,
        tokenizer_path=tokenizer_path,
    )
    cache_key = cache_spec["cache_key"]
    output_path = output_dir / f"{_sanitize_filename(cache_key)}.pt"
    legacy_output_path = output_dir / f"{_sanitize_filename(view_id)}.pt"

    existing_path = _find_existing_prompt_embedding(
        output_dir=output_dir,
        cache_spec=cache_spec,
        preferred_paths=(output_path, legacy_output_path),
    )
    if existing_path is not None:
        return existing_path

    if not allow_create:
        return _wait_for_prompt_embedding(
            output_dir=output_dir,
            paths=(output_path, legacy_output_path),
            cache_spec=cache_spec,
            timeout_sec=float(wait_timeout_sec),
            poll_interval_sec=float(poll_interval_sec),
        )

    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_owner = _try_acquire_prompt_embedding_lock(lock_path)
    if not lock_owner:
        return _wait_for_prompt_embedding(
            output_dir=output_dir,
            paths=(output_path, legacy_output_path),
            cache_spec=cache_spec,
            timeout_sec=float(wait_timeout_sec),
            poll_interval_sec=float(poll_interval_sec),
            lock_path=lock_path,
        )

    prompt_embedding = None
    try:
        encoder = T5EncoderModel(
            text_len=int(text_len),
            dtype=parse_torch_dtype(dtype),
            device=str(device),
            checkpoint_path=str(t5_checkpoint_path),
            tokenizer_path=str(tokenizer_path),
        )
        try:
            with torch.no_grad():
                prompt_embedding = encoder([prompt_text], str(device))[0].detach().cpu().float()
        finally:
            try:
                encoder.model.cpu()
            except Exception:
                pass
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        payload = {
            "prompt_text": prompt_text,
            "view_id": view_id,
            "text_len": int(text_len),
            "source_checkpoint_dir": str(checkpoint_dir),
            "source_t5_checkpoint": str(t5_checkpoint_path),
            "source_tokenizer_path": str(tokenizer_path),
            "source_dtype": str(dtype),
            "cache_key": str(cache_key),
            "cache_spec": cache_spec,
            "prompt_embedding": prompt_embedding,
        }
        torch.save(payload, output_path)
    finally:
        _release_prompt_embedding_lock(lock_path, owner_pid=lock_owner)
    return output_path


def resolve_wan_prompt_embedding_id(
    *,
    task_cfg: Any | None = None,
    dataset_cfg: Any | None = None,
) -> str:
    explicit_id = ""
    for cfg in (task_cfg, dataset_cfg):
        if cfg is None:
            continue
        explicit_id = str(getattr(cfg, "prompt_embedding_id", "")).strip()
        if explicit_id and explicit_id not in {"None", "null"}:
            return explicit_id

    if dataset_cfg is not None:
        view_id = str(getattr(dataset_cfg, "view_id", "")).strip()
        if view_id:
            return view_id

        sources = getattr(dataset_cfg, "sources", None)
        if sources is not None:
            source_keys = []
            for raw_source in sources:
                if raw_source is None:
                    continue
                if isinstance(raw_source, Mapping):
                    source_map = raw_source
                elif hasattr(raw_source, "items") and callable(getattr(raw_source, "items")):
                    source_map = dict(raw_source.items())
                else:
                    continue
                datahouse_id = str(source_map.get("datahouse_id", "")).strip()
                source_view_id = str(source_map.get("view_id", "")).strip()
                if not source_view_id:
                    continue
                source_key = source_view_id if not datahouse_id else f"{datahouse_id}:{source_view_id}"
                source_keys.append(source_key)
            if source_keys:
                if len(source_keys) == 1:
                    return source_keys[0]
                payload = json.dumps(source_keys, ensure_ascii=False, separators=(",", ":"))
                digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
                prefix = "_".join(_sanitize_filename(key) for key in source_keys[:2]).strip("_")
                if prefix:
                    prefix = prefix[:80]
                    return f"multi_{prefix}_{digest}"
                return f"multi_{digest}"

    task_name = "" if task_cfg is None else str(getattr(task_cfg, "name", "")).strip()
    if task_name:
        return task_name

    raise ValueError("Could not resolve prompt embedding id from task/dataset config.")


def _sanitize_filename(name: str) -> str:
    cleaned = []
    for char in str(name):
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "view"


def _build_prompt_embedding_cache_spec(
    *,
    prompt_text: str,
    checkpoint_dir: Path,
    dtype: str,
    text_len: int,
    tokenizer_path: str,
) -> dict[str, object]:
    spec = {
        "prompt_text": str(prompt_text),
        "text_len": int(text_len),
        "dtype": str(dtype).lower(),
        "tokenizer_path": str(tokenizer_path),
        "checkpoint_dir": str(checkpoint_dir),
        "t5_checkpoint_path": str(checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
    }
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    prefix = _sanitize_filename(prompt_text)[:48].strip("_")
    spec["cache_key"] = f"prompt_{prefix}_{digest}" if prefix else f"prompt_{digest}"
    return spec


def _payload_matches_prompt_spec(path: Path, cache_spec: Mapping[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    payload_spec = payload.get("cache_spec")
    if isinstance(payload_spec, Mapping):
        return dict(payload_spec) == dict(cache_spec)

    if payload.get("prompt_text") != cache_spec.get("prompt_text"):
        return False
    if int(payload.get("text_len", -1)) != int(cache_spec.get("text_len", -2)):
        return False
    if str(payload.get("source_t5_checkpoint", "")) != str(cache_spec.get("t5_checkpoint_path", "")):
        return False

    payload_tokenizer = str(payload.get("source_tokenizer_path", "")).strip()
    if payload_tokenizer and payload_tokenizer != str(cache_spec.get("tokenizer_path", "")):
        return False

    payload_dtype = str(payload.get("source_dtype", "")).strip().lower()
    if payload_dtype and payload_dtype != str(cache_spec.get("dtype", "")).lower():
        return False

    return True


def _try_acquire_prompt_embedding_lock(lock_path: Path) -> int | None:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
    finally:
        os.close(fd)
    return int(os.getpid())


def _release_prompt_embedding_lock(lock_path: Path, *, owner_pid: int | None) -> None:
    if owner_pid is None:
        return
    try:
        if lock_path.exists():
            lock_text = lock_path.read_text(encoding="utf-8").strip()
            if lock_text and int(lock_text) != int(owner_pid):
                return
            lock_path.unlink()
    except Exception:
        pass


def _wait_for_prompt_embedding(
    *,
    output_dir: Path,
    paths: tuple[Path, ...],
    cache_spec: Mapping[str, object],
    timeout_sec: float,
    poll_interval_sec: float,
    lock_path: Path | None = None,
) -> Path:
    deadline = time.time() + max(1.0, float(timeout_sec))
    sleep_sec = max(0.1, float(poll_interval_sec))
    while time.time() < deadline:
        existing_path = _find_existing_prompt_embedding(
            output_dir=output_dir,
            cache_spec=cache_spec,
            preferred_paths=paths,
        )
        if existing_path is not None:
            return existing_path
        if lock_path is not None and not lock_path.exists():
            existing_path = _find_existing_prompt_embedding(
                output_dir=output_dir,
                cache_spec=cache_spec,
                preferred_paths=paths,
            )
            if existing_path is not None:
                return existing_path
        time.sleep(sleep_sec)
    searched = ", ".join(str(path) for path in paths)
    raise TimeoutError(f"Timed out waiting for prompt embedding cache: {searched}")


def _find_existing_prompt_embedding(
    *,
    output_dir: Path,
    cache_spec: Mapping[str, object],
    preferred_paths: tuple[Path, ...],
) -> Path | None:
    seen: set[str] = set()
    candidates: list[Path] = []
    for path in preferred_paths:
        path_text = str(path)
        if path_text in seen:
            continue
        seen.add(path_text)
        candidates.append(path)
    for path in sorted(output_dir.glob("*.pt")):
        path_text = str(path)
        if path_text in seen:
            continue
        seen.add(path_text)
        candidates.append(path)
    for candidate_path in candidates:
        if _payload_matches_prompt_spec(candidate_path, cache_spec):
            return candidate_path
    return None


def should_create_wan_prompt_embedding() -> bool:
    raw_rank = os.environ.get("RANK")
    if raw_rank is None or str(raw_rank).strip() == "":
        return True
    try:
        return int(raw_rank) == 0
    except ValueError:
        return True


__all__ = [
    "ensure_wan_prompt_embedding",
    "parse_torch_dtype",
    "resolve_wan_prompt_embedding_id",
    "should_create_wan_prompt_embedding",
]
