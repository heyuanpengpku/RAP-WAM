from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch


def log_action_mse(step_log, category, pred_action, gt_action):
    pred_action = pred_action.view(pred_action.shape[0], pred_action.shape[1], -1)
    gt_action = gt_action.view(gt_action.shape[0], gt_action.shape[1], -1)
    error = torch.nn.functional.mse_loss(pred_action, gt_action)
    step_log[f"{category}_action_mse_error"] = float(error.item())


def create_logger(name: str = "rap_wam.core"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def save_video_generation_visualization(
    *,
    obs_dict: dict[str, torch.Tensor],
    pred_future: torch.Tensor,
    step: int,
    output_dir: str | Path,
    prefix: str = "train",
    max_samples: int = 1,
    num_future_frames_to_show: int = 5,
) -> list[Path]:
    """Save a compact PNG comparing condition / GT future / predicted future frames."""

    if "rgb_head" not in obs_dict or "rgb_head_future" not in obs_dict:
        return []

    vis_dir = Path(output_dir) / "train_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    condition = obs_dict["rgb_head"].detach().cpu()
    gt_future = obs_dict["rgb_head_future"].detach().cpu()
    pred_future = pred_future.detach().cpu()

    num_samples = min(
        int(max_samples),
        int(condition.shape[0]),
        int(gt_future.shape[0]),
        int(pred_future.shape[0]),
    )
    saved_paths: list[Path] = []
    for sample_idx in range(num_samples):
        canvas = _build_video_generation_canvas(
            condition_frame=condition[sample_idx, -1],
            gt_future=gt_future[sample_idx],
            pred_future=pred_future[sample_idx],
            num_future_frames_to_show=num_future_frames_to_show,
        )
        save_path = vis_dir / f"step_{int(step):08d}_{prefix}_sample_{sample_idx:02d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        saved_paths.append(save_path)
    return saved_paths


def _build_video_generation_canvas(
    *,
    condition_frame: torch.Tensor,
    gt_future: torch.Tensor,
    pred_future: torch.Tensor,
    num_future_frames_to_show: int,
) -> np.ndarray:
    gt_len = int(gt_future.shape[0])
    pred_len = int(pred_future.shape[0])
    usable_frames = min(gt_len, pred_len)
    if usable_frames <= 0:
        raise ValueError("No future frames available for visualization.")

    frame_indices = _select_frame_indices(
        num_frames=usable_frames,
        max_frames=max(1, int(num_future_frames_to_show)),
    )
    top_tiles = [_add_image_label(_to_uint8_hwc(condition_frame), "Condition")]
    bottom_tiles = [_add_image_label(_to_uint8_hwc(condition_frame), "Condition")]

    for frame_idx in frame_indices:
        label = f"t+{int(frame_idx) + 1}"
        top_tiles.append(_add_image_label(_to_uint8_hwc(gt_future[frame_idx]), f"GT {label}"))
        bottom_tiles.append(_add_image_label(_to_uint8_hwc(pred_future[frame_idx]), f"Pred {label}"))

    top_row = np.concatenate(top_tiles, axis=1)
    bottom_row = np.concatenate(bottom_tiles, axis=1)
    separator = np.full((10, top_row.shape[1], 3), 235, dtype=np.uint8)
    return np.concatenate([top_row, separator, bottom_row], axis=0)


def _select_frame_indices(*, num_frames: int, max_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames <= max_frames:
        return list(range(num_frames))
    return np.linspace(0, num_frames - 1, num=max_frames, dtype=int).tolist()


def _to_uint8_hwc(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {frame.shape}")
    if frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    frame = np.clip(frame, 0.0, 1.0)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] > 3:
        frame = frame[..., :3]
    return (frame * 255.0).astype(np.uint8)


def _add_image_label(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image for labeling, got {image.shape}")
    banner_height = max(28, image.shape[0] // 8)
    banner = np.full((banner_height, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        banner,
        str(label),
        (10, max(18, banner_height - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([banner, image], axis=0)
