from __future__ import annotations

import copy
import json
import itertools
import math
import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler

from rap_wam.core.common.checkpoint_util import TopKCheckpointManager
from rap_wam.core.common.json_logger import JsonLogger
from rap_wam.core.common.workspace_utils import (
    create_logger,
    log_action_mse,
    save_video_generation_visualization,
)
from rap_wam.core.model.common.lr_scheduler import get_scheduler
from rap_wam.core.model.common.normalizer import LinearNormalizer
from rap_wam.core.model.diffusion.ema_model import EMAModel
from rap_wam.core.model.wan.prompt_embedding import ensure_wan_prompt_embedding
from rap_wam.core.model.wan.prompt_embedding import resolve_wan_prompt_embedding_id
from rap_wam.core.model.wan.prompt_embedding import should_create_wan_prompt_embedding
from rap_wam.core.workspace.base_workspace import BaseWorkspace

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class WanTrainWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch", "_resumed_from_checkpoint")

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        super().__init__(cfg, output_dir=output_dir)
        self.logger = create_logger()
        resume_path = self._resolve_resume_path() if bool(getattr(cfg.training, "resume", False)) else None
        if bool(getattr(cfg.training, "resume", False)) and resume_path is None:
            raise FileNotFoundError("Resuming requires an existing checkpoint path")


        seed = int(getattr(cfg.training, "seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.global_step = 0
        self.epoch = 0
        self._prepared = False
        self._tracker_enabled = False
        self._resumed_from_checkpoint = False

        self.train_dataset = self._build_dataset(split="train")
        self.validation_dataset = self._build_dataset(split="val")
        self.model_spec = self.train_dataset.export_model_spec()
        self._prepare_prompt_embedding_cache()

        self.obs_encoder = hydra.utils.instantiate(
            cfg.policy.obs_encoder,
            obs_shapes=self.train_dataset.obs_shapes,
            obs_horizon=int(getattr(cfg, "n_obs_steps", self.train_dataset.n_obs_steps)),
            target_horizon=int(getattr(cfg, "n_video_future_steps", self.train_dataset.video_future_horizon)),
        )
        self.model = hydra.utils.instantiate(
            cfg.policy,
            _recursive_=False,
            obs_encoder=self.obs_encoder,
            action_dim=int(self.model_spec.get("action_dim", 0)),
            action_horizon=int(self.model_spec.get("action_horizon", 0)),
        )

        self.ema_model = None
        self.ema = None
        if bool(getattr(cfg.training, "use_ema", False)):
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = None

        if bool(getattr(cfg.training, "resume", False)):
            payload = self.load_checkpoint(path=resume_path)
            if not {"model", "optimizer"}.issubset(payload["state_dicts"]):
                raise ValueError("A training resume checkpoint must contain model and optimizer state")
            if payload.get("format_version", 1) < 2:
                self.logger.warning("Legacy checkpoint: verify epoch and auxiliary schedule continuation")
            self._resumed_from_checkpoint = True

    def _build_dataset(self, *, split: str):
        dataset_cfg = getattr(self.cfg.task, "dataset", None)
        if dataset_cfg is None or "_target_" not in dataset_cfg:
            raise ValueError("task.dataset must contain a valid _target_")

        return hydra.utils.instantiate(
            dataset_cfg,
            n_obs_steps=int(getattr(self.cfg, "n_obs_steps", 1)),
            n_action_steps=int(getattr(self.cfg, "n_action_steps", 0)),
            video_future_horizon=int(getattr(self.cfg, "n_video_future_steps", 16)),
            include_action=bool(
                getattr(
                    self.cfg,
                    "include_action",
                    getattr(self.cfg, "use_action_condition", False),
                )
            ),
            split=split,
        )

    def _build_optimizer(self):
        optimizer_cfg = getattr(self.cfg, "optimizer", None)
        if optimizer_cfg is None:
            raise ValueError("optimizer config is required")
        return self.model.get_optimizer(**dict(optimizer_cfg))

    def _build_dataloader(self, dataset, dataloader_cfg, *, start_epoch: int = 0, tag: str = "train"):
        loader_kwargs = dict(dataloader_cfg)
        shuffle = bool(loader_kwargs.pop("shuffle", False))
        sampler = None

        build_sampler = getattr(dataset, "build_dataloader_sampler", None)
        if callable(build_sampler):
            sampler = build_sampler(shuffle=shuffle, start_epoch=int(start_epoch))

        if sampler is not None:
            loader_kwargs["sampler"] = sampler
            if tag == "train":
                describe_sampling = getattr(dataset, "describe_dataloader_sampling", None)
                if callable(describe_sampling):
                    summary = describe_sampling()
                    self.logger.info(
                        "Using custom train sampler: mode=%s epoch_size=%s min_source_length=%s source_lengths=%s",
                        summary.get("mode"),
                        summary.get("epoch_size"),
                        summary.get("min_source_length"),
                        summary.get("source_lengths"),
                    )
        else:
            loader_kwargs["shuffle"] = shuffle

        limit = getattr(self.cfg.training, "max_train_steps", None)
        if tag == "train" and limit is not None:
            loader_kwargs.pop("shuffle", None)
            loader_kwargs.pop("sampler", None)
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            loader_kwargs["batch_sampler"] = LimitedBatchSampler(
                sampler, batch_size=int(loader_kwargs.pop("batch_size", 1)),
                drop_last=bool(loader_kwargs.pop("drop_last", False)),
                max_batches=int(limit) * max(1, int(os.environ.get("WORLD_SIZE", "1"))),
            )
        return DataLoader(dataset, **loader_kwargs)

    def _prepare_prompt_embedding_cache(self) -> None:
        task_cfg = getattr(self.cfg, "task", None)
        policy_cfg = getattr(self.cfg, "policy", None)
        if task_cfg is None or policy_cfg is None:
            return

        raw_prompt_path = getattr(policy_cfg, "prompt_embedding_path", None)
        if raw_prompt_path is not None and str(raw_prompt_path).strip() not in {"", "null", "None"}:
            return

        prompt_text = str(getattr(task_cfg, "text_prompt", "")).strip()
        if not prompt_text:
            return

        dataset_cfg = getattr(task_cfg, "dataset", None)
        prompt_embedding_id = resolve_wan_prompt_embedding_id(
            task_cfg=task_cfg,
            dataset_cfg=dataset_cfg,
        )

        embedding_dir = getattr(task_cfg, "text_embedding_dir", None)
        if embedding_dir is None or str(embedding_dir).strip() in {"", "null", "None"}:
            embedding_dir = self._default_text_embedding_dir()

        embedding_path = ensure_wan_prompt_embedding(
            prompt_text=prompt_text,
            view_id=prompt_embedding_id,
            wan_checkpoint_dir=str(getattr(self.cfg, "wan_checkpoint_dir")),
            output_dir=str(embedding_dir),
            device=str(getattr(self.cfg.training, "device", "cuda:0" if torch.cuda.is_available() else "cpu")),
            dtype=str(getattr(task_cfg, "text_embedding_dtype", "bf16")),
            text_len=int(getattr(task_cfg, "text_embedding_text_len", 512)),
            tokenizer_path=str(getattr(task_cfg, "t5_tokenizer_path", "google/umt5-xxl")),
            allow_create=should_create_wan_prompt_embedding(),
        )
        policy_cfg.prompt_embedding_path = str(embedding_path)
        self.logger.info(
            "Using Wan prompt embedding: %s (prompt_embedding_id=%s, creator=%s)",
            embedding_path,
            prompt_embedding_id,
            should_create_wan_prompt_embedding(),
        )

    def _default_text_embedding_dir(self) -> Path:
        return Path.cwd() / ".cache" / "rap_wam" / "text_embeddings"

    def _build_normalizer(self) -> LinearNormalizer:
        normalizer_kwargs = dict(getattr(self.cfg, "normalizer", {}))
        normalizer_kwargs.update(dict(getattr(self.cfg.task, "normalizer", {})))
        fixed_preset = str(normalizer_kwargs.get("fixed_preset", "")).strip()
        if fixed_preset:
            self.logger.info("Building Wan normalizer from fixed preset: %s", fixed_preset)
        else:
            self.logger.info("Computing Wan normalizer from training data...")
        return self.train_dataset.get_normalizer(**normalizer_kwargs)

    def prepare(self) -> None:
        if self._prepared:
            return

        if not self._resumed_from_checkpoint:
            normalizer = self._build_normalizer()
            self.model.set_normalizer(normalizer)
            if self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)
        self._prepared = True

    def _setup_accelerator(self, *, gradient_accumulate_every: int) -> Accelerator:
        log_with = None
        mixed_precision = str(getattr(self.cfg.training, "mixed_precision", "no"))
        logging_cfg = getattr(self.cfg, "logging", None)
        if logging_cfg is not None:
            mode = str(getattr(logging_cfg, "mode", "online")).lower()
            if mode not in {"disabled", "off", "none"}:
                if wandb is None:
                    raise ModuleNotFoundError("Tracking is enabled, but wandb is not installed.")
                log_with = "wandb"

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=int(gradient_accumulate_every),
            log_with=log_with,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            device_placement=True,
        )
        self._tracker_enabled = log_with is not None
        if accelerator.is_main_process:
            self.logger.info("=" * 60)
            self.logger.info("Wan Mixed Precision: %s", accelerator.mixed_precision)
            self.logger.info("Wan Device: %s", accelerator.device)
            self.logger.info("Wan Num Processes: %s", accelerator.num_processes)
            self.logger.info("=" * 60)
        return accelerator

    def _init_trackers(self, accelerator: Accelerator) -> None:
        if not self._tracker_enabled:
            return

        logging_cfg = OmegaConf.to_container(getattr(self.cfg, "logging", {}), resolve=True)
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}

        project_name = str(logging_cfg.pop("project", getattr(self.cfg, "name", "wan_video_policy")))
        if logging_cfg.get("dir") in {None, "", "null"}:
            logging_cfg.pop("dir", None)
        logging_cfg = {key: value for key, value in logging_cfg.items() if value is not None}

        accelerator.init_trackers(
            project_name=project_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            init_kwargs={"wandb": logging_cfg},
        )

    def _resolve_resume_path(self) -> str | None:
        raw_resume_path = getattr(self.cfg.training, "resume_path", None)
        if raw_resume_path is not None:
            text = str(raw_resume_path).strip()
            if text:
                path = Path(text).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"Resume checkpoint not found: {path}")
                return str(path)
        latest_path = self.get_checkpoint_path("latest")
        if latest_path.exists():
            return str(latest_path)
        return None

    @torch.no_grad()
    def _run_validation(
        self,
        *,
        val_loader: DataLoader,
        accelerator: Accelerator,
        max_val_steps: int | None,
    ) -> float | None:
        if len(val_loader) == 0:
            return None

        model = self.ema_model if self.ema_model is not None else self.model
        was_training = bool(getattr(model, "training", False))
        model.eval()
        losses = []
        val_steps_limit = min(len(val_loader), int(max_val_steps)) if max_val_steps is not None else len(val_loader)
        try:
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= val_steps_limit:
                    break
                batch = _to_device(batch, accelerator.device)
                loss = model(batch, training=False)
                losses.append(loss.detach())
        finally:
            if was_training:
                model.train()

        if not losses:
            return None
        loss_tensor = torch.stack(losses)
        gathered = accelerator.gather_for_metrics(loss_tensor)
        if not accelerator.is_main_process:
            return None
        return float(torch.mean(gathered).item())

    @torch.no_grad()
    def _run_sampling_evaluation(
        self,
        *,
        batch: dict[str, Any],
        policy,
        device: torch.device,
        step: int | None = None,
        output_dir: str | None = None,
    ) -> dict[str, float]:
        eval_metrics: dict[str, float] = {}
        batch = _to_device(batch, device)

        was_training = bool(getattr(policy, "training", False))
        policy.eval()
        try:
            obs = dict(batch["obs"])
            if "task_index" in batch:
                obs["task_index"] = batch["task_index"]
            pred_outputs = policy.predict_visual(obs)
            pred_latents = pred_outputs["visual_pred"]
            pred_future = policy.decode_visual_tokens(pred_latents)
            gt_future = obs["rgb_head_future"].float()
            eval_metrics["sample_visual_mse"] = float(torch.mean((pred_future - gt_future) ** 2).item())
            if step is not None and output_dir is not None:
                save_video_generation_visualization(
                    obs_dict=obs,
                    pred_future=pred_future,
                    step=int(step),
                    output_dir=output_dir,
                    prefix="train",
                    max_samples=1,
                    num_future_frames_to_show=5,
                )
            if "action" in batch:
                try:
                    pred_action = pred_outputs.get("action_pred")
                    if pred_action is None:
                        pred_action = policy.predict_action(obs)["action_pred"]
                    pred_action = pred_action.detach().clone()
                    log_action_mse(eval_metrics, "sample", pred_action, batch["action"])
                except (AttributeError, NotImplementedError):
                    pass
            if step is not None and output_dir is not None and eval_metrics:
                metrics_dir = Path(output_dir) / "train_visualizations"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / f"step_{int(step):08d}_train_metrics.json"
                metrics_path.write_text(
                    json.dumps(eval_metrics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        finally:
            if was_training:
                policy.train()
        return eval_metrics

    def _save_checkpoint_bundle(
        self,
        *,
        accelerator: Accelerator,
        ckpt_dir: Path,
        topk_manager: TopKCheckpointManager,
        monitor_payload: dict[str, float | int],
    ) -> None:
        if not accelerator.is_main_process:
            return

        wrapped_model = self.model
        self.model = accelerator.unwrap_model(self.model)
        try:
            latest_path = str(ckpt_dir / "latest.ckpt")
            self.save_checkpoint(latest_path)
            topk_path = topk_manager.get_ckpt_path(monitor_payload)
            if topk_path:
                self.save_checkpoint(topk_path)
        finally:
            self.model = wrapped_model

    def _get_eval_policy(self, accelerator: Accelerator):
        policy = accelerator.unwrap_model(self.model)
        if self.ema_model is not None:
            policy = self.ema_model
        return policy

    def run(self):
        if len(self.train_dataset) == 0:
            raise ValueError("The training dataset is empty")

        self.prepare()
        cfg = self.cfg
        grad_accum = max(1, int(getattr(cfg.training, "gradient_accumulate_every", 1)))
        grad_clip = float(getattr(cfg.training, "grad_norm_clip", 1.0))
        max_train_steps = getattr(cfg.training, "max_train_steps", None)
        max_val_steps = getattr(cfg.training, "max_val_steps", None)
        log_every_n_steps = max(1, int(getattr(cfg.training, "log_every_n_steps", 1)))
        tqdm_interval_sec = float(getattr(cfg.training, "tqdm_interval_sec", 1.0))
        checkpoint_every_steps = int(getattr(cfg.training, "checkpoint_every_steps", 0))
        sample_every_steps = int(getattr(cfg.training, "sample_every_steps", 0))

        accelerator = self._setup_accelerator(gradient_accumulate_every=grad_accum)
        self._init_trackers(accelerator)

        train_loader = self._build_dataloader(
            self.train_dataset,
            cfg.train_dataloader,
            start_epoch=int(self.epoch),
            tag="train",
        )
        val_loader = self._build_dataloader(
            self.validation_dataset,
            cfg.val_dataloader,
            start_epoch=0,
            tag="val",
        )

        local_batches = math.ceil(len(train_loader) / accelerator.num_processes)
        updates_per_epoch = math.ceil(local_batches / grad_accum)
        # Accelerate 1.2 advances the prepared scheduler once per process.
        total_steps = max(1, updates_per_epoch * int(cfg.training.num_epochs) * accelerator.num_processes)
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            self.optimizer,
            num_warmup_steps=int(getattr(cfg.training, "lr_warmup_steps", 0)),
            num_training_steps=total_steps,
            last_epoch=int(self.global_step) * accelerator.num_processes - 1,
        )

        if bool(getattr(cfg.training, "use_ema", False)):
            if self.ema_model is None:
                self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(accelerator.device)
            self.ema = EMAModel(model=self.ema_model, **dict(cfg.ema))

        train_loader, val_loader, self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            train_loader,
            val_loader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        ckpt_dir = Path(self.output_dir) / "checkpoints"
        if accelerator.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        topk_manager = TopKCheckpointManager(save_dir=str(ckpt_dir), **cfg.checkpoint.topk)

        log_path = str(Path(self.output_dir) / "logs.json.txt")
        json_logger_ctx = JsonLogger(log_path) if accelerator.is_main_process else _NoopJsonLogger()

        try:
            with json_logger_ctx as json_logger:
                for epoch in range(int(self.epoch), int(cfg.training.num_epochs)):
                    self.epoch = int(epoch)
                    self.model.train()
                    train_losses = []
                    train_component_history: dict[str, list[float]] = {}
                    train_sampling_batch = None

                    num_batches = len(train_loader)
                    train_steps_limit = (
                        min(num_batches, int(max_train_steps))
                        if max_train_steps is not None
                        else num_batches
                    )

                    with tqdm.tqdm(
                        train_loader,
                        desc=f"Wan training epoch {self.epoch}",
                        leave=False,
                        mininterval=tqdm_interval_sec,
                        disable=not accelerator.is_main_process,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            if batch_idx >= train_steps_limit:
                                break

                            batch = _to_device(batch, accelerator.device)
                            train_sampling_batch = batch

                            with accelerator.accumulate(self.model):
                                loss = self.model(batch, training=True)
                                if not torch.isfinite(loss.detach()).all():
                                    raise FloatingPointError("Nonfinite training loss; optimizer update stopped")
                                accelerator.backward(loss)

                                if accelerator.sync_gradients:
                                    accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)

                                self.optimizer.step()
                                self.lr_scheduler.step()
                                self.optimizer.zero_grad(set_to_none=True)

                                if accelerator.sync_gradients and self.ema is not None:
                                    self.ema.step(accelerator.unwrap_model(self.model))

                            gathered = accelerator.gather_for_metrics(loss.detach().reshape(1))
                            raw_loss = float(gathered.mean().item())
                            train_losses.append(raw_loss)
                            tepoch.set_postfix(loss=raw_loss, refresh=False)
                            loss_components = _get_loss_components(self.model)
                            for key, value in loss_components.items():
                                if key == "total_loss":
                                    continue
                                train_component_history.setdefault(str(key), []).append(float(value))

                            if accelerator.sync_gradients:
                                self.global_step += 1
                                step_log = {
                                    "train_loss": raw_loss,
                                    "global_step": self.global_step,
                                    "epoch": self.epoch,
                                    "lr": float(self.lr_scheduler.get_last_lr()[0]),
                                }
                                step_log.update(
                                    {
                                        key: float(values[-1])
                                        for key, values in train_component_history.items()
                                        if values
                                    }
                                )

                                sample_trigger = (
                                    train_sampling_batch is not None
                                    and sample_every_steps > 0
                                    and (self.global_step % sample_every_steps) == 0
                                )
                                if sample_trigger:
                                    accelerator.wait_for_everyone()
                                    if accelerator.is_main_process:
                                        step_log.update(
                                            self._run_sampling_evaluation(
                                                batch=train_sampling_batch,
                                                policy=self._get_eval_policy(accelerator),
                                                device=accelerator.device,
                                                step=self.global_step,
                                                output_dir=self.output_dir,
                                            )
                                        )
                                        _log_sampling_metrics(
                                            logger=self.logger,
                                            metrics=step_log,
                                            step=self.global_step,
                                            epoch=self.epoch,
                                        )
                                    accelerator.wait_for_everyone()

                                checkpoint_trigger = (
                                    checkpoint_every_steps > 0
                                    and (self.global_step % checkpoint_every_steps) == 0
                                )
                                if checkpoint_trigger:
                                    accelerator.wait_for_everyone()
                                    epoch_running_mean_loss = (
                                        float(np.mean(train_losses)) if train_losses else step_log["train_loss"]
                                    )
                                    self._save_checkpoint_bundle(
                                        accelerator=accelerator,
                                        ckpt_dir=ckpt_dir,
                                        topk_manager=topk_manager,
                                        monitor_payload={
                                            "epoch": self.epoch,
                                            "global_step": self.global_step,
                                            "train_loss": epoch_running_mean_loss,
                                            "val_loss": step_log.get("val_loss", epoch_running_mean_loss),
                                        },
                                    )
                                    accelerator.wait_for_everyone()

                                should_log_step = (
                                    (self.global_step % log_every_n_steps) == 0
                                    or sample_trigger
                                    or checkpoint_trigger
                                )
                                if should_log_step:
                                    if accelerator.is_main_process:
                                        json_logger.log(step_log)
                                    if self._tracker_enabled:
                                        accelerator.log(step_log, step=self.global_step)

                    avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                    step_log = {
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                        "train_loss": avg_train_loss,
                        "lr": float(self.lr_scheduler.get_last_lr()[0]),
                    }
                    step_log.update(
                        {
                            key: float(values[-1]) if key.endswith("_total") else float(np.mean(values))
                            for key, values in train_component_history.items()
                            if values
                        }
                    )

                    completed_epochs = self.epoch + 1
                    val_every = int(getattr(cfg.training, "val_every", 0))
                    if val_every > 0 and (completed_epochs % val_every) == 0:
                        val_loss = self._run_validation(
                            val_loader=val_loader,
                            accelerator=accelerator,
                            max_val_steps=max_val_steps,
                        )
                        if val_loss is not None:
                            step_log["val_loss"] = val_loss

                    sample_every = int(getattr(cfg.training, "sample_every", 0))
                    if (
                        sample_every_steps <= 0
                        and accelerator.is_main_process
                        and train_sampling_batch is not None
                        and sample_every > 0
                        and (completed_epochs % sample_every) == 0
                    ):
                        step_log.update(
                            self._run_sampling_evaluation(
                                batch=train_sampling_batch,
                                policy=self._get_eval_policy(accelerator),
                                device=accelerator.device,
                                step=self.global_step,
                                output_dir=self.output_dir,
                            )
                        )
                        _log_sampling_metrics(
                            logger=self.logger,
                            metrics=step_log,
                            step=self.global_step,
                            epoch=self.epoch,
                        )

                    if accelerator.is_main_process:
                        self.logger.info(
                            "wan epoch=%s step=%s train_loss=%.6f%s",
                            self.epoch,
                            self.global_step,
                            avg_train_loss,
                            f" val_loss={step_log['val_loss']:.6f}" if "val_loss" in step_log else "",
                        )
                        json_logger.log(step_log)

                    if self._tracker_enabled:
                        accelerator.log(step_log, step=self.global_step)

                    accelerator.wait_for_everyone()
                    checkpoint_every = int(getattr(cfg.training, "checkpoint_every", 0))
                    if checkpoint_every_steps <= 0 and checkpoint_every > 0 and (completed_epochs % checkpoint_every) == 0:
                        self.epoch = completed_epochs
                        self._save_checkpoint_bundle(
                            accelerator=accelerator,
                            ckpt_dir=ckpt_dir,
                            topk_manager=topk_manager,
                            monitor_payload={
                                "epoch": completed_epochs - 1,
                                "global_step": self.global_step,
                                "train_loss": step_log["train_loss"],
                                "val_loss": step_log.get("val_loss", step_log["train_loss"]),
                            },
                        )
                        accelerator.wait_for_everyone()
        finally:
            accelerator.end_training()


class LimitedBatchSampler(BatchSampler):
    """Bound the actual loader length so accumulation flushes at the final batch."""
    def __init__(self, sampler, batch_size, drop_last, max_batches):
        super().__init__(sampler, batch_size, drop_last)
        if max_batches < 1:
            raise ValueError("max_train_steps must be positive")
        self.max_batches = int(max_batches)

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.max_batches)

    def __len__(self):
        return min(super().__len__(), self.max_batches)


class _NoopJsonLogger:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log(self, data: dict) -> None:
        del data


def _to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_device(val, device) for val in value]
    if isinstance(value, tuple):
        return tuple(_to_device(val, device) for val in value)
    return value


def _get_loss_components(model: Any) -> dict[str, float]:
    candidates = [model, getattr(model, "module", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        getter = getattr(candidate, "get_last_loss_components", None)
        if callable(getter):
            payload = getter()
            if isinstance(payload, dict):
                return {
                    str(key): float(value)
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                }
    return {}


def _log_sampling_metrics(
    *,
    logger,
    metrics: dict[str, Any],
    step: int,
    epoch: int,
) -> None:
    visual_mse = metrics.get("sample_visual_mse")
    action_mse = metrics.get("sample_action_mse_error")
    if visual_mse is None and action_mse is None:
        return

    parts = [f"wan sample epoch={epoch} step={step}"]
    if visual_mse is not None:
        parts.append(f"visual_mse={float(visual_mse):.6f}")
    if action_mse is not None:
        parts.append(f"action_mse={float(action_mse):.6f}")
    logger.info(" ".join(parts))


__all__ = ["WanTrainWorkspace"]
