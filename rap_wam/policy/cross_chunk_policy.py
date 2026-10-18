"""Predictive rehearsal across adjacent demonstration chunks (Section 3.2).

Generate the first chunk without gradients, decode its last future image, and
use that image to condition supervised training on the recorded next chunk.
Only the next-chunk model forward carries gradients. Public recipes surface
auxiliary errors; an explicit compatibility option permits a logged fallback."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

import torch

from rap_wam.policy.residual_policy import (
    ResidualActionPolicy,
)


class CrossChunkPolicy(ResidualActionPolicy):
    """Supervise the recorded next chunk under a generated image condition."""

    def __init__(
        self,
        *args,
        ccrt_lambda_max: float = 1.0,
        ccrt_warmup_steps: int = 12000,
        ccrt_p_max: float = 0.25,
        ccrt_sample_steps: int = 4,
        strict_auxiliary_errors: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strict_auxiliary_errors = bool(strict_auxiliary_errors)
        self.ccrt_lambda_max = float(ccrt_lambda_max)
        self.ccrt_warmup_steps = int(ccrt_warmup_steps)
        self.ccrt_p_max = float(ccrt_p_max)
        self.ccrt_sample_steps = int(ccrt_sample_steps)
        self._ccrt_step = 0
        # cond / future obs keys (used to build the chunk-1 obs dict)
        self._cond_key = (
            getattr(self.obs_encoder, "image_key", None)
            or getattr(self.obs_encoder, "condition_key", None)
            or "rgb_head"
        )
        self._future_key = getattr(self.obs_encoder, "target_key", None) or "rgb_head_future"
        self._dbg_lam = 0.0
        self._dbg_c1 = 0.0
        self._dbg_ap = 0.0
        self._auxiliary_counts = {name: {key: 0 for key in
            ("attempted", "succeeded", "applied", "failed")}
            for name in ("rehearsal", "credit")}

        if not hasattr(self, "_last_loss_components") or self._last_loss_components is None:
            self._last_loss_components = {}  # type: ignore[assignment]
        self._last_loss_components["ccrt_lambda_max"] = float(self.ccrt_lambda_max)
        self._last_loss_components["ccrt_lambda_current"] = 0.0
        self._last_loss_components["ccrt_chunk1_loss"] = 0.0
        self._last_loss_components["ccrt_applied"] = 0.0

    def _record_auxiliary(self, branch: str, event: str):
        self._auxiliary_counts[branch][event] += 1
        self._last_loss_components[f"{branch}_{event}"] = 1.0

    def _auxiliary_error(self, branch: str, error: Exception):
        self._record_auxiliary(branch, "failed")
        if self.strict_auxiliary_errors:
            raise RuntimeError(f"{branch} auxiliary failed; training stopped") from error
        import warnings
        warnings.warn(f"{branch} auxiliary failed ({type(error).__name__}); "
                      "see auxiliary failure counters", RuntimeWarning, stacklevel=2)

    def _publish_auxiliary_counts(self):
        for branch, counts in self._auxiliary_counts.items():
            for event, count in counts.items():
                self._last_loss_components[f"{branch}_{event}_total"] = float(count)

    def _anneal(self, mx: float) -> float:
        if self.ccrt_warmup_steps <= 0:
            return mx
        return mx * min(1.0, self._ccrt_step / float(self.ccrt_warmup_steps))

    # ------------------------------------------------------------------
    # chunk-1 supervised loss: drifted chunk-0 prediction as condition,
    # genuine next-chunk GT future/action as supervision.
    # ------------------------------------------------------------------
    def _ccrt_chunk1_loss(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        obs = batch.get("obs", None)
        obs2 = batch.get("obs2", None)
        if obs is None or obs2 is None:
            return None
        cond = obs.get(self._cond_key, None)
        gt_future2 = obs2.get(self._future_key, None)
        if cond is None or gt_future2 is None:
            return None

        # ---- 1) chunk-0 prediction (NO grad, few-step) -> drifted last frame ----
        chunk0_obs: Dict[str, torch.Tensor] = {self._cond_key: cond}
        if self.state_key in obs:
            chunk0_obs[self.state_key] = obs[self.state_key]
        saved_steps = self.num_inference_steps
        try:
            self.num_inference_steps = int(self.ccrt_sample_steps)
            with torch.no_grad():
                pred = self.predict_visual(chunk0_obs, show_progress=False)
                pred_future_pix = self.decode_visual_tokens(pred["visual_pred"])  # [B,T,C,H,W] in [0,1]
        finally:
            self.num_inference_steps = saved_steps
        if pred_future_pix is None or pred_future_pix.shape[1] < 1:
            return None
        drift_last = pred_future_pix[:, -1:].detach().to(cond.dtype)  # [B,1,C,H,W]

        # ---- 2) chunk-1 obs = drifted cond + GENUINE GT future / state ----
        chunk1_obs: Dict[str, torch.Tensor] = {
            self._cond_key: drift_last,
            self._future_key: gt_future2,
        }
        if self.state_key in obs2:
            chunk1_obs[self.state_key] = obs2[self.state_key]

        # ---- 3) chunk-1 supervised flow-matching loss (full gradient on model) ----
        nobs = self._normalize_non_image_obs(chunk1_obs)
        batch_inputs = self.obs_encoder.prepare_batch(nobs, training=True)
        # targets are constants: detach so the ONLY grad path is the model forward
        clean_latents = batch_inputs["clean_latents"].detach()
        condition_latents = batch_inputs["condition_latents"].detach()
        latent_condition_mask = batch_inputs["latent_condition_mask"].detach()
        num_condition_latent_steps = int(batch_inputs["num_condition_latent_steps"])

        batch_size = clean_latents.shape[0]
        timestep_idx = torch.randint(
            0, self.train_timesteps.shape[0], (batch_size,), device=self.device
        )
        timesteps = self.train_timesteps[timestep_idx]

        video_sigmas = (
            self.train_sigmas[timestep_idx].view(batch_size, 1, 1, 1, 1).to(clean_latents.dtype)
        )
        video_noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - video_sigmas) + video_noise * video_sigmas
        target_video_velocity = video_noise - clean_latents

        action2 = batch.get("action_2", None)
        have_action = action2 is not None
        if have_action:
            if "action" in self.normalizer.params_dict:
                clean_actions = self.normalizer["action"].normalize(action2.float())
            else:
                clean_actions = action2.float()
            action_sigmas = (
                self.train_sigmas[timestep_idx].view(batch_size, 1, 1).to(clean_actions.dtype)
            )
            action_noise = torch.randn_like(clean_actions)
            noisy_actions = clean_actions * (1.0 - action_sigmas) + action_noise * action_sigmas
            target_action_velocity = action_noise - clean_actions
        else:
            noisy_actions = torch.randn(
                batch_size, self.action_horizon, self.action_dim,
                device=self.device, dtype=clean_latents.dtype,
            )

        pred_video_velocity, pred_action_velocity, _ = self._run_joint_model_with_queries(
            noisy_latents=noisy_latents,
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            context_list=self._build_context_list(batch_size=batch_size),
            condition_latents=condition_latents,
            latent_condition_mask=latent_condition_mask,
            state_obs=nobs.get(self.state_key),
        )

        if self.use_future_only_loss:
            pred_video_velocity = pred_video_velocity[:, :, num_condition_latent_steps:, :, :]
            target_video_velocity = target_video_velocity[:, :, num_condition_latent_steps:, :, :]

        video_loss = self._compute_loss_tensor(pred_video_velocity, target_video_velocity)
        if have_action:
            action_loss = self._compute_loss_tensor(pred_action_velocity, target_action_velocity)
        else:
            action_loss = torch.zeros((), device=self.device, dtype=video_loss.dtype)

        return self.video_loss_weight * video_loss + self.action_loss_weight * action_loss

    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, Any], training: bool = True):
        loss = super().compute_loss(batch, training=training)  # chunk-0 main loss (parent, GT cond)
        self._dbg_lam = 0.0
        self._dbg_c1 = 0.0
        self._dbg_ap = 0.0
        for branch in self._auxiliary_counts:
            for event in self._auxiliary_counts[branch]:
                self._last_loss_components[f"{branch}_{event}"] = 0.0

        if training and self.ccrt_lambda_max > 0.0 and ("obs2" in batch):
            self._ccrt_step += 1
            p = self._anneal(self.ccrt_p_max)
            if random.random() < p:
                self._record_auxiliary("rehearsal", "attempted")
                try:
                    c1 = self._ccrt_chunk1_loss(batch)
                    if c1 is not None:
                        self._record_auxiliary("rehearsal", "succeeded")
                        lam = self._anneal(self.ccrt_lambda_max)
                        if lam > 0.0:
                            loss = loss + lam * c1.to(loss.dtype)
                            self._dbg_lam = float(lam)
                            self._dbg_c1 = float(c1.detach().item())
                            self._dbg_ap = 1.0
                            self._record_auxiliary("rehearsal", "applied")
                except Exception as error:
                    self._auxiliary_error("rehearsal", error)

        try:
            if isinstance(self._last_loss_components, dict):
                self._last_loss_components["ccrt_lambda_current"] = float(self._dbg_lam)
                self._last_loss_components["ccrt_chunk1_loss"] = float(self._dbg_c1)
                self._last_loss_components["ccrt_applied"] = float(self._dbg_ap)
                self._last_loss_components["ccrt_step"] = float(self._ccrt_step)
                self._last_loss_components["total_loss"] = float(loss.detach().item())
        except Exception:
            pass

        self._publish_auxiliary_counts()
        return loss


__all__ = ["CrossChunkPolicy"]
