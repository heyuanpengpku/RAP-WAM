"""Directional attention for the future-supervised action interface.

Action rows cannot read video keys directly. State-mediated visual paths remain
available. Video rows can read action keys for the action-response probe. The
same connectivity and learned queries are used for training and sampling."""

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# NOTE: import order matters. ``rap_wam.policy.imagination_policy`` transitively
# imports ``rap_wam.core.model.wan.runtime``, whose module-load side effect
# inserts ``third_party/wan_official`` onto ``sys.path`` -- only after that is
# the official ``wan`` package importable. Keep this import strictly before the
# ``from wan.modules...`` import below (mirrors the ordering in
# rap_wam/core/policy/wan_joint_action_policy.py).
from rap_wam.policy.imagination_policy import ImaginationQueryPolicy

from wan.modules.model import rope_apply


class DecoupledPolicy(ImaginationQueryPolicy):
    """Directional action-to-video read mask, shared by training and sampling.

    Action tokens read query, action and (by default) state keys. Queries are
    future-supervised, but are not the only indirect visual-information path."""

    def __init__(
        self,
        *,
        decouple_action_from_video: bool = True,
        decouple_action_attends_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.decouple_action_from_video = bool(decouple_action_from_video)
        self.decouple_action_attends_state = bool(decouple_action_attends_state)

        if self.decouple_action_from_video and self.K <= 0:
            raise ValueError(
                "DecoupledPolicy requires num_latent_queries (K) > 0 "
                "when decouple_action_from_video is True: the K-query tokens are "
                "the action's only remaining video-information path."
            )

        # Token-count layout discovered per-forward in _joint_self_attention;
        # cached only for diagnostics / logging.
        self._decouple_last_video_tokens: int = 0
        self._decouple_last_action_tokens: int = 0
        self._decouple_last_state_tokens: int = 0
        self._decouple_last_query_tokens: int = 0

        # Per-forward layout hint: how many K-query tokens the *current*
        # _run_joint_model* forward appended.  Read by _joint_self_attention so
        # the decoupling mask tracks the ACTUAL sequence layout instead of
        # hard-coding self.K.  Initialised to self.K because BOTH the inherited
        # training forward (_run_joint_model_with_queries) and parent's own
        # inference override (_run_joint_model) append exactly self.K queries;
        # the training forward does not touch this attribute, so the default
        # must already be correct for it.  parent's _run_joint_model override sets
        # it explicitly per-forward for defensive robustness.
        self._decouple_pending_query_tokens: int = int(self.K) if self.K > 0 else 0

    # ------------------------------------------------------------------
    # Attention mask: action-query rows masked off the video-key columns.
    # ------------------------------------------------------------------
    def _build_decouple_attn_mask(
        self,
        *,
        seq_len: int,
        video_token_count: int,
        action_token_count: int,
        state_token_count: int,
        query_token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Additive ``[L, L]`` self-attention mask (broadcast over B and heads).

        Token order in the joint sequence (assembled identically by the
        inherited training forward ``_run_joint_model_with_queries`` and by
        parent's own inference forward ``_run_joint_model``):

            [ video : [0, V)               ]
            [ action: [V, V + A)           ]
            [ state : [V + A, V + A + S)   ]
            [ query : [V + A + S, L)       ]   (L = V + A + S + K)

        Returns an additive float mask where masked entries hold the dtype's
        most-negative value and all other entries are 0.  Returns ``None`` when
        no masking is needed (decoupling off, K=0, or no video/action tokens) so
        the caller can take the unmasked fast path.
        """
        if not self.decouple_action_from_video:
            return None
        if video_token_count <= 0 or action_token_count <= 0 or query_token_count <= 0:
            # Nothing to decouple, or the bottleneck is absent: no-op.
            return None

        v = int(video_token_count)
        a = int(action_token_count)
        # Action token row range.
        a_start, a_end = v, v + a

        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        neg = torch.finfo(dtype).min

        # Action-query rows: block out the video-key columns [0, V).
        mask[a_start:a_end, 0:v] = neg
        if not self.decouple_action_attends_state:
            # Also block the state-key columns so the action attends to
            # {action, query} only.
            s = int(state_token_count)
            if s > 0:
                s_start = v + a
                mask[a_start:a_end, s_start : s_start + s] = neg
        return mask

    # ------------------------------------------------------------------
    # Joint self-attention with the decoupling mask.
    #
    # Reproduces WanJointActionPolicy._joint_self_attention exactly (same q/k/v
    # projection, same video-only RoPE), but routes the attention through SDPA
    # with an explicit additive mask instead of the upstream ``wan_attention``
    # wrapper (which hard-codes attn_mask=None).  flash-attn is absent in this
    # environment, so ``wan_attention`` itself already uses SDPA -- calling SDPA
    # here is therefore the same kernel, just with the mask plumbed through.
    # ------------------------------------------------------------------
    def _joint_self_attention(
        self,
        *,
        attn_module,
        x: torch.Tensor,
        video_token_count: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        b, l, n, d = x.size(0), x.size(1), attn_module.num_heads, attn_module.head_dim
        q = attn_module.norm_q(attn_module.q(x)).view(b, l, n, d)
        k = attn_module.norm_k(attn_module.k(x)).view(b, l, n, d)
        v = attn_module.v(x).view(b, l, n, d)

        # Video-only RoPE -- identical to the parent implementation.
        if video_token_count > 0:
            q_video = rope_apply(q[:, :video_token_count], grid_sizes, freqs)
            k_video = rope_apply(k[:, :video_token_count], grid_sizes, freqs)
            if video_token_count < l:
                q = torch.cat([q_video, q[:, video_token_count:]], dim=1)
                k = torch.cat([k_video, k[:, video_token_count:]], dim=1)
            else:
                q = q_video
                k = k_video

        # Infer the action / state / query split from the token layout.
        # action_horizon is fixed.  The K-query count is whatever the CURRENT
        # forward appended -- recorded in self._decouple_pending_query_tokens by the
        # _run_joint_model* methods (the inherited training forward leaves it at
        # its self.K default; parent's _run_joint_model override sets it
        # explicitly).  This is read PER-FORWARD rather than hard-coding self.K,
        # so the mask tracks the actual sequence and can never be silently
        # dropped by a wrong assumption about which forward is running.
        action_token_count = int(self.action_horizon)
        query_token_count = int(getattr(self, "_decouple_pending_query_tokens", self.K if self.K > 0 else 0))
        if query_token_count < 0:
            query_token_count = 0
        state_token_count = (
            int(l) - int(video_token_count) - action_token_count - query_token_count
        )
        if state_token_count < 0:
            # Layout mismatch under the recorded query count: try once more
            # assuming no K-queries were appended (defensive -- should not
            # happen now that both forwards append self.K, but never mis-mask).
            query_token_count = 0
            state_token_count = (
                int(l) - int(video_token_count) - action_token_count
            )
        if state_token_count < 0:
            # Still inconsistent: fall back to the unmasked parent behaviour
            # rather than mis-masking.
            state_token_count = 0
            attn_mask = None
        else:
            self._decouple_last_video_tokens = int(video_token_count)
            self._decouple_last_action_tokens = action_token_count
            self._decouple_last_state_tokens = state_token_count
            self._decouple_last_query_tokens = query_token_count
            attn_mask = self._build_decouple_attn_mask(
                seq_len=int(l),
                video_token_count=int(video_token_count),
                action_token_count=action_token_count,
                state_token_count=state_token_count,
                query_token_count=query_token_count,
                device=x.device,
                dtype=(q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16),
            )

        attn_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        # SDPA expects [B, N, L, D]; wan_attention's SDPA fallback does exactly
        # this transpose/cast.  attn_mask broadcasts over the B and N axes.
        q_sdpa = q.transpose(1, 2).to(attn_dtype)
        k_sdpa = k.transpose(1, 2).to(attn_dtype)
        v_sdpa = v.transpose(1, 2).to(attn_dtype)
        mask_sdpa = attn_mask.to(attn_dtype) if attn_mask is not None else None

        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask_sdpa,
            is_causal=False,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous()  # [B, L, N, D]

        out = out.flatten(2)
        out = attn_module.o(out)
        return out

    # ------------------------------------------------------------------
    # Inference / sampling forward -- parent's SELF-CONTAINED inference code flow.
    #
    # This is the method reached from predict_visual / predict_action ->
    # _sample_joint_predictions, i.e. the ``wan sample`` eval path AND infer.py
    # AR-state rollout inference.
    #
    # The baseline WanJointActionPolicy._run_joint_model that parent would
    # otherwise inherit appends NO K-query tokens, so parent's decoupling mask was
    # silently dropped at inference (state_token_count went negative).  parent
    # therefore overrides _run_joint_model and replicates the K-query
    # construction / modulation / sequence assembly of the training forward
    # ImaginationQueryPolicy._run_joint_model_with_queries -- WITHOUT the training-only
    # ``query_hidden`` return and ``align_loss``.  The sampling joint sequence
    # becomes [video | action | state | K-query], byte-identical in layout to
    # the training sequence, so the decoupling mask applies identically.
    #
    # This override is the ONLY change to parent's inference path.  parent's TRAINING
    # path is untouched: training keeps using the inherited
    # _run_joint_model_with_queries (which is already correct).
    # ------------------------------------------------------------------
    def _run_joint_model(
        self,
        *,
        noisy_latents: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context_list: List[torch.Tensor],
        condition_latents: torch.Tensor,
        latent_condition_mask: torch.Tensor,
        state_obs: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        autocast_context = contextlib.nullcontext()
        if model_device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            autocast_context = torch.autocast(device_type="cuda", dtype=model_dtype)

        noisy_latents = noisy_latents.to(device=model_device, dtype=model_dtype)
        condition_latents = condition_latents.to(device=model_device, dtype=model_dtype)
        latent_condition_mask = latent_condition_mask.to(device=model_device, dtype=model_dtype)
        noisy_actions = noisy_actions.to(
            device=model_device,
            dtype=next(self.action_token_encoder.parameters()).dtype,
        )

        with autocast_context:
            model_inputs = (
                latent_condition_mask * noisy_latents
                + (1.0 - latent_condition_mask) * condition_latents
            )
            video_embed = model.patch_embedding(model_inputs)
            grid_sizes = torch.tensor(
                [list(video_embed.shape[2:]) for _ in range(video_embed.shape[0])],
                dtype=torch.long,
                device=model_device,
            )
            video_tokens = video_embed.flatten(2).transpose(1, 2).contiguous()
            B = video_tokens.shape[0]
            video_token_count = int(video_tokens.shape[1])

            action_tokens = self.action_token_encoder(noisy_actions)
            action_tokens = action_tokens.to(device=model_device, dtype=model_dtype)
            action_token_count = int(action_tokens.shape[1])

            state_tokens = None
            if self.use_state_condition:
                state_tokens = self._encode_state_tokens(
                    state_obs,
                    model_device=model_device,
                    model_dtype=model_dtype,
                )
            state_token_count = int(state_tokens.shape[1]) if state_tokens is not None else 0

            # --- parent: append the K learnable imagination-query tokens, exactly
            # as ImaginationQueryPolicy._run_joint_model_with_queries does for the
            # training forward.  This is what makes the decoupling mask apply
            # at inference identically to training.
            query_tokens = None
            query_token_count = 0
            if self.K > 0:
                query_tokens = self.latent_queries.to(device=model_device, dtype=model_dtype)
                query_tokens = query_tokens.expand(B, -1, -1).contiguous()
                query_token_count = self.K

            parts = [video_tokens, action_tokens]
            if state_tokens is not None:
                parts.append(state_tokens)
            if query_tokens is not None:
                parts.append(query_tokens)
            total_tokens = torch.cat(parts, dim=1)

            # Time conditioning -- identical to the parent / training forward.
            video_timesteps = self._format_masked_timesteps_for_wan(
                timesteps,
                latent_condition_mask=latent_condition_mask,
                seq_len=video_token_count,
                device=model_device,
            )
            action_timesteps = self._format_timesteps_for_wan(
                timesteps,
                seq_len=action_token_count,
                device=model_device,
            )

            video_time_embed, video_mod = self._compute_time_condition(
                video_timesteps, model=model
            )
            action_time_embed, action_mod = self._compute_time_condition(
                action_timesteps, model=model
            )
            if action_mod.shape[1] == 1:
                action_mod = action_mod.expand(-1, action_token_count, -1, -1)

            modulation_parts = [video_mod, action_mod]
            if state_token_count > 0:
                modulation_parts.append(
                    torch.zeros(
                        B,
                        state_token_count,
                        6,
                        model.dim,
                        device=model_device,
                        dtype=torch.float32,
                    )
                )
            if query_token_count > 0:
                # Use the action-side time embed as the query modulation so
                # AdaLN-Zero stays on-distribution for the K-query positions --
                # identical to _run_joint_model_with_queries.
                query_mod = (
                    action_mod[:, :1].expand(-1, query_token_count, -1, -1).contiguous()
                )
                modulation_parts.append(query_mod)
            total_mod = torch.cat(modulation_parts, dim=1)

            # Hand the per-forward K-query count to _joint_self_attention so its
            # decoupling mask uses the ACTUAL layout of THIS forward.
            self._decouple_pending_query_tokens = int(query_token_count)

            if model.freqs.device != model_device:
                model.freqs = model.freqs.to(model_device)
            context = self._build_text_context(context_list, model=model)

            for block in model.blocks:
                total_tokens = self._joint_block_forward(
                    block=block,
                    tokens=total_tokens,
                    modulation=total_mod,
                    video_token_count=video_token_count,
                    grid_sizes=grid_sizes,
                    freqs=model.freqs,
                    context=context,
                )

            video_tokens = total_tokens[:, :video_token_count]
            action_tokens = total_tokens[
                :, video_token_count : video_token_count + action_token_count
            ]

            video_pred_tokens = model.head(video_tokens, video_time_embed)
            video_pred = model.unpatchify(video_pred_tokens, grid_sizes)
            video_pred = torch.stack([sample.float() for sample in video_pred], dim=0)
            action_pred = self.action_head(action_tokens, action_time_embed).float()

        # Restore the per-forward hint to its self.K default so a subsequent
        # forward that does NOT set it (e.g. the inherited training forward)
        # still sees the correct count.
        self._decouple_pending_query_tokens = int(self.K) if self.K > 0 else 0
        return video_pred, action_pred


__all__ = ["DecoupledPolicy"]
