"""Coupled Action Projection (CAP; paper Section 3.1).

A compact MLP and a time-modulated sequence head predict the same action-flow
velocity through their sum. The MLP output projection starts at zero. Internal
coarse/residual parameter names are retained for checkpoint compatibility; they
do not denote separately supervised targets or a physical action hierarchy."""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn

# CAP preserves the directional attention and joint-forward interface.
from rap_wam.policy.decoupled_policy import (
    DecoupledPolicy,
)
# The time-modulated branch reuses the parent action-head initialization.
from rap_wam.core.policy.wan_joint_action_policy import WanSequenceHead


class ResidualActionHead(nn.Module):
    """CAP: sum a compact MLP and time-modulated action-flow head.

    Both branches learn one shared velocity target. The compact branch output
    starts at zero; coarse/residual attribute names identify parameters only."""

    def __init__(
        self,
        *,
        dim: int,
        out_dim: int,
        coarse_head_hidden_mult: float = 0.25,
        coarse_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"ResidualActionHead: dim must be > 0; got {dim}.")
        if out_dim <= 0:
            raise ValueError(f"ResidualActionHead: out_dim must be > 0; got {out_dim}.")
        if coarse_head_hidden_mult < 0.0:
            raise ValueError(
                "ResidualActionHead: coarse_head_hidden_mult must be "
                f">= 0 (got {coarse_head_hidden_mult})."
            )
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.coarse_head_hidden_mult = float(coarse_head_hidden_mult)
        self.coarse_zero_init = bool(coarse_zero_init)

        # --- Residual branch: byte-identical to parent's monolithic action_head.
        self.residual_head = WanSequenceHead(dim=int(dim), out_dim=int(out_dim))

        # --- Coarse branch: small MLP, no AdaLN-time modulation by design.
        hidden = max(1, int(round(self.dim * self.coarse_head_hidden_mult)))
        self.coarse_hidden_dim = int(hidden)
        # Norm + project-up + GELU + project-down(out_dim).
        # The down-projection is zero-init when coarse_zero_init is true so
        # the coarse branch outputs zero at step 0.
        self.coarse_norm = nn.LayerNorm(self.dim, eps=1.0e-6)
        self.coarse_in = nn.Linear(self.dim, self.coarse_hidden_dim)
        self.coarse_act = nn.GELU()
        self.coarse_out = nn.Linear(self.coarse_hidden_dim, self.out_dim)
        if self.coarse_zero_init:
            # Zero the final projection so coarse_head(x) == 0 at step 0
            # regardless of upstream init.  Bias is also zeroed (default
            # for nn.Linear, set explicitly here for safety).
            with torch.no_grad():
                self.coarse_out.weight.zero_()
                if self.coarse_out.bias is not None:
                    self.coarse_out.bias.zero_()

        # Per-forward diagnostics (filled by forward, read by the policy
        # wrapper into _last_loss_components).  Floats, not tensors.
        self._last_coarse_norm: float = 0.0
        self._last_residual_norm: float = 0.0

    @torch.no_grad()
    def copy_residual_init_from(self, ref_head: WanSequenceHead) -> None:
        """Copy a fresh ``WanSequenceHead`` state into the residual branch.

        Called by ``ResidualActionPolicy.__init__`` after the
        parent constructs its own ``self.action_head``: we replace
        ``self.action_head`` with this wrapper and seed the wrapper's
        residual branch with the parent's freshly-constructed head -- so
        the residual branch's init is the parent-default init, identical
        to parent's action_head init at step 0.
        """
        # The two modules have IDENTICAL structure (same class, same args)
        # so state_dict load_state_dict(strict=True) is safe.
        state = ref_head.state_dict()
        self.residual_head.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # Residual branch -- byte-identical computation to parent's action_head.
        residual = self.residual_head(x, e)

        # Coarse branch -- small MLP, no AdaLN, no time-embed.
        coarse_h = self.coarse_norm(x)
        # Match the residual head's autocast policy.  WanSequenceHead.forward
        # wraps its inner ops in fp32 autocast on cuda; we mirror it so the
        # two branches' outputs are on the same dtype for the sum.
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float32,
            enabled=x.device.type == "cuda",
        ):
            coarse_h = self.coarse_in(coarse_h)
            coarse_h = self.coarse_act(coarse_h)
            coarse = self.coarse_out(coarse_h)

        # Cache per-forward diagnostics (detached scalars).
        try:
            self._last_coarse_norm = float(coarse.detach().float().abs().mean().item())
            self._last_residual_norm = float(residual.detach().float().abs().mean().item())
        except Exception:
            # Never let a diagnostic op crash training.
            pass

        return coarse + residual


class ResidualActionPolicy(DecoupledPolicy):
    """Replace the action projection with CAP while preserving its interface."""

    def __init__(
        self,
        *,
        coarse_head_hidden_mult: float = 0.25,
        coarse_zero_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(coarse_head_hidden_mult, (int, float)):
            raise TypeError(
                "ResidualActionPolicy: coarse_head_hidden_mult "
                f"must be float (got {type(coarse_head_hidden_mult).__name__})."
            )
        if float(coarse_head_hidden_mult) < 0.0:
            raise ValueError(
                "ResidualActionPolicy: coarse_head_hidden_mult "
                f"must be >= 0 (got {coarse_head_hidden_mult})."
            )
        self.coarse_head_hidden_mult = float(coarse_head_hidden_mult)
        self.coarse_zero_init = bool(coarse_zero_init)

        # Snapshot the parent-built action_head BEFORE we replace it -- the
        # residual branch will copy its init.
        parent_action_head: WanSequenceHead = self.action_head  # type: ignore[assignment]
        if not isinstance(parent_action_head, WanSequenceHead):
            raise TypeError(
                "ResidualActionPolicy: parent self.action_head must "
                f"be a WanSequenceHead; got {type(parent_action_head).__name__}."
            )

        # Build the parent wrapper with the same dim / out_dim the parent used.
        model_dim = int(self.model.dim)
        out_dim = int(self.action_dim)
        model_dtype = next(self.model.parameters()).dtype

        cap_head = ResidualActionHead(
            dim=model_dim,
            out_dim=out_dim,
            coarse_head_hidden_mult=self.coarse_head_hidden_mult,
            coarse_zero_init=self.coarse_zero_init,
        )
        # Seed the residual branch with the parent-default init.
        cap_head.copy_residual_init_from(parent_action_head)
        # Match the parent's dtype placement (parent did .to(dtype=model_dtype)).
        cap_head = cap_head.to(dtype=model_dtype)

        # SWAP: parent's wrapper replaces parent's action_head Module.  parent's two
        # joint forwards call ``self.action_head(action_tokens,
        # action_time_embed)`` -- now they call into the wrapper.
        self.action_head = cap_head

        # Sanity: confirm exactly the two submodules we expect.
        expected_submodules = {
            "residual_head",
            "coarse_norm",
            "coarse_in",
            "coarse_act",
            "coarse_out",
        }
        got = {n for n, _ in self.action_head.named_children()}
        missing = expected_submodules - got
        if missing:
            raise RuntimeError(
                "ResidualActionPolicy: wrapper missing submodules: "
                f"{sorted(missing)} (have {sorted(got)})."
            )

        # Pre-create _last_loss_components if absent (parent populates it
        # lazily inside compute_loss; we keep diagnostics sticky from step 0).
        if not hasattr(self, "_last_loss_components") or self._last_loss_components is None:
            self._last_loss_components = {}  # type: ignore[assignment]
        # Initial values (overwritten after every compute_loss step).
        self._last_loss_components["cap_coarse_head_hidden_mult"] = float(self.coarse_head_hidden_mult)
        self._last_loss_components["cap_coarse_zero_init"] = float(1.0 if self.coarse_zero_init else 0.0)
        self._last_loss_components["cap_coarse_norm"] = 0.0
        self._last_loss_components["cap_residual_norm"] = 0.0
        self._last_loss_components["cap_ratio"] = 0.0

    # ------------------------------------------------------------------
    # compute_loss: thin wrapper that pulls per-forward diagnostics from
    # the ResidualActionHead wrapper into the policy's
    # _last_loss_components dict (the workspace's logger auto-injects every
    # numeric field into logs.json.txt).
    #
    # No tensor / gradient touched.  Parent (parent -> ImaginationQueryPolicy) runs
    # exactly as before; parent only surfaces head-branch norms.
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, Any], training: bool = True):
        loss = super().compute_loss(batch, training=training)
        try:
            head = self.action_head
            if isinstance(head, ResidualActionHead) and isinstance(
                self._last_loss_components, dict
            ):
                coarse_norm = float(head._last_coarse_norm)
                residual_norm = float(head._last_residual_norm)
                eps = 1.0e-12
                ratio = coarse_norm / max(residual_norm, eps)
                self._last_loss_components["cap_coarse_norm"] = coarse_norm
                self._last_loss_components["cap_residual_norm"] = residual_norm
                self._last_loss_components["cap_ratio"] = float(ratio)
                self._last_loss_components["cap_coarse_head_hidden_mult"] = float(
                    self.coarse_head_hidden_mult
                )
                self._last_loss_components["cap_coarse_zero_init"] = float(
                    1.0 if self.coarse_zero_init else 0.0
                )
        except Exception:
            # Diagnostics must never crash training.
            pass
        return loss


__all__ = [
    "ResidualActionHead",
    "ResidualActionPolicy",
]
