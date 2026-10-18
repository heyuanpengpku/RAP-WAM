"""Transformer action decoder with timestep and observation conditioning. Implementation attribution: PSI Policy Team."""

from typing import Union, Optional, Tuple, List
import logging
import torch
import torch.nn as nn
import numpy as np
from rap_wam.core.model.diffusion.positional_embedding import SinusoidalPosEmb
from rap_wam.core.model.common.module_attr_mixin import ModuleAttrMixin
import math

from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ================================================================================

# ================================================================================

# ================================================================================
class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps using sinusoidal features and an MLP."""
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.float32):
        super().__init__()
        
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """Create sinusoidal features for a one-dimensional timestep tensor."""
        
        half = dim // 2
        
        
        
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        
        
        args = t[:, None].float() * freqs[None]
        
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding.to(self.dtype)

    def forward(self, t):
        """Compute the forward update for the input tensors."""
        
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        
        
        t_emb = self.mlp(t_freq)
        
        return t_emb

# ================================================================================

# ================================================================================

# ================================================================================
class CrossAttention(nn.Module):
    """Cross-attention from action tokens to observation-conditioning tokens."""
    fused_attn: Final[bool]  
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_mask_kwargs: dict = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ================================================================
        
        # ================================================================
        self.use_attn_mask = attn_mask_kwargs['use_attn_mask']
        if self.use_attn_mask:
            raw_part_length = attn_mask_kwargs['obs_part_length']
            max_cond_tokens = int(attn_mask_kwargs['max_cond_tokens'])
            action_horizon = int(attn_mask_kwargs['action_horizon'])

            part_lengths = [int(v) for v in raw_part_length]
            if len(part_lengths) < 2:
                raise ValueError(
                    "use_attn_mask=True requires at least two segments in obs_part_length."
                )
            if any(v <= 0 for v in part_lengths):
                raise ValueError(f"obs_part_length must contain positive integers; got {part_lengths}")
            if sum(part_lengths) != max_cond_tokens:
                raise ValueError(
                    "obs_part_length does not match max_cond_tokens: "
                    f"sum(obs_part_length)={sum(part_lengths)}, max_cond_tokens={max_cond_tokens}"
                )

            
            segment_ranges: list[tuple[int, int]] = []
            start = 0
            for length in part_lengths:
                end = start + length
                segment_ranges.append((start, end))
                start = end

            mask_candidates: list[torch.Tensor] = []
            for start, end in segment_ranges:
                
                mask = torch.ones((action_horizon, max_cond_tokens), dtype=torch.bool)
                mask[:, start:end] = False
                mask_candidates.append(mask)

            
            mask_candidates.append(torch.ones((action_horizon, max_cond_tokens), dtype=torch.bool))

            masks = torch.stack(mask_candidates, dim=0)
            self.register_buffer('masks', masks)

            
            masked_mode_count = len(mask_candidates) - 1
            probs = torch.full((len(mask_candidates),), 0.0, dtype=torch.float32)
            probs[:-1] = 0.3 / float(masked_mode_count)
            probs[-1] = 0.7
            self.register_buffer('probs', probs)

    def _sample_attn_mask(self, *, training: bool) -> Optional[torch.Tensor]:
        if not (self.use_attn_mask and training):
            return None
        idx = torch.multinomial(self.probs, num_samples=1)
        return self.masks[idx.squeeze()]

    def _forward_impl(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        training: bool,
        capture_attn_map: bool,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """Apply cross-attention from action features x [B,T,D] to conditioning features c [B,S,D]."""
        B, N, C = x.shape  
        _, L, _ = c.shape   
        
        # ================================================================
        
        # ================================================================
        # Q: (B, T, n_emb) -> (B, num_heads, T, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # ================================================================
        
        # ================================================================
        
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  
        
        # ================================================================
        
        # ================================================================
        q, k = self.q_norm(q), self.k_norm(k)

        # ================================================================
        
        # ================================================================
        attn_mask = self._sample_attn_mask(training=training)
        
        # ================================================================
        
        # ================================================================
        attn_weights = None
        if self.fused_attn:
            
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )
            
            if capture_attn_map:
                with torch.no_grad():
                    attn_weights = (q @ k.transpose(-2, -1)) * self.scale
                    if attn_mask is not None:
                        
                        attn_weights = attn_weights.masked_fill_(attn_mask.logical_not(), float('-inf'))
                    attn_weights = attn_weights.softmax(dim=-1)  # (B, num_heads, T, L)
                    
                    attn_weights = attn_weights.detach().cpu().to(torch.float16).numpy()
                    attn_weights = attn_weights[:2, :, 0, :]
        else:
            
            q = q * self.scale  
            attn = q @ k.transpose(-2, -1)  # (B, num_heads, T, L)
            
            
            if attn_mask is not None:
                attn = attn.masked_fill_(attn_mask.logical_not(), float('-inf'))
            
            attn = attn.softmax(dim=-1)  
            
            
            if capture_attn_map:
                attn_weights = attn.detach().cpu().to(torch.float16).numpy()  # (B, num_heads, T, L)
                attn_weights = attn_weights[:2, :, 0, :]
            
            # Dropout
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            
            
            x = attn @ v  # (B, num_heads, T, head_dim)
            
        # ================================================================
        
        # ================================================================
        
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        
        
        x = self.proj(x)
        
        
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
            
        return x, attn_weights

    def forward(self, x: torch.Tensor, c: torch.Tensor, training=True) -> torch.Tensor:
        x, _ = self._forward_impl(x, c, training=training, capture_attn_map=False)
        return x

    def forward_with_attn_map(self, x: torch.Tensor, c: torch.Tensor, training=True) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return self._forward_impl(x, c, training=training, capture_attn_map=True)


# ================================================================================

# ================================================================================


# ================================================================================
class RDTBlock(nn.Module):
    """Action transformer block with self-attention, cross-attention and a feed-forward network."""
    def __init__(self, hidden_size, num_heads, attn_mask_kwargs, **block_kwargs):
        super().__init__()
        
        # ================================================================
        
        # ================================================================
        
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True,  
            norm_layer=RmsNorm, **block_kwargs)
        
        # ================================================================
        
        # ================================================================
        
        
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm, 
            attn_mask_kwargs=attn_mask_kwargs, **block_kwargs)
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        
        # ================================================================
        
        # ================================================================
        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size, 
            hidden_features=hidden_size,  
            act_layer=approx_gelu, 
            drop=0  
        )
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, training=True):
        """Compute the forward update for the input tensors."""
        # ================================================================
        
        # ================================================================
        origin_x = x
        x = self.norm1(x)  
        x = self.attn(x)   
        x = x + origin_x   
        
        # ================================================================
        
        # ================================================================
        origin_x = x
        x = self.norm2(x)  
        x = self.cross_attn(x, c, training=training)
        x = x + origin_x   
                
        # ================================================================
        
        # ================================================================
        origin_x = x
        x = self.norm3(x)  
        x = self.ffn(x)    
        x = x + origin_x   
        
        return x

    def forward_with_attn_map(self, x, c, training=True):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        x, attn_weights = self.cross_attn.forward_with_attn_map(x, c, training=training)
        x = x + origin_x

        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        return x, attn_weights



# ================================================================================

# ================================================================================

# ================================================================================
class TransformerForAction(ModuleAttrMixin):
    """Transformer action-sequence predictor with timestep and observation conditioning."""
    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int=800,
        p_drop_attn: float = 0.1,
        obs_part_length: List[int] = None,
        use_attn_mask: bool = True,
        ) -> None:
        super().__init__()
        
        # ================================================================
        
        # ================================================================
        
        self.input_emb = nn.Linear(input_dim, n_emb)
        
        # ================================================================
        
        # ================================================================
        
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        
        
        self.cond_pos_emb = nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))

        attn_mask_kwargs = {
            'use_attn_mask': use_attn_mask,
            'obs_part_length': obs_part_length,
            'max_cond_tokens': max_cond_tokens,
            'action_horizon': action_horizon
        }

        # RDT blocks
        self.blocks = nn.ModuleList([
            RDTBlock(
                hidden_size=n_emb,
                num_heads=n_head,
                attn_drop=p_drop_attn,
                proj_drop=p_drop_attn,
                attn_mask_kwargs=attn_mask_kwargs
            )
            for _ in range(n_layer)
        ])

        # ================================================================
        
        # ================================================================
        
        
        
        
        self.ln_f = RmsNorm(n_emb, eps=1e-6)
        self.head = nn.Linear(n_emb, output_dim)  

        self.action_horizon = action_horizon
        
        # init
        self.apply(self._init_weights)
        logger.info(
            "Number of parameters in DiT: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            ## SinusoidalPosEmb, 
            TimestepEmbedder,
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.Embedding,
            nn.SiLU,
            nn.GELU,
            nn.Identity,
            RDTBlock)
        if isinstance(module, (nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (Attention, CrossAttention)):
        
            if hasattr(module, 'q'):
                torch.nn.init.normal_(module.q.weight, mean=0.0, std=0.02)
                if module.q.bias is not None:
                    torch.nn.init.zeros_(module.q.bias)
            if hasattr(module, 'k'):
                torch.nn.init.normal_(module.k.weight, mean=0.0, std=0.02)
                if module.k.bias is not None:
                    torch.nn.init.zeros_(module.k.bias)
            if hasattr(module, 'v'):
                torch.nn.init.normal_(module.v.weight, mean=0.0, std=0.02)
                if module.v.bias is not None:
                    torch.nn.init.zeros_(module.v.bias)
            if hasattr(module, 'proj'):
                torch.nn.init.normal_(module.proj.weight, mean=0.0, std=0.02)
                if module.proj.bias is not None:
                    torch.nn.init.zeros_(module.proj.bias)
        elif isinstance(module, Mlp):
            
            if hasattr(module, 'fc1'):
                torch.nn.init.normal_(module.fc1.weight, mean=0.0, std=0.02)
                if module.fc1.bias is not None:
                    torch.nn.init.zeros_(module.fc1.bias)
            if hasattr(module, 'fc2'):
                torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=0.02)
                if module.fc2.bias is not None:
                    torch.nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, (nn.LayerNorm)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RmsNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForAction):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif "q_norm" in fpn or "k_norm" in fpn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if hasattr(self, "cond_pos_emb"):
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def _forward_impl(
        self,
        sample: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        training: bool = True,
    ):
        cond_emb = cond
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond_emb + cond_pos_emb

        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = input_emb + pos_emb

        for block in self.blocks:
            x = block(x, cond_emb, training=training)

        x = self.ln_f(x)
        x = self.head(x)
        return x, None

    def forward_with_attn_map(
        self,
        sample: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        training: bool = True,
    ):
        cond_emb = cond
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond_emb + cond_pos_emb

        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = input_emb + pos_emb

        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward_with_attn_map(x, cond_emb, training=training)
            attention_weights.append(attn_weights)

        x = self.ln_f(x)
        x = self.head(x)
        return x, attention_weights

    def forward(self, 
        sample: torch.Tensor, 
        cond: Optional[torch.Tensor]=None,
        training=True,
        gen_attn_map=False, 
        **kwargs):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,N,n_emb)
        output: (B,T,input_dim)
        return: 
            if gen_attn_map:
                output tensor (B,T,input_dim), attention_maps List[(B,num_heads,T,L)]
            else:
                output tensor (B,T,input_dim), None
        """
        
        if gen_attn_map:
            return self.forward_with_attn_map(sample, cond=cond, training=training)
        return self._forward_impl(sample, cond=cond, training=training)
