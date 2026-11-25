# model.py
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict, List

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


def temporal_patchify(videos_btchw: torch.Tensor, patch: int) -> torch.Tensor:
    """
    videos: (B,T,C,H,W) float in [0,1]
    returns: (B,T,Np,Dp) where Dp = patch*patch*C and Np = (H/patch)*(W/patch)
    """
    assert videos_btchw.dim() == 5
    B, T, C, H, W = videos_btchw.shape
    assert H % patch == 0 and W % patch == 0
    x = videos_btchw.reshape(B * T, C, H, W)
    cols = F.unfold(x, kernel_size=patch, stride=patch)          # (BT, C*pp, Np)
    cols = cols.transpose(1, 2).contiguous()                     # (BT, Np, Dp)
    Np, Dp = cols.shape[1], cols.shape[2]
    return cols.reshape(B, T, Np, Dp)


def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """
    patches: (B,T,Np,Dp) -> (B,T,C,H,W)
    """
    assert patches_btnd.dim() == 4
    B, T, Np, Dp = patches_btnd.shape
    assert Dp == C * patch * patch
    x = patches_btnd.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()  # (BT, Dp, Np)
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)  # (BT, C, H, W)
    return out.reshape(B, T, C, H, W)


class EmaRms(nn.Module):
    """
    Running root-mean-square normalizer using exponential moving average (EMA).
    Per the Dreamer 4 paper: "we normalize all loss terms by running estimates
    of their root-mean-square (RMS)." This makes loss coefficients interpretable
    as relative weights regardless of each term's absolute scale.
    """
    def __init__(self, decay: float = 0.99):
        super().__init__()
        self.decay = float(decay)
        self.register_buffer("sq_ema", torch.tensor(1.0))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        v = float(x.detach().float().item())
        self.sq_ema.mul_(self.decay).add_((1.0 - self.decay) * v * v)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.sq_ema.sqrt().clamp_min(1e-8)

    @torch.no_grad()
    def sync(self, world_size: int) -> None:
        """Average sq_ema across all DDP ranks so normalization is consistent."""
        import torch.distributed as dist
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(self.sq_ema, op=dist.ReduceOp.AVG)

    @property
    def rms_val(self) -> float:
        return float(self.sq_ema.sqrt().item())


class MAEReplacer(nn.Module):
    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches_btnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: (B,T,Np,D)
        returns:
          replaced: (B,T,Np,D)
          mae_mask: (B,T,Np,1) bool, True where masked (must reconstruct)
          keep_prob:(B,T,1) float
        """
        B, T, Np, D = patches_btnd.shape
        device = patches_btnd.device

        # fast path: deterministic "no MAE"
        if self.p_min == 0.0 and self.p_max == 0.0:
            keep_prob = torch.ones((B, T, 1), device=device, dtype=patches_btnd.dtype)
            mae_mask = torch.zeros((B, T, Np, 1), device=device, dtype=torch.bool)
            return patches_btnd, mae_mask, keep_prob

        p_bt = torch.empty((B, T), device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p_bt).unsqueeze(-1)                          # (B,T,1)
        keep = (torch.rand((B, T, Np), device=device) < keep_prob)      # (B,T,Np)
        keep_ = keep.unsqueeze(-1)
        mask_tok = self.mask_token.to(dtype=patches_btnd.dtype)
        replaced = torch.where(keep_, patches_btnd, mask_tok.view(1, 1, 1, D))
        mae_mask = (~keep_).to(torch.bool)
        return replaced, mae_mask, keep_prob


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cache_device = None  # torch.device

    @torch._dynamo.disable
    def compute_cos_sin(self, seq_len: int, *, device, dtype, offset: int = 0):
        """Uncached get_cos_sin: plain tensor ops, so a compiled graph needs no break."""
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def get_cos_sin(self, seq_len: int, *, device, dtype, offset: int = 0):
        needed = seq_len + offset
        need_new = (
            self._cos_cached is None or
            self._sin_cached is None or
            self._seq_len_cached < needed or
            self._cache_device != device
        )

        if need_new:
            # Build in fp32 and cache in fp32
            t = torch.arange(needed, device=device, dtype=torch.float32)  # (needed,)
            inv = self.inv_freq.to(device=device)                         # (dim/2,)
            freqs = torch.einsum("i,j->ij", t, inv)                       # (needed, dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)                       # (needed, dim)

            self._cos_cached = emb.cos()  # fp32
            self._sin_cached = emb.sin()  # fp32
            self._seq_len_cached = needed
            self._cache_device = device

        # Slice then cast to requested dtype (bf16/fp16/etc)
        cos = self._cos_cached[offset:offset + seq_len].to(dtype=dtype)
        sin = self._sin_cached[offset:offset + seq_len].to(dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., D) with D even, interleaved pairs
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (N, H, L, D), cos/sin: (L, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio * 2 / 3)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

        # RoPE (always on)
        self.rope = RotaryEmbedding(self.head_dim)

        # QKNorm (always on): per-head temperature, positive by construction.
        # Init g ~= sqrt(head_dim) so initial logit scale matches typical 1/sqrt(d) behavior.
        init = math.sqrt(self.head_dim)
        self.log_qk_scale = nn.Parameter(torch.full((n_heads,), math.log(init), dtype=torch.float32))

        # numeric stability for normalize
        self.qk_eps = 1e-6

    def forward(
        self,
        x_nld: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rope_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv_cache: bool = False,
    ):
        N, L, D = x_nld.shape
        q, k, v = self.qkv(x_nld).chunk(3, dim=-1)

        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)  # (N,H,L,hd)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE — when using KV cache, offset by the cached sequence length
        if kv_cache is not None:
            rope_offset = kv_cache[0].shape[2]
        rope = self.rope.compute_cos_sin if torch.compiler.is_compiling() else self.rope.get_cos_sin
        cos, sin = rope(L, device=x_nld.device, dtype=q.dtype, offset=rope_offset)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QKNorm: normalize along head_dim (cosine attention)
        q = F.normalize(q, p=2, dim=-1, eps=self.qk_eps)
        k = F.normalize(k, p=2, dim=-1, eps=self.qk_eps)

        # Save cache (normalized, RoPE'd K,V) before prepending cached entries
        new_cache = (k, v) if return_kv_cache else None

        # Prepend cached K,V for decode mode
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            is_causal = False
            attn_mask = None

        # Fold learnable per-head temperature into q so SDPA yields logits = g * (q·k)
        # SDPA internally multiplies by 1/sqrt(head_dim). We multiply q by g*sqrt(head_dim).
        g = self.log_qk_scale.exp().to(device=q.device, dtype=q.dtype)  # (H,)
        # Safety clamp to prevent runaway attention temperatures.
        g = g.clamp(0.0, 100.0)
        q = q * (g.view(1, self.n_heads, 1, 1) * math.sqrt(self.head_dim))

        drop = self.dropout_p if self.training else 0.0
        if L <= 4 and k.shape[2] == L and attn_mask is None and not is_causal and drop == 0.0:
            # A few keys (RecurrentHierarchy's pair axis, L=2): flash is slow at
            # this shape; elementwise ops fuse under compile. Same math.
            w = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1).float() / math.sqrt(self.head_dim)
            y = (w.softmax(-1).to(v.dtype).unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=drop,
                is_causal=is_causal,
            )
        y = y.transpose(1, 2).contiguous().view(N, L, D)

        if return_kv_cache:
            return self.out(y), new_cache
        return self.out(y)


class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_latents: int, mode: str, dropout: float):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)                               # (S,S) True=allowed
        attn_mask = torch.zeros(1, 1, S, S, dtype=torch.float32)
        attn_mask.masked_fill_(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)  # (S,1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)  # (1,S)

        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            allow_lat_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonlat_q = same_mod
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "wm_agent":
            # - Non-agent q (Action, Obs) -> all non-agent k (full mixing)
            # - Agent q   -> all keys
            # - Non-agent q never sees Agent k
            # Action and obs mix bidirectionally; agent tokens stay isolated from the rest.

            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))

            allow_for_agent_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonagent = ~is_k_agent  # non-agent queries see all non-agent keys

            return torch.where(is_q_agent, allow_for_agent_q, allow_nonagent)

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, latents_only: bool, n_latents: int,
                 causal: bool = True):
        super().__init__()
        self.causal = bool(causal)  # False: every block sees every block (RecurrentHierarchy's pair axis)
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(
        self,
        x_btSd: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        if self.latents_only:
            L = self.n_latents
            lat = x_btSd[:, :, :L, :]  # (B,T,L,D)
            lat_nld = lat.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)
            out = self.attn(lat_nld, is_causal=True)
            out = out.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            return torch.cat([out, x_btSd[:, :, L:, :]], dim=2)
        else:
            x_nld = x_btSd.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
            if return_kv_cache:
                out, cache = self.attn(x_nld, is_causal=True, return_kv_cache=True)
                out = out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()
                return out, cache
            elif kv_cache is not None:
                out = self.attn(x_nld, is_causal=True, kv_cache=kv_cache)
                out = out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()
                return out
            else:
                out = self.attn(x_nld, is_causal=self.causal)
                return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
        time_causal: bool = True,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)

        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout)
        self.drop1 = nn.Dropout(dropout)

        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents, causal=time_causal)
            self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        new_cache = None
        if self.do_time:
            if return_kv_cache:
                time_out, new_cache = self.time(self.norm2(x), return_kv_cache=True)
                x = x + self.drop2(time_out)
            elif kv_cache is not None:
                x = x + self.drop2(self.time(self.norm2(x), kv_cache=kv_cache))
            else:
                x = x + self.drop2(self.time(self.norm2(x)))
        x = x + self.mlp(self.norm3(x))
        if return_kv_cache:
            return x, new_cache
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
        time_causal: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(
                d_model=d_model, n_heads=n_heads, n_latents=n_latents,
                modality_ids=modality_ids, space_mode=space_mode,
                dropout=dropout, mlp_ratio=mlp_ratio,
                layer_index=i, time_every=time_every,
                latents_only_time=latents_only_time, time_causal=time_causal,
            )
            for i in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor:
        new_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [] if return_kv_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if return_kv_cache:
                x, cache = layer(x, return_kv_cache=True)
                new_caches.append(cache)
            elif layer_cache is not None:
                x = layer(x, kv_cache=layer_cache)
            else:
                x = layer(x)
        if return_kv_cache:
            return x, new_caches
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int,
        n_latents: int,
        n_patches: int,
        n_heads: int,
        depth: int,
        d_bottleneck: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = self.layout.modality_ids()  # CPU buffer, moves with .to(device)

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            n_latents=n_latents, modality_ids=modality_ids,
            space_mode="encoder",
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=latents_only_time,
        )
        self.mae = MAEReplacer(d_model=d_model, p_min=mae_p_min, p_max=mae_p_max)

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, patch_tokens_btnd: torch.Tensor):
        B, T, Np, Dp = patch_tokens_btnd.shape
        assert Np == self.n_patches

        proj = self.patch_proj(patch_tokens_btnd)            # (B,T,Np,D)
        proj_masked, mae_mask, keep_prob = self.mae(proj)    # (B,T,Np,D), (B,T,Np,1), (B,T,1)

        lat = self.latents.view(1, 1, self.n_latents, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)        # (B,T,S,D)

        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents, :]))
        return z, (mae_mask, keep_prob)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_bottleneck: int,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        n_patches: int,
        d_patch: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(d_model, d_patch)

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            n_latents=n_latents, modality_ids=modality_ids,
            space_mode="decoder",
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=latents_only_time,
        )

    def forward(self, z_btLd: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z_btLd.shape
        assert L == self.n_latents

        lat = self.up_proj(z_btLd)                                              # (B,T,L,D)
        qry = self.patch_queries.view(1, 1, self.n_patches, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, qry], dim=2)                                  # (B,T,S,D)

        x = self.transformer(tokens)
        x_p = x[:, :, L:, :]
        return torch.sigmoid(self.patch_head(x_p))                             # (B,T,Np,Dp)


class Tokenizer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, patches_btnd: torch.Tensor):
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob


def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    """
    z: (B,T,L,D_b) where L == n_spatial * k
    -> (B,T,n_spatial,D_b*k)
    """
    B, T, L, D = z_btLd.shape
    assert L == n_spatial * k, f"L={L} must equal n_spatial*k={n_spatial*k}"
    return z_btLd.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z_btSd: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    z: (B,T,n_spatial,D_b*k) -> (B,T,n_spatial*k,D_b)
    """
    B, T, S, DK = z_btSd.shape
    assert DK % k == 0, f"D={DK} must be divisible by k={k}"
    D = DK // k
    return z_btSd.view(B, T, S * k, D)


class ActionEncoder(nn.Module):
    """
    Continuous actions in [-1,1], shape (B,T,A) -> token (B,T,1,D).
    If actions is None (unlabeled pretrain), emits a learned base token.
    """
    def __init__(self, d_model: int, action_dim: int = 16, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)

        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)

        self.fc1 = nn.Linear(self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)

        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],                 # (B,T,A) or None
        *,
        batch_time_shape: Optional[Tuple[int,int]] = None,
        act_mask: Optional[torch.Tensor] = None,         # (B,T,A) or (A,)
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions
            if act_mask is not None:
                x = x * act_mask
            x = x.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base.view(1, 1, -1)

        return out[:, :, None, :]


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.expm1(y.abs()))


@torch.no_grad()
def twohot_from_symlog(y: torch.Tensor, centers_log: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    y: (...,) in symlog-space
    centers_log: (K,) monotonically increasing in symlog-space
    returns: (..., K) two-hot distribution (linear interp between neighbors)
    """
    K = centers_log.numel()
    y = y.clamp(centers_log[0], centers_log[-1])
    y1 = y.unsqueeze(-1)  # (...,1)

    idx = torch.searchsorted(centers_log, y1, right=False).clamp(1, K - 1)  # (...,1)
    lo = centers_log.gather(0, (idx - 1).view(-1)).view_as(idx).to(y.dtype)
    hi = centers_log.gather(0, idx.view(-1)).view_as(idx).to(y.dtype)

    w_hi = (y1 - lo) / (hi - lo).clamp_min(eps)
    w_lo = 1.0 - w_hi

    out = torch.zeros((*y.shape, K), device=y.device, dtype=y.dtype)
    out.scatter_(-1, idx, w_hi)
    out.scatter_(-1, idx - 1, w_lo)
    return out


def dist_cross_entropy_from_symlog(
    logits: torch.Tensor,          # (..., K)
    target_symlog: torch.Tensor,   # (...,)
    centers_log: torch.Tensor,     # (K,)
    mask: Optional[torch.Tensor] = None,  # (...) bool/float
) -> torch.Tensor:
    with torch.no_grad():
        tgt = twohot_from_symlog(target_symlog, centers_log).to(dtype=logits.dtype)
    logp = logits.log_softmax(dim=-1)
    ce = -(tgt * logp).sum(dim=-1)  # (...)

    if mask is None:
        return ce.mean()
    m = mask.to(dtype=ce.dtype)
    return (ce * m).sum() / m.sum().clamp_min(1.0)


class RewardHeadMTP(nn.Module):
    """
    Plain MLP reward head over already task-conditioned agent tokens h_t.
    Per the paper, language/task conditioning enters via the agent tokens inside
    the dynamics transformer; the reward head simply reads off h_t.

    Input:  h_t: (B,T,n_agent,D) or (B,T,D)
    Output: logits: (B,T,L,K), centers_log: (K,)
    """
    def __init__(
        self,
        *,
        d_model: int,
        L: int = 8,
        num_bins: int = 101,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        log_low: float = -8.0,
        log_high: float = 8.0,
        pool_agent: str = "attn",  # "attn" | "mean" | "first"
    ):
        super().__init__()
        self.L = int(L)
        self.num_bins = int(num_bins)
        self.d_model = int(d_model)
        self.pool_agent = pool_agent
        if pool_agent not in ("attn", "mean", "first"):
            raise ValueError(f"pool_agent must be one of attn|mean|first, got {pool_agent}")

        # Attention pool: a single learnable query reads over the n_agent tokens,
        # letting the head put task-dependent weights on each agent slot.
        if pool_agent == "attn":
            self.pool_query = nn.Parameter(torch.randn(self.d_model) * 0.02)
            self.pool_kv = nn.Linear(self.d_model, 2 * self.d_model, bias=False)

        self.projector = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(d_model, self.L * self.num_bins)

        centers = torch.linspace(log_low, log_high, self.num_bins, dtype=torch.float32)
        self.register_buffer("centers_log", centers, persistent=True)

        # Bias-init each MTP head toward the bin closest to symlog(0)=0 to match the
        # sparse-reward marginal, so initial CE is small. Weight init is left at
        # default so reward gradients still flow into the backbone from step 1.
        with torch.no_grad():
            zero_bin = int(torch.argmin(centers.abs()).item())
            bias = torch.full((self.L, self.num_bins), -5.0, dtype=self.out.bias.dtype)
            bias[:, zero_bin] = 5.0
            self.out.bias.copy_(bias.view(-1))

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pool agent tokens if present
        if h_t.dim() == 4:
            if self.pool_agent == "first":
                h = h_t[:, :, 0, :]
            elif self.pool_agent == "mean":
                h = h_t.mean(dim=2)
            else:  # attn
                B, T, N, D = h_t.shape
                kv = self.pool_kv(h_t)                                   # (B,T,N,2D)
                k, v = kv.chunk(2, dim=-1)                               # (B,T,N,D) each
                q = self.pool_query.to(dtype=k.dtype)                    # (D,)
                scores = (k * q).sum(dim=-1) / math.sqrt(D)              # (B,T,N)
                attn = scores.softmax(dim=-1)
                h = (attn.unsqueeze(-1) * v).sum(dim=2)                  # (B,T,D)
        else:
            h = h_t  # (B,T,D)

        B, T, _ = h.shape
        x = self.projector(h)
        logits = self.out(x).view(B, T, self.L, self.num_bins)
        return logits, self.centers_log


class PolicyHeadMTP(nn.Module):
    """
    Deterministic-MSE BC policy head over task-conditioned agent tokens h_t.
    Mirrors RewardHeadMTP's attn-pool + projector + MTP slicing, but outputs
    L x act_dim_max real-valued action means (tanh-squashed to [-1, 1]) rather
    than logits over bins — matches our continuous action space.

    Task conditioning arrives through the agent tokens (initialized from
    task_proj(lang_emb) inside the dynamics transformer); the head itself is
    task-agnostic.

    **Gradient isolation**: callers are expected to pass `h_t.detach()` so that
    BC gradients update only this head's parameters, not the dynamics /
    agent-token init. This is a deliberate deviation from the Dreamer-4 paper,
    which backprops BC gradients into the transformer.

    Input:  h_t: (B,T,n_agent,D) or (B,T,D)
    Output: action_means: (B,T,L,A) in [-1, 1]
    """
    def __init__(
        self,
        *,
        d_model: int,
        L: int = 8,
        act_dim_max: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        pool_agent: str = "attn",  # "attn" | "mean" | "first"
    ):
        super().__init__()
        self.L = int(L)
        self.act_dim_max = int(act_dim_max)
        self.d_model = int(d_model)
        self.pool_agent = pool_agent
        if pool_agent not in ("attn", "mean", "first"):
            raise ValueError(f"pool_agent must be one of attn|mean|first, got {pool_agent}")

        if pool_agent == "attn":
            self.pool_query = nn.Parameter(torch.randn(self.d_model) * 0.02)
            self.pool_kv = nn.Linear(self.d_model, 2 * self.d_model, bias=False)

        self.projector = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(d_model, self.L * self.act_dim_max)

        # Small-normal weight init + zero bias: predicted means start near 0 (small
        # initial BC loss) while still letting gradient flow through `out` into the
        # projector and attn-pool from step 1 (a zero-weight init would block that
        # gradient until Adam moves the weights off zero).
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        # pool agent tokens if present
        if h_t.dim() == 4:
            if self.pool_agent == "first":
                h = h_t[:, :, 0, :]
            elif self.pool_agent == "mean":
                h = h_t.mean(dim=2)
            else:  # attn
                B, T, N, D = h_t.shape
                kv = self.pool_kv(h_t)                                   # (B,T,N,2D)
                k, v = kv.chunk(2, dim=-1)                               # (B,T,N,D) each
                q = self.pool_query.to(dtype=k.dtype)                    # (D,)
                scores = (k * q).sum(dim=-1) / math.sqrt(D)              # (B,T,N)
                attn = scores.softmax(dim=-1)
                h = (attn.unsqueeze(-1) * v).sum(dim=2)                  # (B,T,D)
        else:
            h = h_t  # (B,T,D)

        B, T, _ = h.shape
        x = self.projector(h)
        means = self.out(x).view(B, T, self.L, self.act_dim_max)
        return torch.tanh(means)


class RecurrentHierarchy(nn.Module):
    """A stack of recurrent levels under a per-frame denoising head (replaces the
    block-causal transformer when --arch recurrent).

    Level n carries a state S_n (B,S,D). Each frame t, bottom to top, unit n reads
    the 2-block input (S_n^{t-1}, λ_{n-1}^t) (λ_0 = the frame's tokens x_t) and
    writes (S_n^t, λ_n^t): the next state (RMS-normalized) and the upward signal.
    Every unit layer runs space attention within each block and attention across
    the two blocks at the same slot, so slot i of the state meets slot i of its
    input. Time is only the recurrence: cost per frame is constant.

    The head is the per-frame denoiser: it reads (x_t, S_N^{t-1}), the frame and
    the top state from before the frame, with cross-block attention every other
    layer. Its agent slots are replaced by λ_N^t, so the reward / BC heads read
    the top level. Flow and BC gradients reach every level top-down (through the
    head and λ_N) and back in time (through the carried states).

    Local loss (aux_loss, (B,T-1)): level n predicts its next input λ_{n-1}^{t+1}
    from S_n^t, MLP(LN(S_n^t + c_{t+1})), c = the next frame's action and shortcut
    tokens. S_n^t is recomputed by re-running unit n on detached inputs, so this
    loss trains only unit n and its predictor (no gradient into the level below,
    which writes the target, and none back in time). The predictors are training
    only. The trainer weights it by the target frame's signal level.

    kv_cache is the list of per-level states. A prefill (return_kv_cache=True)
    runs the scan and returns the new states. A denoise step (cache set,
    return_kv_cache=False, one frame) reads the cached states without writing
    them, so K noisy passes do not advance the state.
    """

    def __init__(self, *, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_levels: int,
                 unit_layers: int, head_layers: int, dropout: float, mlp_ratio: float,
                 unit_mlp_ratio: Optional[float] = None):
        super().__init__()
        assert n_levels >= 1
        self.d_model, self.n_heads, self.n_levels = int(d_model), int(n_heads), int(n_levels)
        self.dropout = float(dropout)
        ids = modality_ids.detach().to(dtype=torch.int32).cpu().clone()
        self.S = int(ids.numel())
        # Slots as plain slices: a GPU bool-mask index would force a host sync on every scan step.
        agent_idx = (ids == int(Modality.AGENT)).nonzero().flatten().tolist()
        assert not agent_idx or agent_idx == list(range(agent_idx[0], agent_idx[-1] + 1)), "agent slots must be contiguous"
        self.agent_slice = slice(agent_idx[0], agent_idx[-1] + 1) if agent_idx else None
        cidx = [i for i, m in enumerate(ids.tolist()) if m in (int(Modality.SHORTCUT_SIGNAL), int(Modality.ACTION))]
        assert cidx == list(range(cidx[0], cidx[-1] + 1)), "action/shortcut slots must be contiguous"
        self.cond_slice = slice(cidx[0], cidx[-1] + 1)
        u_mlp = float(mlp_ratio if unit_mlp_ratio is None else unit_mlp_ratio)
        self.units = nn.ModuleList([self._stack(ids, int(unit_layers), u_mlp, 1) for _ in range(self.n_levels)])
        self.head = self._stack(ids, int(head_layers), float(mlp_ratio), 2)
        self.state_norms = nn.ModuleList([RMSNorm(self.d_model) for _ in range(self.n_levels)])
        self.aux_pred = nn.ModuleList([MLP(self.d_model, 1.0) for _ in range(self.n_levels)])  # training only
        self.aux_loss: Optional[torch.Tensor] = None
        self.aux_levels: Optional[torch.Tensor] = None
        # What the scan calls: the stacks, or compiled wrappers (compile_stacks). A dict,
        # so the wrappers are not registered as submodules and the state dict keeps its keys.
        self._run = {"units": list(self.units), "head": self.head, "scan": self._scan_step}

    def _stack(self, ids: torch.Tensor, depth: int, mlp_ratio: float, time_every: int) -> BlockCausalTransformer:
        # The "time" axis holds the stacked input blocks (non-causal); real time is the recurrence.
        return BlockCausalTransformer(
            d_model=self.d_model, n_heads=self.n_heads, depth=depth, n_latents=0, modality_ids=ids,
            space_mode="wm_agent", dropout=self.dropout, mlp_ratio=mlp_ratio, time_every=time_every,
            latents_only_time=False, time_causal=False,
        )

    def compile_stacks(self) -> None:
        """torch.compile the units, the head and one scan step (same math; the scan is launch-bound)."""
        cfg = torch._dynamo.config
        for name in ("recompile_limit", "cache_size_limit"):  # new name, old alias
            if hasattr(cfg, name):
                setattr(cfg, name, max(getattr(cfg, name), 64))
        self._run = {"units": [torch.compile(u) for u in self.units], "head": torch.compile(self.head)}
        # ponytail: no CUDA graphs. mode="reduce-overhead" on the scan step stalls in training (torch 2.8): T graph
        # calls per forward before one backward miss cudagraph trees' fast path ("pending, uninvoked backwards").
        # Upgrade path: capture the whole train step as one graph.
        self._run["scan"] = torch.compile(self._scan_step)

    def _unit(self, i: int, s_in: torch.Tensor, lam_in: torch.Tensor):
        """Unit i on (S^{t-1}, λ^t) for a batch of frames: (S^t, λ_i^t)."""
        y = self._run["units"][i](torch.stack([s_in, lam_in], dim=1))
        return self.state_norms[i](y[:, 0]), y[:, 1]

    def _scan_step(self, states: List[torch.Tensor], lam: torch.Tensor):
        new_s, lams = [], []
        for i in range(self.n_levels):
            s, lam = self._unit(i, states[i], lam)
            new_s.append(s)
            lams.append(lam)
        return new_s, lams

    def _head_out(self, tok: torch.Tensor, s_top: torch.Tensor) -> torch.Tensor:
        return self._run["head"](torch.stack([tok, s_top], dim=1))[:, 0]

    def _mix(self, h: torch.Tensor, lam_top: torch.Tensor) -> torch.Tensor:
        a = self.agent_slice
        return h if a is None else torch.cat([h[:, : a.start], lam_top[:, a], h[:, a.stop :]], dim=1)

    def _aux_local(self, x: torch.Tensor, ss: List, lams: List, init: List[torch.Tensor]) -> None:
        # All T-1 steps of a level in one batched call: the (detached) inputs are known once the scan has run.
        B, T, S, D = x.shape
        c = x[:, 1:, self.cond_slice].detach().sum(2, keepdim=True)  # frame t+1's action + shortcut tokens
        errs = []
        for n in range(self.n_levels):
            below = x if n == 0 else torch.stack([l[n - 1] for l in lams], 1)  # λ_{n-1}^t, all t
            s_in = torch.stack([init[n]] + [ss[t][n] for t in range(T - 2)], 1)  # S_n^{t-1}, t < T-1
            s, _ = self._unit(n, s_in.detach().flatten(0, 1), below[:, :-1].detach().flatten(0, 1))
            pred = self.aux_pred[n](F.layer_norm(s.view(B, T - 1, S, D) + c, (D,)))
            tgt = F.layer_norm(below[:, 1:].float(), (D,)).detach()
            errs.append((F.layer_norm(pred.float(), (D,)) - tgt).pow(2).mean(dim=(2, 3)))  # (B,T-1)
        per = torch.stack(errs)  # (N,B,T-1)
        self.aux_levels = per.mean(dim=(1, 2)).detach()  # per-level error, for logging
        self.aux_loss = per.mean(0)

    def forward(self, x: torch.Tensor, kv_cache: Optional[List[torch.Tensor]] = None,
                return_kv_cache: bool = False):
        B, T, S, _ = x.shape
        assert S == self.S
        self.aux_loss = self.aux_levels = None
        if kv_cache is not None and not return_kv_cache:  # denoise step: the carried states stay put
            assert T == 1, "denoise step takes one frame"
            h = self._head_out(x[:, 0], kv_cache[-1])
            if self.agent_slice is not None:  # the BC / reward heads read λ_N of this frame
                h = self._mix(h, self._scan_step(kv_cache, x[:, 0])[1][-1])
            return h.unsqueeze(1)
        states = list(kv_cache) if kv_cache is not None else [x.new_zeros(B, S, self.d_model)] * self.n_levels
        init = states
        ss, lams, prev = [], [], []
        for t in range(T):  # only the (small) units recur
            prev.append(states[-1])  # what frame t's head reads
            states, lam = self._run["scan"](states, x[:, t])
            ss.append(states)
            lams.append(lam)
        # The head reads only states the scan already wrote, so it runs once on all frames.
        lam_top = torch.stack([l[-1] for l in lams], 1).flatten(0, 1)
        y = self._mix(self._head_out(x.flatten(0, 1), torch.stack(prev, 1).flatten(0, 1)), lam_top).view(B, T, S, -1)
        if T > 1 and torch.is_grad_enabled():
            self._aux_local(x, ss, lams, init)
        if return_kv_cache:
            return y, states
        return y


class Dynamics(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        depth: int,
        k_max: Optional[int] = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        lang_dim: int = 0,
        arch: str = "transformer",
        n_levels: int = 4,
        unit_layers: int = 1,
        head_layers: int = 10,
        unit_mlp_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.arch = str(arch)
        assert d_spatial % d_bottleneck == 0, "expected packing: d_spatial = d_bottleneck * packing_factor"
        self.d_model = int(d_model)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.n_register = int(n_register)
        self.n_agent = int(n_agent)
        self.lang_dim = int(lang_dim)
        assert k_max is not None, "k_max must be provided"
        self.k_max = int(k_max)

        # Task projection: maps lang_emb -> initial agent token values (finetuning phase).
        # Only created when lang_dim > 0 and n_agent > 0.
        if self.n_agent > 0 and self.lang_dim > 0:
            self.task_proj = nn.Linear(self.lang_dim, self.n_agent * self.d_model)
            nn.init.normal_(self.task_proj.weight, std=0.02)
            nn.init.zeros_(self.task_proj.bias)
        else:
            self.task_proj = None

        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=16)

        # shortcut conditioning: σ and d embeddings are half-sized and concatenated
        # into a single token per the paper ("their channels are concatenated").
        assert self.d_model % 2 == 0, "d_model must be even for shortcut token channel concatenation"
        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embedding(self.num_step_bins, self.d_model // 2)
        self.signal_embed = nn.Embedding(self.k_max + 1, self.d_model // 2)

        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),  # combined σ+d token
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))
        modality_ids = self.layout.modality_ids()

        if self.arch == "recurrent":
            self.transformer = RecurrentHierarchy(
                d_model=self.d_model, n_heads=int(n_heads), modality_ids=modality_ids, n_levels=int(n_levels),
                unit_layers=int(unit_layers), head_layers=int(head_layers), dropout=float(dropout),
                mlp_ratio=float(mlp_ratio), unit_mlp_ratio=unit_mlp_ratio,
            )
        elif self.arch == "transformer":
            self.transformer = BlockCausalTransformer(
                d_model=self.d_model,
                n_heads=int(n_heads),
                depth=int(depth),
                n_latents=0,
                modality_ids=modality_ids,
                space_mode="wm_agent",
                dropout=float(dropout),
                mlp_ratio=float(mlp_ratio),
                time_every=int(time_every),
                latents_only_time=False,
            )
        else:
            raise ValueError(f"arch must be transformer|recurrent, got {self.arch}")

        self.flow_x_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],          # (B,T,16) or None
        step_idxs: Optional[torch.Tensor],        # (B,T)
        signal_idxs: Optional[torch.Tensor],      # (B,T)
        packed_enc_tokens: torch.Tensor,          # (B,T,n_spatial,d_spatial)
        *,
        act_mask: Optional[torch.Tensor] = None,  # (B,T,16) or (16,) or None
        agent_tokens: Optional[torch.Tensor] = None,
        lang_emb: Optional[torch.Tensor] = None,  # (B,lang_dim) task embedding
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_kv_cache: bool = False,
    ):
        B, T = packed_enc_tokens.shape[:2]

        spatial_tokens = self.spatial_proj(packed_enc_tokens)  # (B,T,n_spatial,d_model)

        action_tokens = self.action_encoder(
            actions,
            batch_time_shape=(B, T),
            act_mask=act_mask,
        )  # (B,T,1,d_model)

        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)

        assert step_idxs is not None and signal_idxs is not None, \
            "step_idxs/signal_idxs are required"
        sig_emb  = self.signal_embed(signal_idxs.to(torch.long))           # (B,T,d_model//2)
        step_emb = self.step_embed(step_idxs.to(torch.long))               # (B,T,d_model//2)
        shortcut_tok = torch.cat([sig_emb, step_emb], dim=-1)[:, :, None, :]  # (B,T,1,d_model)

        if self.n_agent > 0:
            if agent_tokens is None:
                if self.task_proj is not None and lang_emb is not None:
                    # Project lang_emb to agent token initial values, broadcast over T.
                    agent_tokens = self.task_proj(lang_emb.to(dtype=spatial_tokens.dtype))  # (B, n_agent*d_model)
                    agent_tokens = agent_tokens.view(B, 1, self.n_agent, self.d_model).expand(B, T, -1, -1)
                else:
                    agent_tokens = torch.zeros((B, T, self.n_agent, self.d_model), device=spatial_tokens.device, dtype=spatial_tokens.dtype)
            toks = [action_tokens, shortcut_tok, spatial_tokens, reg, agent_tokens]
        else:
            toks = [action_tokens, shortcut_tok, spatial_tokens, reg]

        tokens = torch.cat(toks, dim=2)  # (B,T,S,D)

        if return_kv_cache:
            # kv_cache is forwarded so a recurrent prefill continues the carried states
            # (the attention stack ignores it and prefills from the given tokens).
            x, new_cache = self.transformer(tokens, kv_cache=kv_cache, return_kv_cache=True)
        elif kv_cache is not None:
            x = self.transformer(tokens, kv_cache=kv_cache)
        else:
            x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)  # (B,T,n_spatial,d_spatial)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]   # (B,T,n_agent,d_model)

        self.aux_loss = getattr(self.transformer, "aux_loss", None)  # recurrent: local loss (B,T-1)
        self.aux_levels = getattr(self.transformer, "aux_levels", None)
        if return_kv_cache:
            return x1_hat, h_t, new_cache
        return x1_hat, h_t


def recon_loss_from_mae(pred_btnd: torch.Tensor,
                        target_btnd: torch.Tensor,
                        mae_mask_btNp1: torch.Tensor) -> torch.Tensor:
    # mask: (B,T,Np,1) bool, True where masked
    mask = mae_mask_btNp1.to(dtype=torch.float32)  # (B,T,Np,1)

    # compute in fp32 to avoid fp16 overflow on reduction
    diff = (pred_btnd.float() - target_btnd.float())          # (B,T,Np,Dp)
    sq = diff.mul(diff) * mask                                # broadcast mask over Dp
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]        # (#masked patches) * Dp
    return sq.sum() / denom


def lpips_on_mae_recon(
    lpips_fn,
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    recon_masked_btnd = torch.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon = temporal_unpatchify(recon_masked_btnd.float(), H, W, C, patch)
    tgt   = temporal_unpatchify(target_btnd.float(),        H, W, C, patch)

    if subsample_frac < 1.0:
        B, T = recon.shape[:2]
        step = max(1, int(1.0 / subsample_frac))
        recon = recon[:, ::step]
        tgt   = tgt[:, ::step]

    recon = (recon.clamp(0, 1) * 2.0 - 1.0).float()
    tgt   = (tgt.clamp(0, 1)   * 2.0 - 1.0).float()

    B, T = recon.shape[:2]
    recon = recon.reshape(B * T, C, H, W)
    tgt   = tgt.reshape(B * T, C, H, W)

    with torch.autocast(device_type="cuda", enabled=False):
        lp = lpips_fn(recon, tgt)
    return lp.mean()
