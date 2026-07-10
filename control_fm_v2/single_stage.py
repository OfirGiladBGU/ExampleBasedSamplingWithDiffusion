"""Single-stage, attention-free conditioning for control_fm_v2 -- config-file free.

Same two conditioners as v1 (`concat` and `spade`), but:
  * builders read ``flow_matching.MODEL_CONFIG`` instead of a JSON path;
  * both gain an optional zero-init bottleneck attention block (Tier 2).

Conditioning (all aligned to the GxG offset grid):
    - target_density  (1)  -- primary condition
    - smart_init_grid (1)  -- optional; usually redundant under the smartinit coupling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Layers import (
    Dense, Conv2dSame, DownsampleLayer, UpsampleLayer,
    NormalizeLayer, IdentityWithEmbedding, get_timestep_embedding, nonlinearity,
)

from control_fm_v2.attention import BottleneckAttention, attach_bottleneck_attention
from control_fm_v2.flow_matching import _denoiser_kwargs, get_model_config, resolve_attention


def build_conditional_velocity_network(extra_in_channels, device="cpu", zero_init_out=True,
                                       model_config=None, attn_middle=None, attn_heads=None):
    """Velocity U-Net with ``extra_in_channels`` additional *input* channels (concat path)."""
    cfg = get_model_config(model_config)
    attn_middle, attn_heads = resolve_attention(cfg, attn_middle, attn_heads)
    kwargs = _denoiser_kwargs(cfg)
    kwargs["num_channels"] += int(extra_in_channels)
    from models.Denoiser import DenoiserModel

    model = DenoiserModel(**kwargs)
    if attn_middle:
        attach_bottleneck_attention(model, num_heads=attn_heads)
    if zero_init_out:
        # Zero-init the final conv so the net starts by predicting ~zero velocity and learns
        # residually -- standard (ADM/DiT) and consistent with the rest of this codebase.
        out_conv = model.out_conv.net[1]
        nn.init.zeros_(out_conv.weight)
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)
    return model.to(device)


class SingleStageConditioner(nn.Module):
    """One conditional velocity net: concat aligned conditions onto the state, then run."""

    def __init__(self, denoiser, use_smart_init_grid=False):
        super().__init__()
        self.denoiser = denoiser
        self.use_smart_init_grid = bool(use_smart_init_grid)
        self.n_cond = 1 + (1 if self.use_smart_init_grid else 0)
        self._target_density = None
        self._smart_init_grid = None

    def set_condition(self, target_density, smart_init_grid=None):
        self._target_density = target_density
        self._smart_init_grid = smart_init_grid
        if self.use_smart_init_grid and smart_init_grid is None:
            raise ValueError("use_smart_init_grid=True but no smart_init_grid was provided")

    def _cond_tensor(self, x):
        parts = [self._target_density]
        if self.use_smart_init_grid:
            parts.append(self._smart_init_grid)
        cond = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        cond = cond.to(device=x.device, dtype=x.dtype)
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="area")
        if cond.shape[0] != x.shape[0]:
            cond = cond.expand(x.shape[0], -1, -1, -1).contiguous()
        return cond

    def forward(self, x, t, timing_breakdown=None):
        assert self._target_density is not None, "call set_condition(target_density, ...) first"
        x_cat = torch.cat([x, self._cond_tensor(x)], dim=1)
        return self.denoiser(x_cat, t)


# ==============================================================================
# SPADE conditioning (Park et al., 2019) -- per-level, attention-free trunk.
# ==============================================================================


class SPADE(nn.Module):
    """GroupNorm(affine=False) then a per-pixel affine (gamma, beta) predicted from the
    condition map. gamma/beta are zero-initialised so the block starts as a plain
    normalization and learns the conditioning gradually."""

    def __init__(self, num_channels, cond_channels, hidden=64):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, 1e-6, affine=False)
        self.shared = nn.Conv2d(cond_channels, hidden, 3, padding=1)
        self.gamma = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, num_channels, 3, padding=1)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        normalized = self.norm(x)
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="nearest")
        c = nonlinearity(self.shared(cond))
        return normalized * (1.0 + self.gamma(c)) + self.beta(c)


class SPADEResnetBlock(nn.Module):
    """ResnetBlock with SPADE norms (mirrors models.Layers.ResnetBlock's data flow)."""

    def __init__(self, num_channels, emb_dim, cond_channels, out_channels=None, dropout=0.1):
        super().__init__()
        out_channels = out_channels or num_channels
        self.norm1 = SPADE(num_channels, cond_channels)
        self.conv1 = Conv2dSame(num_channels, out_channels)
        self.norm2 = SPADE(out_channels, cond_channels)
        self.conv2 = Conv2dSame(out_channels, out_channels)
        self.dense = Dense(emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            Conv2dSame(num_channels, out_channels, kernel_size=(1, 1))
            if num_channels != out_channels else nn.Identity()
        )

    def forward(self, x, emb, cond):
        h = nonlinearity(self.norm1(x, cond))
        h = self.conv1(h)
        h = h + self.dense(nonlinearity(emb))[..., None, None]
        h = nonlinearity(self.norm2(h, cond))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class SPADEUNet(nn.Module):
    """Velocity U-Net with SPADE conditioning at every ResnetBlock.

    ``attn_middle=True`` inserts a zero-init, position-encoding-free BottleneckAttention
    between the two middle blocks -- the same Tier-2 block used on the DenoiserModel path.
    The encoder/decoder trunk stays attention-free.
    """

    def __init__(self, in_ch, out_ch, ch, ch_mult, num_res, cond_channels, dropout=0.1,
                 attn_middle=False, attn_heads=4):
        super().__init__()
        self.ch = ch
        embdim = ch * 4
        chs = [ch * m for m in ch_mult]

        self.dense1 = Dense(embdim, embdim)
        self.dense2 = Dense(embdim, embdim)
        self.conv_in = Conv2dSame(in_ch, chs[0])

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for i, c in enumerate(chs):
            level = nn.ModuleList()
            for _ in range(num_res):
                level.append(SPADEResnetBlock(prev, embdim, cond_channels, out_channels=c, dropout=dropout))
                prev = c
            self.enc_blocks.append(level)
            self.downs.append(DownsampleLayer(prev) if i != len(chs) - 1 else IdentityWithEmbedding())

        # middle (+ optional bottleneck attention)
        self.mid1 = SPADEResnetBlock(prev, embdim, cond_channels, dropout=dropout)
        self.mid_attn = BottleneckAttention(prev, num_heads=attn_heads) if attn_middle else None
        self.mid2 = SPADEResnetBlock(prev, embdim, cond_channels, dropout=dropout)

        # decoder
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(chs))):
            enc_ch = chs[i]
            level = nn.ModuleList()
            pc = chs[i]
            for _ in range(num_res):
                level.append(SPADEResnetBlock(2 * enc_ch, embdim, cond_channels, out_channels=pc, dropout=dropout))
                pc = chs[i - 1] if i > 0 else chs[0]
            self.dec_blocks.append(level)
            self.ups.append(UpsampleLayer(pc) if i != 0 else IdentityWithEmbedding())

        self.out_norm = NormalizeLayer(chs[0])
        self.out_conv = Conv2dSame(chs[0], out_ch)

    def forward(self, x, t, cond):
        temb = get_timestep_embedding(t, self.ch * 4)
        temb = self.dense1(temb)
        temb = self.dense2(temb)

        x = self.conv_in(x)

        skips = []
        for level, down in zip(self.enc_blocks, self.downs):
            cur = []
            for block in level:
                x = block(x, temb, cond)
                cur.append(x)
            skips.append(cur[::-1])
            x = down(x, temb, cond)

        x = self.mid1(x, temb, cond)
        if self.mid_attn is not None:
            x = self.mid_attn(x, temb, cond)
        x = self.mid2(x, temb, cond)

        for level, up, tensors in zip(self.dec_blocks, self.ups, reversed(skips)):
            for block, enc in zip(level, tensors):
                x = block(torch.cat((x, enc), 1), temb, cond)
            x = up(x, temb, cond)

        x = nonlinearity(self.out_norm(x))
        x = self.out_conv(x)
        return x


def build_spade_velocity_network(cond_channels=1, device="cpu", zero_init_out=True,
                                 model_config=None, attn_middle=None, attn_heads=None):
    """Build the SPADE-conditioned velocity U-Net from MODEL_CONFIG."""
    m = get_model_config(model_config)
    attn_middle, attn_heads = resolve_attention(m, attn_middle, attn_heads)
    net = SPADEUNet(
        in_ch=int(m["num_channels"]), out_ch=int(m["out_ch"]),
        ch=int(m["ch"]), ch_mult=list(m["ch_mult"]),
        num_res=int(m["num_res"]), cond_channels=int(cond_channels),
        dropout=float(m.get("dropout", 0.1)),
        attn_middle=attn_middle, attn_heads=attn_heads,
    )
    if zero_init_out:
        out_conv = net.out_conv.net[1]
        nn.init.zeros_(out_conv.weight)
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)
    return net.to(device)


class SPADEConditioner(nn.Module):
    """Drop-in conditioner (set_condition / forward(x, t)) using SPADE per-level modulation."""

    def __init__(self, net, use_smart_init_grid=False):
        super().__init__()
        self.net = net
        self.use_smart_init_grid = bool(use_smart_init_grid)
        self._target_density = None
        self._smart_init_grid = None

    def set_condition(self, target_density, smart_init_grid=None):
        self._target_density = target_density
        self._smart_init_grid = smart_init_grid
        if self.use_smart_init_grid and smart_init_grid is None:
            raise ValueError("use_smart_init_grid=True but no smart_init_grid was provided")

    def _cond(self, x):
        parts = [self._target_density]
        if self.use_smart_init_grid:
            parts.append(self._smart_init_grid)
        c = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        c = c.to(device=x.device, dtype=x.dtype)
        if c.shape[0] != x.shape[0]:
            c = c.expand(x.shape[0], -1, -1, -1).contiguous()
        return c

    def forward(self, x, t, timing_breakdown=None):
        assert self._target_density is not None, "call set_condition(target_density, ...) first"
        return self.net(x, t, self._cond(x))