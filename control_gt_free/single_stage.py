"""Single-stage, attention-free conditioning for the control_fm Flow-Matching baseline.

Replaces the ControlNet dual-branch (deep-copied encoder + zero-conv/gate injections)
with the standard image-to-image conditioning used by conditional diffusion / flow
matching (Palette, SR3, I2SB): the aligned condition maps are **concatenated** onto the
input state and a single velocity U-Net consumes them. No parallel encoder, no attention.

Because the whole network is trained from scratch, there is no frozen prior to protect,
so the ControlNet machinery (whose purpose is forgetting-free grafting onto a frozen
backbone) buys nothing here -- a single conditional net is the right tool.

Conditioning (all aligned to the GxG offset grid, concatenated to the 2-ch state):
    - target_density  (1)  -- primary condition: the image / density map at grid resolution
    - smart_init_grid (1)  -- optional; usually redundant when the smart-init is already the
                              Flow-Matching coupling *source* (it enters via the state itself)

FiLM / SPADE modulation at each U-Net level is the natural next increment (keeps the
condition from washing out at depth); it needs backbone hooks and is intentionally left
out of this first concat-only version so it can be validated in isolation.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Layers import (
    Dense, Conv2dSame, DownsampleLayer, UpsampleLayer,
    NormalizeLayer, IdentityWithEmbedding, get_timestep_embedding, nonlinearity,
)


def build_conditional_velocity_network(config_path, extra_in_channels, device="cpu", zero_init_out=True):
    """Build the velocity U-Net with ``extra_in_channels`` additional *input* channels.

    Reuses the DenoiserModel architecture but widens ``conv1`` (num_channels) to accept the
    concatenated condition maps. The output channels (velocity, 2) are unchanged.
    """
    with open(config_path) as f:
        config = json.load(f)
    model_cfg = dict(config["model"])
    model_cfg.pop("cond", None)  # global-vector conditioning path is unused here
    model_cfg["num_channels"] = int(model_cfg["num_channels"]) + int(extra_in_channels)
    from models.Denoiser import DenoiserModel

    model = DenoiserModel(**model_cfg)
    if zero_init_out:
        # Zero-init the final conv so the net starts by predicting ~zero velocity and learns
        # residually -- the warm-start ControlNet gets for free from its zero-conv injections,
        # and a standard trick (ADM/DiT) for faster, more stable early convergence.
        out_conv = model.out_conv.net[1]
        nn.init.zeros_(out_conv.weight)
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)
    return model.to(device)


class SingleStageConditioner(nn.Module):
    """One conditional velocity net: concat aligned conditions onto the state, then run.

    Drop-in replacement for ``DynamicControlledDenoiser`` on the concat path. Call
    ``set_condition(target_density, smart_init_grid=None)`` once, then ``forward(x, t)``.
    """

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
        # match spatial size to the state (supports inference at a different grid size)
        if cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(cond, size=x.shape[-2:], mode="area")
        # broadcast a single condition across the sample batch
        if cond.shape[0] != x.shape[0]:
            cond = cond.expand(x.shape[0], -1, -1, -1).contiguous()
        return cond

    def forward(self, x, t, timing_breakdown=None):
        assert self._target_density is not None, "call set_condition(target_density, ...) first"
        x_cat = torch.cat([x, self._cond_tensor(x)], dim=1)
        return self.denoiser(x_cat, t)


# ==============================================================================
# SPADE conditioning (Park et al., 2019) -- attention-free, from-scratch, per-level.
# Self-contained U-Net mirroring DenoiserModel's structure (ch/ch_mult/num_res, no
# attention per the GBN config) but replacing every ResnetBlock GroupNorm with a
# spatially-adaptive normalization driven by the density map. No edits to shared
# model code, so control_v1..v4 stay untouched.
# ==============================================================================


class SPADE(nn.Module):
    """Spatially-adaptive normalization: GroupNorm(affine=False) then a per-pixel
    affine (gamma, beta) predicted from the condition map by small convs.

    gamma/beta are zero-initialized so the block starts as a plain normalization
    (neutral modulation) and learns the conditioning gradually -- the same warm-start
    idea as the zero-init output conv and ControlNet zero-convs.
    """

    def __init__(self, num_channels, cond_channels, hidden=64):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, 1e-6, affine=False)  # matches NormalizeLayer (num_groups=1)
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
    """Velocity U-Net with SPADE conditioning at every ResnetBlock (no attention)."""

    def __init__(self, in_ch, out_ch, ch, ch_mult, num_res, cond_channels, dropout=0.1):
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

        # middle
        self.mid1 = SPADEResnetBlock(prev, embdim, cond_channels, dropout=dropout)
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
        x = self.mid2(x, temb, cond)

        for level, up, tensors in zip(self.dec_blocks, self.ups, reversed(skips)):
            for block, enc in zip(level, tensors):
                x = block(torch.cat((x, enc), 1), temb, cond)
            x = up(x, temb, cond)

        x = nonlinearity(self.out_norm(x))
        x = self.out_conv(x)
        return x


def build_spade_velocity_network(config_path, cond_channels=1, device="cpu", zero_init_out=True):
    """Build the SPADE-conditioned velocity U-Net from the model config."""
    with open(config_path) as f:
        config = json.load(f)
    m = dict(config["model"])
    net = SPADEUNet(
        in_ch=int(m["num_channels"]), out_ch=int(m["out_ch"]),
        ch=int(m["ch"]), ch_mult=list(m["ch_mult"]),
        num_res=int(m["num_res"]), cond_channels=int(cond_channels),
        dropout=float(m.get("dropout", 0.1)),
    )
    if zero_init_out:
        out_conv = net.out_conv.net[1]
        nn.init.zeros_(out_conv.weight)
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)
    return net.to(device)


class SPADEConditioner(nn.Module):
    """Drop-in conditioner (set_condition / forward(x, t)) using SPADE per-level modulation.

    The density map (optionally + smart-init grid) modulates every ResnetBlock; the state
    is the only network input (no concat). Same interface as SingleStageConditioner so the
    training and sampling code paths treat both uniformly.
    """

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
