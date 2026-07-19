"""aniso_control.py -- M4 anisotropy control branch for control_v4.

Architecture (option b): the trained stippling ControlNet -- frozen base
denoiser + frozen density control branch (ep10000) -- is left UNTOUCHED. A NEW
parallel control branch reads the (theta, kappa) field and injects ADDITIONAL
residuals into the same frozen U-Net. Only the new branch trains.

  AnisoControlNet          new branch; hint = [offsets, density, (cos2th,sin2th,logk)]
                           -> conv hint block (grid-res, translation-equivariant,
                           no absolute pos-enc -> grid-transfer safe) -> copied
                           encoder -> zero-init injections (starts as a no-op).
  DualControlledDenoiser   frozen denoiser + frozen density branch + trainable
                           aniso branch; sums both branches' controls.

Run scripts from the repo root OR let this module bootstrap sys.path.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from models.Layers import get_timestep_embedding
from control_v4.DynamicControlNet import DynamicControlNet


CONTROL_FIELD_CH = 3   # [cos2theta, sin2theta, logkappa] (magnitude-weighted)


class AnisoControlNet(DynamicControlNet):
    """Second control branch steering local anisotropy via a (theta,kappa) field.

    Built as a fresh copy of the frozen denoiser encoder (like any control
    branch) but with a hint block that additionally ingests the 3-channel
    control field. Injections are zero-initialized (inherited), so at step 0 the
    branch contributes nothing and the dual model is identical to the frozen one.
    """

    def __init__(self, denoiser, grid_size=32, include_density=True):
        # Minimal base features; we override the hint block below to add control.
        super().__init__(
            denoiser,
            grid_size=grid_size,
            enable_gecco=False,
            gecco_channels=0,
            enable_adaptive_gate_injection=True,
            smart_init_features=False,
            sdf_features=False,
            batch_coords_features=False,
        )
        self.include_density = bool(include_density)
        in_ch = 2 + (1 if self.include_density else 0) + CONTROL_FIELD_CH
        first_hidden = denoiser.conv1.net[1].weight.shape[0]
        # Same conv stack as the parent hint block, new input channel count.
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(64, first_hidden, 3, padding=4, dilation=4),
        )

    def forward(self, offsets_t, t, control_field, target_density_map=None):
        """control_field: (B, 3, G, G). Returns (encoder_controls, middle_control)."""
        parts = [offsets_t]
        if self.include_density:
            if target_density_map is None:
                raise ValueError("target_density_map required when include_density=True")
            parts.append(target_density_map)
        parts.append(control_field)
        hint = self.input_hint_block(torch.cat(parts, dim=1))

        x = self.ctrl_conv1(offsets_t) + hint

        temb = get_timestep_embedding(t, self.ch * 4)
        temb = self.ctrl_dense1(temb)
        temb = self.ctrl_dense2(temb)

        encoder_controls = []
        for enc_layer, dns, level_inj in zip(
            self.ctrl_encoder_layers, self.ctrl_downsamp_layers, self.injections
        ):
            current = []
            for layer, inj in zip(enc_layer, level_inj):
                x = layer(x, temb, None)
                current.append(inj(x))
            encoder_controls.append(current[::-1])
            x = dns(x, temb, None)

        x = self.ctrl_middle(x, temb, None)
        middle_control = self.inject_middle(x)
        return (encoder_controls, middle_control)


def sum_controls(a, b):
    """Elementwise-sum two (encoder_controls, middle_control) structures."""
    enc_a, mid_a = a
    enc_b, mid_b = b
    enc = [[ea + eb for ea, eb in zip(la, lb)] for la, lb in zip(enc_a, enc_b)]
    return (enc, mid_a + mid_b)


class DualControlledDenoiser(nn.Module):
    """Frozen base denoiser + frozen density branch + trainable aniso branch.

    forward(x, t, cond) matches DenoiserModel.forward so it can replace
    diffusion.model in the sampling loop. Call set_condition(...) once first.
    """

    def __init__(self, denoiser, density_control, aniso_control):
        super().__init__()
        self.locked = denoiser
        self.density = density_control     # frozen
        self.aniso = aniso_control         # trainable
        self._high_res_image = None
        self._target_density = None
        self._control_field = None

    def set_condition(self, high_res_image, target_density_map, control_field):
        self._high_res_image = high_res_image
        self._target_density = target_density_map
        self._control_field = control_field

    def _expand(self, tsr, b):
        if tsr is not None and tsr.shape[0] != b:
            return tsr.expand(b, *([-1] * (tsr.dim() - 1))).contiguous()
        return tsr

    def forward(self, x, t, cond=None, timing_breakdown=None):
        assert self._control_field is not None, "call set_condition(...) first"
        b = x.shape[0]
        hrs = self._expand(self._high_res_image, b)
        tgt = self._expand(self._target_density, b)
        ctl = self._expand(self._control_field, b)

        controls_density = self.density(
            x, t, hrs, tgt,
            high_res_sdf=None, target_sdf_map=None, target_smart_init_map=None,
        )
        controls_aniso = self.aniso(x, t, ctl, target_density_map=tgt)
        controls = sum_controls(controls_density, controls_aniso)
        return self.locked(x, t, cond=cond, controls=controls)


def offsets_to_coords(offsets, device=None):
    """(B,2,G,G) offset grid -> (B,G*G,2) point coords in [0,1].
    point = grid_center + offset / G   (matches DynamicControlNet GECCO mapping)."""
    B, _, G, W = offsets.shape
    cx = torch.arange(W, dtype=torch.float32, device=offsets.device) / W + 0.5 / W
    cy = torch.arange(G, dtype=torch.float32, device=offsets.device) / G + 0.5 / G
    gx, gy = torch.meshgrid(cx, cy, indexing="xy")     # matches _build_grid_centers
    centers = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1,2,G,G)
    pos = centers + offsets / G
    return pos.permute(0, 2, 3, 1).reshape(B, G * W, 2)


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module
