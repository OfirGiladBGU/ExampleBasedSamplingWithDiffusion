"""Multi-oracle style-conditioned ControlNet (WVS / GBN / DITHER / ...).

Generalizes DynamicControlNetStyle from a scalar s to a K-dim style vector. A small MLP maps the
K-vector (one-hot at training, any convex combination at inference) to a vector ADDED to the
timestep embedding temb, conditioning every control resblock. Position-free, zero-init last layer
(identity to the unconditioned baseline at step 0). Only forward() is overridden; GECCO / hint /
injection paths are untouched. Base classes stay frozen in control_v4.
"""

import torch
import torch.nn as nn

from models.Layers import get_timestep_embedding
from control_v4.DynamicControlNet import (
    DynamicControlNet,
    DynamicControlledDenoiser,
)


def style_scalar_to_vec(s, style_dim=2):
    """Backward-compatible scalar addressing: map a single value s in [0,1] to a K-vector.

    For style_dim == 2 this returns [1 - s, s], reproducing the original single-value interface:
    s=0 -> oracle 0 (WVS), s=1 -> oracle 1 (GBN), s=0.5 -> [0.5, 0.5]. Only defined for K=2 (the
    simplex edge); for K > 2 there is no single canonical scalar axis, so pass an explicit vector.

    Accepts a python float or a tensor (any shape); returns a tensor with a trailing dim of size 2.
    """
    if int(style_dim) != 2:
        raise ValueError(
            f"scalar->vec addressing is only defined for 2 oracles; got K={style_dim}. "
            "Pass an explicit K-vector for K>2 (e.g. [0.5,0.5,0])."
        )
    s = torch.as_tensor(s, dtype=torch.float32)
    return torch.stack([1.0 - s, s], dim=-1)


class DynamicControlNetMultiStyle(DynamicControlNet):
    def __init__(self, *args, style_dim=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.style_dim = int(style_dim)
        hidden = self.ch * 4
        self.style_mlp = nn.Sequential(
            nn.Linear(self.style_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.style_mlp[-1].weight)
        nn.init.zeros_(self.style_mlp[-1].bias)

    def forward(
        self,
        offsets_t,
        t,
        high_res_image,
        target_density_map,
        high_res_sdf=None,
        target_sdf_map=None,
        target_smart_init_map=None,
        style_vec=None,
        timing_breakdown=None,
    ):
        B, _, H, W = offsets_t.shape
        cuda_timing_enabled = (
            timing_breakdown is not None
            and torch.cuda.is_available()
            and offsets_t.is_cuda
        )

        hint_parts = [offsets_t, target_density_map]
        if self.sdf_features:
            if target_sdf_map is None:
                raise ValueError("target_sdf_map is required when sdf_features=True")
            hint_parts.append(target_sdf_map)
        if self.smart_init_features:
            if target_smart_init_map is None:
                raise ValueError("target_smart_init_map is required when smart_init_features=True")
            hint_parts.append(target_smart_init_map)
        if self.batch_coords_features:
            coord_grid = self._time_block(
                timing_breakdown,
                "ctrl.grid_gen",
                lambda: self._build_coord_grid(H, W, offsets_t.device),
                cuda_timing_enabled,
            )
            hint_parts.append(coord_grid.expand(B, -1, -1, -1))

        if self.enable_gecco:
            gecco_dynamic = self.compute_gecco_features(
                offsets_t, high_res_image, high_res_sdf, timing_breakdown=timing_breakdown)
            hint_parts.append(gecco_dynamic)

        hint_input = torch.cat(hint_parts, dim=1)
        hint = self._time_block(
            timing_breakdown,
            "ctrl.hint_encode",
            lambda: self.input_hint_block(hint_input),
            cuda_timing_enabled,
        )

        x = self._time_block(
            timing_breakdown,
            "ctrl.conv1",
            lambda: self.ctrl_conv1(offsets_t),
            cuda_timing_enabled,
        )
        x = x + hint

        def build_temb():
            t_emb = get_timestep_embedding(t, self.ch * 4)
            t_emb = self.ctrl_dense1(t_emb)
            return self.ctrl_dense2(t_emb)

        temb = self._time_block(
            timing_breakdown,
            "ctrl.temb",
            build_temb,
            cuda_timing_enabled,
        )

        # ?? MULTI-STYLE INJECTION (the only change vs the base forward) ????????
        if style_vec is not None:
            s = torch.as_tensor(style_vec, device=temb.device, dtype=temb.dtype)
            s = s.reshape(-1, self.style_dim)
            if s.shape[0] == 1 and temb.shape[0] > 1:
                s = s.expand(temb.shape[0], -1)
            temb = temb + self.style_mlp(s)
        # ???????????????????????????????????????????????????????????????????????

        encoder_controls = []
        for enc_layer, dns, level_inj in zip(
            self.ctrl_encoder_layers,
            self.ctrl_downsamp_layers,
            self.injections,
        ):
            current_enc = []
            for layer, inj in zip(enc_layer, level_inj):
                x = self._time_block(
                    timing_breakdown,
                    "ctrl.down_blocks",
                    lambda layer=layer, x=x: layer(x, temb, None),
                    cuda_timing_enabled,
                )
                current_enc.append(
                    self._time_block(
                        timing_breakdown,
                        "ctrl.injections",
                        lambda inj=inj, x=x: inj(x),
                        cuda_timing_enabled,
                    )
                )
            encoder_controls.append(current_enc[::-1])

            x = self._time_block(
                timing_breakdown,
                "ctrl.down_blocks",
                lambda dns=dns, x=x: dns(x, temb, None),
                cuda_timing_enabled,
            )

        x = self._time_block(
            timing_breakdown,
            "ctrl.mid_block",
            lambda: self.ctrl_middle(x, temb, None),
            cuda_timing_enabled,
        )

        middle_control = self._time_block(
            timing_breakdown,
            "ctrl.injections",
            lambda: self.inject_middle(x),
            cuda_timing_enabled,
        )

        return (encoder_controls, middle_control)


class DynamicControlledMultiStyleDenoiser(DynamicControlledDenoiser):
    """Inference wrapper threading style_vec into the control branch (deferred-use: sampling/eval)."""

    def set_condition(self, high_res_image, high_res_sdf, target_density_map, target_sdf_map,
                      smart_init_grid, style_vec=None):
        super().set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map,
                              smart_init_grid)
        self._style_vec = style_vec

    def forward(self, x, t, cond=None, timing_breakdown=None):
        assert self._high_res_image is not None, "Call set_condition(..., style_vec=) first"
        hrs = self._high_res_image
        hrs_sdf = self._high_res_sdf
        tgt = self._target_density
        tgt_sdf = self._target_sdf
        smart = self._smart_init_grid
        if hrs.shape[0] != x.shape[0]:
            hrs = hrs.expand(x.shape[0], -1, -1, -1)
        if hrs_sdf is not None and hrs_sdf.shape[0] != x.shape[0]:
            hrs_sdf = hrs_sdf.expand(x.shape[0], -1, -1, -1)
        if tgt.shape[0] != x.shape[0]:
            tgt = tgt.expand(x.shape[0], -1, -1, -1)
        if tgt_sdf is not None and tgt_sdf.shape[0] != x.shape[0]:
            tgt_sdf = tgt_sdf.expand(x.shape[0], -1, -1, -1)
        if smart is not None and smart.shape[0] != x.shape[0]:
            smart = smart.expand(x.shape[0], -1, -1, -1)

        controls = self.control(
            x, t, hrs, tgt,
            high_res_sdf=hrs_sdf if self.control.sdf_features else None,
            target_sdf_map=tgt_sdf if self.control.sdf_features else None,
            target_smart_init_map=smart if self.control.smart_init_features else None,
            style_vec=getattr(self, "_style_vec", None),
            timing_breakdown=timing_breakdown,
        )
        return self.locked(x, t, cond=cond, controls=controls, timing_breakdown=timing_breakdown)
