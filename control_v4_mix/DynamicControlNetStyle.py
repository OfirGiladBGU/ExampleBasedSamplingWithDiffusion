"""Style-conditioned ControlNet for the WVS<->GBN spacing-regularity axis (Phase 2).

Adds a single continuous style scalar s to control_v4's DynamicControlNet by mapping s through a
small MLP to a vector ADDED to the timestep embedding temb. Because every control resblock
consumes temb, this conditions the whole control branch on s. The injection is position-free (s is
a global scalar; no positional embedding is introduced) and the final MLP layer is ZERO-INIT, so
at initialization the model is bit-identical to the unconditioned baseline and learns to use s.

Only forward() is overridden (copied from the parent with one added block) so the GECCO / hint /
injection paths are untouched. The base classes stay frozen in control_v4.
"""

import torch
import torch.nn as nn

from models.Layers import get_timestep_embedding
from control_v4.DynamicControlNet import (
    DynamicControlNet,
    DynamicControlledDenoiser,
)


class DynamicControlNetStyle(DynamicControlNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.ch * 4
        self.style_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Zero-init the last layer -> style contributes nothing at step 0 (identity to baseline).
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
        style_s=None,
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

        # ?? STYLE INJECTION (the only change vs the base forward) ??????????????
        # s -> vector added to temb. Zero-init MLP => no effect until trained.
        if style_s is not None:
            s = torch.as_tensor(style_s, device=temb.device, dtype=temb.dtype).reshape(-1, 1)
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


class DynamicControlledStyleDenoiser(DynamicControlledDenoiser):
    """Inference wrapper that threads style_s into the control branch. Deferred-use (sampling/eval);
    training calls the control net directly."""

    def set_condition(self, high_res_image, high_res_sdf, target_density_map, target_sdf_map,
                      smart_init_grid, style_s=None):
        super().set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map,
                              smart_init_grid)
        self._style_s = style_s

    def forward(self, x, t, cond=None, timing_breakdown=None):
        assert self._high_res_image is not None, (
            "Call set_condition(..., style_s=) first"
        )
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
            style_s=getattr(self, "_style_s", None),
            timing_breakdown=timing_breakdown,
        )
        return self.locked(x, t, cond=cond, controls=controls, timing_breakdown=timing_breakdown)
