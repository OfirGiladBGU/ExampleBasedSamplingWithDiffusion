"""ControlNet V3.8 -- SDF-aware conditioning with optional GECCO sampling.

Active injection path uses AdaptiveGateInjection (revert of V3.2) so control
signals are sigmoid-gated 1x1 projections, initialized near zero at step 1.

Hint input channels: offsets_t(2) + target_density(1) + target_sdf(1) +
coord_grid(2) = 6 (plus optional GECCO dynamic channels when enabled).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Layers import get_timestep_embedding


class AdaptiveGateInjection(nn.Module):
    """Sigmoid-gated 1x1 control injection (pre-V3.2 behavior).

    Computes ``sigmoid(gate(ctrl)) * transform(ctrl)`` so the frozen U-Net
    receives a spatially adaptive injected residual.
    """

    def __init__(self, channels):
        super().__init__()
        self.transform = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)

        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, ctrl_feature):
        g = torch.sigmoid(self.gate(ctrl_feature))
        return g * self.transform(ctrl_feature)


class DynamicControlNet(nn.Module):
    """Trainable control branch with SDF-aware static and dynamic conditioning.

    At each denoising step the module:
        1. Builds hint input from static channels
            [offsets_t(2), target_density(1), target_sdf(1), coord_grid(2)] and,
            when ``enable_gecco=True``, appends GECCO-sampled dynamic features
            from a high-resolution [image, sdf] feature map.
      2. Passes offsets_t through the copied conv1, adds the 128ch hint,
         then runs through the control encoder + middle blocks.
        3. Returns (encoder_controls, middle_control) tensors transformed by
            AdaptiveGateInjection, ready for
         additive injection into the frozen U-Net.

    Parameters
    ----------
    denoiser : DenoiserModel
        Pretrained (frozen) denoiser whose encoder + middle blocks are
        deep-copied to form the trainable control encoder.
    grid_size : int
        Spatial resolution of the offset grid (default 32).
    enable_gecco : bool
        If True, enable dynamic feature sampling from the high-res image
        using GECCO-style ``grid_sample`` at current offset positions.
    gecco_channels : int
        Number of dynamic channels produced by the GECCO feature extractor.
    """

    def __init__(self, denoiser, grid_size=32, enable_gecco=False, gecco_channels=16):
        super().__init__()
        self.ch = denoiser.ch
        self.grid_size = grid_size
        self.enable_gecco = enable_gecco
        self.gecco_channels = gecco_channels

        # ── fixed grid of cell centers in [0, 1] ────────────────────
        coords = torch.arange(grid_size, dtype=torch.float32) / grid_size + 0.5 / grid_size
        gx, gy = torch.meshgrid(coords, coords, indexing="xy")
        grid_centers = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, H, W)
        self.register_buffer("grid_centers", grid_centers)

        # ── static coordinate grid in [-1, 1] for spatial awareness ──
        lin = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        coord_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
        self.register_buffer("coord_grid", coord_grid)

        # ── optional GECCO feature extractor on high-res condition image ──
        if self.enable_gecco:
            self.gecco_extractor = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(8, gecco_channels, 3, padding=1),
                nn.SiLU(),
            )

        # ── trainable copies of encoder infrastructure ───────────────
        self.ctrl_dense1 = copy.deepcopy(denoiser.dense1)
        self.ctrl_dense2 = copy.deepcopy(denoiser.dense2)
        self.ctrl_conv1 = copy.deepcopy(denoiser.conv1)
        self.ctrl_encoder_layers = copy.deepcopy(denoiser.encoder_layers)
        self.ctrl_downsamp_layers = copy.deepcopy(denoiser.downsamp_layers)
        self.ctrl_middle = copy.deepcopy(denoiser.middle)

        # ── hint encoder: static 6ch (+ optional GECCO dynamic channels) ──
        hint_in_ch = 2 + 1 + 1 + 2  # offsets_t + target_density + target_sdf + coord_grid
        if self.enable_gecco:
            hint_in_ch += gecco_channels
        first_hidden = denoiser.conv1.net[1].weight.shape[0]  # ch_mult[0] = 128
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),   # ~13px receptive field
            nn.SiLU(),
            nn.Conv2d(64, first_hidden, 3, padding=4, dilation=4),  # ~29px
        )

        # ── injection layers (adaptive gated projection) ──────────────
        self.injections = nn.ModuleList()
        for level_layers in self.ctrl_encoder_layers:
            level_inj = nn.ModuleList()
            for block in level_layers:
                resblock = list(block.children())[0]
                out_ch = resblock.conv2.net[1].weight.shape[0]
                level_inj.append(AdaptiveGateInjection(out_ch))
            self.injections.append(level_inj)

        middle_first_resblock = list(self.ctrl_middle.children())[0]
        middle_ch = middle_first_resblock.conv2.net[1].weight.shape[0]
        self.inject_middle = AdaptiveGateInjection(middle_ch)

    def forward(self, offsets_t, t, high_res_image, high_res_sdf, target_density_map, target_sdf_map):
        """Run control encoder and return injection-ready control signals.

        Parameters
        ----------
        offsets_t : Tensor (B, 2, 32, 32)
        t : Tensor (B,)
        high_res_image : Tensor (B, 1, Himg, Wimg)
        high_res_sdf : Tensor (B, 1, Himg, Wimg)
        target_density_map : Tensor (B, 1, 32, 32)
        target_sdf_map : Tensor (B, 1, 32, 32)

        Returns
        -------
        tuple (encoder_controls, middle_control)
        """
        B = offsets_t.shape[0]

        batch_coords = self.coord_grid.expand(B, -1, -1, -1)

        hint_parts = [
            offsets_t,           # 2ch
            target_density_map,  # 1ch
            target_sdf_map,      # 1ch
            batch_coords,        # 2ch
        ]
        if self.enable_gecco:
            gecco_dynamic = self.compute_gecco_features(offsets_t, high_res_image, high_res_sdf)
            hint_parts.append(gecco_dynamic)

        hint_input = torch.cat(hint_parts, dim=1)
        hint = self.input_hint_block(hint_input)  # -> 128ch

        # Pass offsets through conv1 first, THEN add the hint
        x = self.ctrl_conv1(offsets_t)
        x = x + hint

        temb = get_timestep_embedding(t, self.ch * 4)
        temb = self.ctrl_dense1(temb)
        temb = self.ctrl_dense2(temb)

        encoder_controls = []
        for enc_layer, dns, level_inj in zip(
            self.ctrl_encoder_layers,
            self.ctrl_downsamp_layers,
            self.injections,
        ):
            current_enc = []
            for layer, inj in zip(enc_layer, level_inj):
                x = layer(x, temb, None)
                current_enc.append(inj(x))
            encoder_controls.append(current_enc[::-1])
            x = dns(x, temb, None)

        x = self.ctrl_middle(x, temb, None)
        middle_control = self.inject_middle(x)

        return (encoder_controls, middle_control)

    def compute_gecco_features(self, offsets_t, high_res_image, high_res_sdf):
        """Sample SDF-aware high-res GECCO features at current noisy positions."""
        if high_res_image is None or high_res_sdf is None:
            raise ValueError("high_res_image and high_res_sdf are required when enable_gecco=True")

        positions = self.grid_centers + offsets_t / self.grid_size
        sample_coords = positions.permute(0, 2, 3, 1) * 2.0 - 1.0

        gecco_feats_hr = self.gecco_extractor(torch.cat([high_res_image, high_res_sdf], dim=1))
        return F.grid_sample(
            gecco_feats_hr,
            sample_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


class DynamicControlledDenoiser(nn.Module):
    """Drop-in wrapper that plugs a DynamicControlNet V3 into a frozen denoiser.

    The ``forward(x, t, cond)`` signature matches ``DenoiserModel.forward``,
    so it can replace ``diffusion.model`` inside the existing sampling loop.

    Before calling ``forward``, call
    ``set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map)`` once.
    """

    def __init__(self, denoiser, dynamic_control_net):
        super().__init__()
        self.locked = denoiser
        self.control = dynamic_control_net
        self._high_res_image = None
        self._high_res_sdf = None
        self._target_density = None
        self._target_sdf = None

    def set_condition(self, high_res_image, high_res_sdf, target_density_map, target_sdf_map):
        self._high_res_image = high_res_image
        self._high_res_sdf = high_res_sdf
        self._target_density = target_density_map
        self._target_sdf = target_sdf_map

    def forward(self, x, t, cond=None):
        assert self._high_res_image is not None, (
            "Call set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map) first"
        )

        hrs = self._high_res_image
        hrs_sdf = self._high_res_sdf
        tgt = self._target_density
        tgt_sdf = self._target_sdf
        if hrs.shape[0] != x.shape[0]:
            hrs = hrs.expand(x.shape[0], -1, -1, -1)
        if hrs_sdf.shape[0] != x.shape[0]:
            hrs_sdf = hrs_sdf.expand(x.shape[0], -1, -1, -1)
        if tgt.shape[0] != x.shape[0]:
            tgt = tgt.expand(x.shape[0], -1, -1, -1)
        if tgt_sdf.shape[0] != x.shape[0]:
            tgt_sdf = tgt_sdf.expand(x.shape[0], -1, -1, -1)

        controls = self.control(x, t, hrs, hrs_sdf, tgt, tgt_sdf)
        return self.locked(x, t, cond=cond, controls=controls)
