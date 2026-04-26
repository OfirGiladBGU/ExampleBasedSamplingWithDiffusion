"""Lightweight adapter-style control branch for control_t2i.

Compact T2I adapter hierarchy that emits the nested controls tuple expected by
DenoiserModel: (encoder_controls, middle_control).

Conditions on: noisy offsets + density map + optional GECCO high-res features.
No SDF, no smart-init, no coord-grid — lean and fast.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Module):
    """1x1 conv initialized to zero for stable adapter warm-up."""

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class DynamicLightweightAdapter(nn.Module):
    """T2I-Adapter style lightweight control network.

    The module consumes the same conditioning inputs as DynamicControlNet and
    returns controls with the same nested structure expected by DenoiserModel.
    """

    def __init__(self, denoiser, grid_size=32, enable_gecco=False, gecco_channels=16):
        super().__init__()
        self.ch = denoiser.ch
        self.grid_size = grid_size
        self.enable_gecco = enable_gecco
        self.gecco_channels = gecco_channels

        if self.enable_gecco:
            self.gecco_extractor = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(8, gecco_channels, 3, padding=1),
                nn.SiLU(),
            )

        # hint_in_ch: 2 (offsets_t) + 1 (density) [+ gecco_channels]
        hint_in_ch = 2 + 1
        if self.enable_gecco:
            hint_in_ch += gecco_channels

        first_hidden = denoiser.conv1.net[1].weight.shape[0]
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(64, first_hidden, 3, padding=4, dilation=4),
        )

        # Keep a single copied stem conv (small), but replace the full copied encoder.
        self.adapter_conv1 = copy.deepcopy(denoiser.conv1)

        self.adapter_levels = nn.ModuleList()
        self.adapter_downs = nn.ModuleList()
        self.injections = nn.ModuleList()

        current_ch = first_hidden
        n_levels = len(denoiser.encoder_layers)
        for level_idx, level_layers in enumerate(denoiser.encoder_layers):
            level_blocks = nn.ModuleList()
            level_inj = nn.ModuleList()

            for block in level_layers:
                resblock = list(block.children())[0]
                out_ch = resblock.conv2.net[1].weight.shape[0]

                level_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(current_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.SiLU(),
                    )
                )
                level_inj.append(ZeroConv2d(out_ch))
                current_ch = out_ch

            self.adapter_levels.append(level_blocks)
            self.injections.append(level_inj)

            if level_idx < n_levels - 1:
                self.adapter_downs.append(
                    nn.Sequential(
                        nn.Conv2d(current_ch, current_ch, kernel_size=3, stride=2, padding=1),
                        nn.SiLU(),
                    )
                )
            else:
                self.adapter_downs.append(nn.Identity())

        middle_first_resblock = list(denoiser.middle.children())[0]
        middle_ch = middle_first_resblock.conv2.net[1].weight.shape[0]
        self.middle_block = nn.Sequential(
            nn.Conv2d(current_ch, middle_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.inject_middle = ZeroConv2d(middle_ch)

    def forward(
        self,
        offsets_t,
        t,
        high_res_image,
        target_density_map,
    ):
        _ = t
        hint_parts = [
            offsets_t,
            target_density_map,
        ]
        if self.enable_gecco:
            gecco_dynamic = self.compute_gecco_features(offsets_t, high_res_image)
            hint_parts.append(gecco_dynamic)

        hint_input = torch.cat(hint_parts, dim=1)
        hint = self.input_hint_block(hint_input)

        x = self.adapter_conv1(offsets_t)
        x = x + hint

        encoder_controls = []
        for level_blocks, down_layer, level_inj in zip(self.adapter_levels, self.adapter_downs, self.injections):
            current_enc = []
            for block, inj in zip(level_blocks, level_inj):
                x = block(x)
                current_enc.append(inj(x))
            encoder_controls.append(current_enc[::-1])
            x = down_layer(x)

        x = self.middle_block(x)
        middle_control = self.inject_middle(x)
        return (encoder_controls, middle_control)

    def compute_gecco_features(self, offsets_t, high_res_image):
        """Sample high-res image features at current noisy point positions."""
        if high_res_image is None:
            raise ValueError("high_res_image is required when enable_gecco=True")

        # Recompute grid_centers for the actual spatial size (supports grid sizes
        # different from the one used at training time).
        _, _, H, W = offsets_t.shape
        cx = torch.arange(W, dtype=torch.float32, device=offsets_t.device) / W + 0.5 / W
        cy = torch.arange(H, dtype=torch.float32, device=offsets_t.device) / H + 0.5 / H
        gx, gy = torch.meshgrid(cx, cy, indexing="xy")
        grid_centers = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, H, W)

        positions = grid_centers + offsets_t / H  # H == W for square grids
        sample_coords = positions.permute(0, 2, 3, 1) * 2.0 - 1.0

        gecco_feats_hr = self.gecco_extractor(high_res_image)
        return F.grid_sample(
            gecco_feats_hr,
            sample_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


class LightweightControlledDenoiser(nn.Module):
    """Wraps denoiser + adapter for inference."""

    def __init__(self, denoiser, lightweight_adapter):
        super().__init__()
        self.locked = denoiser
        self.control = lightweight_adapter
        self._high_res_image = None
        self._target_density = None

    def set_condition(self, high_res_image, target_density_map):
        self._high_res_image = high_res_image
        self._target_density = target_density_map

    def forward(self, x, t, cond=None):
        assert self._high_res_image is not None, (
            "Call set_condition(high_res_image, target_density_map) first"
        )

        hrs = self._high_res_image
        tgt = self._target_density

        if hrs.shape[0] != x.shape[0]:
            hrs = hrs.expand(x.shape[0], -1, -1, -1)
        if tgt.shape[0] != x.shape[0]:
            tgt = tgt.expand(x.shape[0], -1, -1, -1)

        controls = self.control(x, t, hrs, tgt)
        return self.locked(x, t, cond=cond, controls=controls)
