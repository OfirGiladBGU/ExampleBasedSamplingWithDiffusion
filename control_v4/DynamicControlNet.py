"""ControlNet V4 -- truncated-control conditioning with Smart Init support.

Active injection path uses AdaptiveGateInjection (revert of V3.2) so control
signals are sigmoid-gated 1x1 projections, initialized near zero at step 1.

Hint input channels: offsets_t(2) + target_density(1) + target_sdf(1) +
smart_init_grid(1) + coord_grid(2) = 7 (plus optional GECCO channels).
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
            [offsets_t(2), target_density(1), target_sdf(1), smart_init_grid(1), coord_grid(2)] and,
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

    def __init__(
        self,
        denoiser,
        grid_size=32,
        enable_gecco=False,
        gecco_channels=16,
        smart_init_features=True,
        sdf_features=True,
        batch_coords_features=True,
    ):
        super().__init__()
        self.ch = denoiser.ch
        self.grid_size = grid_size
        self.enable_gecco = enable_gecco
        self.gecco_channels = gecco_channels
        self.smart_init_features = bool(smart_init_features)
        self.sdf_features = bool(sdf_features)
        self.batch_coords_features = bool(batch_coords_features)

        # grid_centers and coord_grid are computed dynamically in forward() so that
        # the model can run inference on any grid size (e.g. train on 32x32, infer on
        # 48x48) without crashing.  They are NOT registered as buffers, which means
        # old checkpoints that contain those keys are simply ignored on load
        # (load_state_dict is called with strict=False everywhere).

        # ── optional GECCO feature extractor on high-res condition image ──
        if self.enable_gecco:
            gecco_in_ch = 1 + (1 if self.sdf_features else 0)
            self.gecco_extractor = nn.Sequential(
                nn.Conv2d(gecco_in_ch, 8, 3, padding=1),
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

        # ── hint encoder: static offsets + enabled conditioning channels (+ optional GECCO) ──
        hint_in_ch = 2 + 1
        if self.smart_init_features:
            hint_in_ch += 1
        if self.sdf_features:
            hint_in_ch += 1
        if self.batch_coords_features:
            hint_in_ch += 2
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

    def forward(
        self,
        offsets_t,
        t,
        high_res_image,
        target_density_map,
        high_res_sdf=None,
        target_sdf_map=None,
        target_smart_init_map=None,
    ):
        """Run control encoder and return injection-ready control signals.

        Parameters
        ----------
        offsets_t : Tensor (B, 2, G, G)
            Noisy offset grid; G can differ from the training grid_size.
        t : Tensor (B,)
        high_res_image : Tensor (B, 1, Himg, Wimg)
        target_density_map : Tensor (B, 1, G, G)
        high_res_sdf : Tensor (B, 1, Himg, Wimg) or None
        target_sdf_map : Tensor (B, 1, G, G) or None
        target_smart_init_map : Tensor (B, 1, G, G) or None

        Returns
        -------
        tuple (encoder_controls, middle_control)
        """
        B, _, H, W = offsets_t.shape

        # Keep the original legacy ordering so checkpoints trained before the
        # feature-flag split still receive the same channel semantics.
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
            # Recompute coord_grid for the actual (possibly different) spatial size.
            lin_y = torch.linspace(-1, 1, H, device=offsets_t.device)
            lin_x = torch.linspace(-1, 1, W, device=offsets_t.device)
            gy2d, gx2d = torch.meshgrid(lin_y, lin_x, indexing="ij")
            coord_grid = torch.stack([gx2d, gy2d], dim=0).unsqueeze(0)  # (1, 2, H, W)
            hint_parts.append(coord_grid.expand(B, -1, -1, -1))

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
        if high_res_image is None:
            raise ValueError("high_res_image is required when enable_gecco=True")
        if self.sdf_features and high_res_sdf is None:
            raise ValueError("high_res_sdf is required when sdf_features=True and enable_gecco=True")

        # Recompute grid_centers for the actual spatial size (supports grid sizes
        # different from the one used at training time).
        _, _, H, W = offsets_t.shape
        cx = torch.arange(W, dtype=torch.float32, device=offsets_t.device) / W + 0.5 / W
        cy = torch.arange(H, dtype=torch.float32, device=offsets_t.device) / H + 0.5 / H
        gx, gy = torch.meshgrid(cx, cy, indexing="xy")
        grid_centers = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, H, W)

        positions = grid_centers + offsets_t / H  # H == W for square grids
        sample_coords = positions.permute(0, 2, 3, 1) * 2.0 - 1.0

        gecco_input = high_res_image if not self.sdf_features else torch.cat([high_res_image, high_res_sdf], dim=1)
        gecco_feats_hr = self.gecco_extractor(gecco_input)
        return F.grid_sample(
            gecco_feats_hr,
            sample_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


class DynamicControlledDenoiser(nn.Module):
    """Drop-in wrapper that plugs a DynamicControlNet V4 into a frozen denoiser.

    The ``forward(x, t, cond)`` signature matches ``DenoiserModel.forward``,
    so it can replace ``diffusion.model`` inside the existing sampling loop.

    Before calling ``forward``, call
    ``set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map, smart_init_grid)`` once.
    """

    def __init__(self, denoiser, dynamic_control_net):
        super().__init__()
        self.locked = denoiser
        self.control = dynamic_control_net
        self._high_res_image = None
        self._high_res_sdf = None
        self._target_density = None
        self._target_sdf = None
        self._smart_init_grid = None

    def set_condition(self, high_res_image, high_res_sdf, target_density_map, target_sdf_map, smart_init_grid):
        self._high_res_image = high_res_image
        self._high_res_sdf = high_res_sdf
        self._target_density = target_density_map
        self._target_sdf = target_sdf_map
        self._smart_init_grid = smart_init_grid

    def forward(self, x, t, cond=None):
        assert self._high_res_image is not None, (
            "Call set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map, smart_init_grid) first"
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
            x,
            t,
            hrs,
            tgt,
            high_res_sdf=hrs_sdf if self.control.sdf_features else None,
            target_sdf_map=tgt_sdf if self.control.sdf_features else None,
            target_smart_init_map=smart if self.control.smart_init_features else None,
        )
        return self.locked(x, t, cond=cond, controls=controls)
