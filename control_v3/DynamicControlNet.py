"""ControlNet V3.2 "Shock the System" -- static conditioning + standard init.

V3.1 removed chaotic dynamic sampling but kept zero-initialized injection
layers (AdaptiveGateInjection).  The zero output at step 1 let the frozen
U-Net learn a uniform-grid shortcut before the ControlNet could influence
it, trapping training in a local minimum around MSE ~0.18.

V3.2 replaces AdaptiveGateInjection with StandardInjection: a plain 1x1
conv using PyTorch's default Kaiming initialization.  The non-zero control
signal from step 1 forces the U-Net to immediately account for the image
condition, preventing the uniform-grid trap.

Hint input channels: offsets_t(2) + target_density(1) + sdf(1) + coord_grid(2) = 6
"""

import copy

import torch
import torch.nn as nn

from models.Layers import get_timestep_embedding


class StandardInjection(nn.Module):
    """1x1 convolution with standard (Kaiming) initialization.

    Unlike the zero-initialized AdaptiveGateInjection, this produces a
    non-zero control signal from step 1, preventing the frozen U-Net from
    settling into the uniform-grid local minimum before the ControlNet
    has any influence.
    """

    def __init__(self, channels):
        super().__init__()
        self.inject = nn.Conv2d(channels, channels, 1)

    def forward(self, ctrl_feature):
        return self.inject(ctrl_feature)


class DynamicControlNet(nn.Module):
    """Trainable control branch with static conditioning (V3.2).

    At each denoising step the module:
      1. Concatenates [offsets_t(2), target_density(1), sdf(1), coord_grid(2)] = 6
         channels and passes through a 3-layer hint encoder to produce a
         128-channel tensor.
      2. Passes offsets_t through the copied conv1, adds the 128ch hint,
         then runs through the control encoder + middle blocks.
      3. Returns (encoder_controls, middle_control) tensors transformed by
         StandardInjection (Kaiming-initialized 1x1 conv), ready for
         additive injection into the frozen U-Net.

    The ``high_res_image`` argument is accepted in ``forward`` for
    call-signature compatibility but is not used internally.

    Parameters
    ----------
    denoiser : DenoiserModel
        Pretrained (frozen) denoiser whose encoder + middle blocks are
        deep-copied to form the trainable control encoder.
    grid_size : int
        Spatial resolution of the offset grid (default 32).
    """

    def __init__(self, denoiser, grid_size=32):
        super().__init__()
        self.ch = denoiser.ch
        self.grid_size = grid_size

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

        # ── trainable copies of encoder infrastructure ───────────────
        self.ctrl_dense1 = copy.deepcopy(denoiser.dense1)
        self.ctrl_dense2 = copy.deepcopy(denoiser.dense2)
        self.ctrl_conv1 = copy.deepcopy(denoiser.conv1)
        self.ctrl_encoder_layers = copy.deepcopy(denoiser.encoder_layers)
        self.ctrl_downsamp_layers = copy.deepcopy(denoiser.downsamp_layers)
        self.ctrl_middle = copy.deepcopy(denoiser.middle)

        # ── hint encoder: 6ch -> ch_mult[0] (128) ───────────────────
        #    offsets_t(2) + target_density(1) + sdf(1) + coord_grid(2) = 6
        hint_in_ch = 2 + 1 + 1 + 2  # = 6
        first_hidden = denoiser.conv1.net[1].weight.shape[0]  # ch_mult[0] = 128
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),   # ~13px receptive field
            nn.SiLU(),
            nn.Conv2d(64, first_hidden, 3, padding=4, dilation=4),  # ~29px
        )

        # ── injection layers (standard Kaiming init, NOT zero) ───────
        self.injections = nn.ModuleList()
        for level_layers in self.ctrl_encoder_layers:
            level_inj = nn.ModuleList()
            for block in level_layers:
                resblock = list(block.children())[0]
                out_ch = resblock.conv2.net[1].weight.shape[0]
                level_inj.append(StandardInjection(out_ch))
            self.injections.append(level_inj)

        middle_first_resblock = list(self.ctrl_middle.children())[0]
        middle_ch = middle_first_resblock.conv2.net[1].weight.shape[0]
        self.inject_middle = StandardInjection(middle_ch)

    def forward(self, offsets_t, t, high_res_image, target_density_map, sdf_map=None):
        """Run control encoder and return injection-ready control signals.

        Parameters
        ----------
        offsets_t : Tensor (B, 2, 32, 32)
        t : Tensor (B,)
        high_res_image : Tensor (B, 1, Himg, Wimg)
            Accepted for call-signature compatibility; unused internally.
        target_density_map : Tensor (B, 1, 32, 32)
        sdf_map : Tensor (B, 1, 32, 32) or None
            SDF of empty space (0=occupied, 1=far from dots). If None,
            a zero-filled channel is used.

        Returns
        -------
        tuple (encoder_controls, middle_control)
        """
        B = offsets_t.shape[0]

        batch_coords = self.coord_grid.expand(B, -1, -1, -1)

        if sdf_map is None:
            sdf_map = torch.zeros(B, 1, offsets_t.shape[2], offsets_t.shape[3],
                                  device=offsets_t.device, dtype=offsets_t.dtype)

        hint_input = torch.cat([
            offsets_t,           # 2ch
            target_density_map,  # 1ch
            sdf_map,             # 1ch
            batch_coords,        # 2ch
        ], dim=1)  # -> 6ch
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


class DynamicControlledDenoiser(nn.Module):
    """Drop-in wrapper that plugs a DynamicControlNet V3 into a frozen denoiser.

    The ``forward(x, t, cond)`` signature matches ``DenoiserModel.forward``,
    so it can replace ``diffusion.model`` inside the existing sampling loop.

    Before calling ``forward``, call
    ``set_condition(high_res_image, target_density_map)`` once.
    """

    def __init__(self, denoiser, dynamic_control_net):
        super().__init__()
        self.locked = denoiser
        self.control = dynamic_control_net
        self._high_res_image = None
        self._target_density = None
        self._sdf = None

    def set_condition(self, high_res_image, target_density_map, sdf_map=None):
        self._high_res_image = high_res_image
        self._target_density = target_density_map
        self._sdf = sdf_map

    def forward(self, x, t, cond=None):
        assert self._high_res_image is not None, (
            "Call set_condition(high_res_image, target_density_map) first"
        )

        hrs = self._high_res_image
        tgt = self._target_density
        sdf = self._sdf
        if hrs.shape[0] != x.shape[0]:
            hrs = hrs.expand(x.shape[0], -1, -1, -1)
        if tgt.shape[0] != x.shape[0]:
            tgt = tgt.expand(x.shape[0], -1, -1, -1)
        if sdf is not None and sdf.shape[0] != x.shape[0]:
            sdf = sdf.expand(x.shape[0], -1, -1, -1)

        controls = self.control(x, t, hrs, tgt, sdf_map=sdf)
        return self.locked(x, t, cond=cond, controls=controls)
