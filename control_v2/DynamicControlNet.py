"""Dynamic ControlNet with adaptive gated injection for stipple generation.

Key differences from V1 ControlNet:
  - Dynamic density feedback: at each denoising step, the current noisy offsets
    are converted to point positions and the high-res image is sampled at those
    locations via F.grid_sample, yielding a per-cell density signal.
  - 4-channel hint input: [offsets_t(2), target_density(1), dynamic_density(1)]
  - AdaptiveGateInjection replaces ZeroConv2d for sigmoid-gated skip injection.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Layers import get_timestep_embedding


class AdaptiveGateInjection(nn.Module):
    """Sigmoid-gated 1x1 convolution for injecting control signals.

    Instead of the additive ``skip + zero_conv(ctrl)`` used by standard
    ControlNet, this module computes ``sigmoid(gate(ctrl)) * transform(ctrl)``
    so the frozen U-Net can do ``skip + gated_output``.

    At init the gate bias is -4.0 (sigmoid ~ 0.018), giving the same
    near-zero startup as a zero convolution while allowing the gate to
    adaptively scale injection per spatial location during training.
    """

    def __init__(self, channels):
        super().__init__()
        self.transform = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, ctrl_feature):
        """Return gated control signal ready for additive injection."""
        g = torch.sigmoid(self.gate(ctrl_feature))
        return g * self.transform(ctrl_feature)


class DynamicControlNet(nn.Module):
    """Trainable control branch with dynamic per-step density feedback.

    At each denoising step the module:
      1. Converts noisy offsets to point positions via the fixed grid.
      2. Samples the high-res grayscale image at those positions
         (``F.grid_sample``), producing a dynamic local density map.
      3. Concatenates ``[offsets_t, target_density, dynamic_density]``
         (4 channels) and passes through a hint encoder -> control encoder.
      4. Returns ``(encoder_controls, middle_control)`` tensors ready for
         additive injection into the frozen U-Net (via DenoiserModel's
         existing ``controls`` parameter).

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

        # ── trainable copies of encoder infrastructure ───────────────
        self.ctrl_dense1 = copy.deepcopy(denoiser.dense1)
        self.ctrl_dense2 = copy.deepcopy(denoiser.dense2)
        self.ctrl_conv1 = copy.deepcopy(denoiser.conv1)
        self.ctrl_encoder_layers = copy.deepcopy(denoiser.encoder_layers)
        self.ctrl_downsamp_layers = copy.deepcopy(denoiser.downsamp_layers)
        self.ctrl_middle = copy.deepcopy(denoiser.middle)

        # ── hint encoder: 4ch -> 2ch (matches UNet input space) ─────
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2, 1),
        )

        # ── adaptive gate injections ────────────────────────────────
        self.gate_injections = nn.ModuleList()
        for level_layers in self.ctrl_encoder_layers:
            level_gates = nn.ModuleList()
            for block in level_layers:
                resblock = list(block.children())[0]
                out_ch = resblock.conv2.net[1].weight.shape[0]
                level_gates.append(AdaptiveGateInjection(out_ch))
            self.gate_injections.append(level_gates)

        middle_first_resblock = list(self.ctrl_middle.children())[0]
        middle_ch = middle_first_resblock.conv2.net[1].weight.shape[0]
        self.gate_middle = AdaptiveGateInjection(middle_ch)

    def compute_dynamic_density(self, offsets_t, high_res_image):
        """Sample the high-res image at current estimated point positions.

        Parameters
        ----------
        offsets_t : Tensor (B, 2, H, W)
            Noisy offsets in **cell-relative units** (range approx [-1, 1]).
        high_res_image : Tensor (B, 1, Himg, Wimg)
            Full-resolution grayscale source image in [0, 1].

        Returns
        -------
        Tensor (B, 1, H, W)
            Sampled image intensity at each grid cell's current position.
        """
        positions = self.grid_centers + offsets_t / self.grid_size

        # F.grid_sample expects (B, H, W, 2) coordinates in [-1, 1]
        sample_coords = positions.permute(0, 2, 3, 1) * 2.0 - 1.0

        return F.grid_sample(
            high_res_image, sample_coords,
            mode="bilinear", padding_mode="border", align_corners=False,
        )

    def forward(self, offsets_t, t, high_res_image, target_density_map):
        """Run control encoder and return injection-ready control signals.

        Parameters
        ----------
        offsets_t : Tensor (B, 2, 32, 32)
        t : Tensor (B,)
        high_res_image : Tensor (B, 1, Himg, Wimg)
        target_density_map : Tensor (B, 1, 32, 32)

        Returns
        -------
        tuple (encoder_controls, middle_control)
            ``encoder_controls`` is list-of-lists matching the encoder
            structure (inner lists reversed per level).
            ``middle_control`` is a single tensor.
            Both are gated and ready for additive injection.
        """
        dynamic_density = self.compute_dynamic_density(offsets_t, high_res_image)

        hint_input = torch.cat([offsets_t, target_density_map, dynamic_density], dim=1)
        hint = self.input_hint_block(hint_input)

        x = offsets_t + hint

        temb = get_timestep_embedding(t, self.ch * 4)
        temb = self.ctrl_dense1(temb)
        temb = self.ctrl_dense2(temb)

        x = self.ctrl_conv1(x)

        encoder_controls = []
        for enc_layer, dns, level_gates in zip(
            self.ctrl_encoder_layers,
            self.ctrl_downsamp_layers,
            self.gate_injections,
        ):
            current_enc = []
            for layer, gate in zip(enc_layer, level_gates):
                x = layer(x, temb, None)
                current_enc.append(gate(x))
            encoder_controls.append(current_enc[::-1])
            x = dns(x, temb, None)

        x = self.ctrl_middle(x, temb, None)
        middle_control = self.gate_middle(x)

        return (encoder_controls, middle_control)


class DynamicControlledDenoiser(nn.Module):
    """Drop-in wrapper that plugs a DynamicControlNet into a frozen denoiser.

    The ``forward(x, t, cond)`` signature matches ``DenoiserModel.forward``,
    so it can replace ``diffusion.model`` inside the existing sampling loop
    without any changes to ``DiffusionModel``.

    Before calling ``forward`` (or running sampling), call
    ``set_condition(high_res_image, target_density_map)`` once.
    """

    def __init__(self, denoiser, dynamic_control_net):
        super().__init__()
        self.locked = denoiser
        self.control = dynamic_control_net
        self._high_res_image = None
        self._target_density = None

    def set_condition(self, high_res_image, target_density_map):
        """Cache the conditioning tensors for all subsequent forward calls.

        Parameters
        ----------
        high_res_image : Tensor (1, 1, H, W) or (B, 1, H, W)
        target_density_map : Tensor (1, 1, 32, 32) or (B, 1, 32, 32)
        """
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
