import copy
import torch
import torch.nn as nn
from models.Layers import get_timestep_embedding


class ZeroConv2d(nn.Module):
    """1x1 convolution initialized to zero, so the control branch
    starts as a no-op and gradually learns to inject signal."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNet(nn.Module):
    """Trainable control branch that produces skip-connection and
    middle-block control signals for a frozen DenoiserModel.

    Architecture: deepcopy of the pretrained encoder + middle, preceded by
    a small hint encoder that maps the grayscale condition image into the
    UNet input space.  Each encoder-block output and the middle output go
    through a zero-initialized 1x1 convolution before being handed to the
    locked decoder.
    """

    def __init__(self, denoiser, condition_channels=1):
        super().__init__()

        self.ch = denoiser.ch

        # --- Trainable copies of the encoder infrastructure ---
        self.ctrl_dense1 = copy.deepcopy(denoiser.dense1)
        self.ctrl_dense2 = copy.deepcopy(denoiser.dense2)
        self.ctrl_conv1 = copy.deepcopy(denoiser.conv1)
        self.ctrl_encoder_layers = copy.deepcopy(denoiser.encoder_layers)
        self.ctrl_downsamp_layers = copy.deepcopy(denoiser.downsamp_layers)
        self.ctrl_middle = copy.deepcopy(denoiser.middle)

        # --- Hint encoder: grayscale (1ch) -> 2ch (matches UNet input) ---
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(condition_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2, 1),
        )

        # --- Zero convolutions (one per encoder ResBlock + one for middle) ---
        self.zero_convs = nn.ModuleList()
        for level_layers in self.ctrl_encoder_layers:
            level_zeros = nn.ModuleList()
            for block in level_layers:
                resblock = list(block.children())[0]
                out_ch = resblock.conv2.net[1].weight.shape[0]
                level_zeros.append(ZeroConv2d(out_ch))
            self.zero_convs.append(level_zeros)

        middle_first_resblock = list(self.ctrl_middle.children())[0]
        middle_ch = middle_first_resblock.conv2.net[1].weight.shape[0]
        self.zero_conv_middle = ZeroConv2d(middle_ch)

    def forward(self, x_noisy, t, condition_img):
        """Run the control encoder and return control signals.

        Returns:
            tuple: (encoder_controls, middle_control) where
                encoder_controls is a list-of-lists matching the
                ``encoders`` structure in DenoiserModel.forward
                (each inner list is reversed per-level).
        """
        hint = self.input_hint_block(condition_img)
        x = x_noisy + hint

        temb = get_timestep_embedding(t, self.ch * 4)
        temb = self.ctrl_dense1(temb)
        temb = self.ctrl_dense2(temb)

        x = self.ctrl_conv1(x)

        encoder_controls = []
        for enc_layer, dns, level_zeros in zip(
            self.ctrl_encoder_layers,
            self.ctrl_downsamp_layers,
            self.zero_convs,
        ):
            current_enc = []
            for layer, zc in zip(enc_layer, level_zeros):
                x = layer(x, temb, None)
                current_enc.append(zc(x))
            encoder_controls.append(current_enc[::-1])
            x = dns(x, temb, None)

        x = self.ctrl_middle(x, temb, None)
        middle_control = self.zero_conv_middle(x)

        return (encoder_controls, middle_control)


class ControlledDenoiser(nn.Module):
    """Thin wrapper that plugs a ControlNet into a frozen DenoiserModel.

    Its ``forward(x, t, cond)`` signature is identical to
    ``DenoiserModel.forward``, so it can be used as a drop-in replacement
    inside ``DiffusionModel`` (no changes to the sampling loop needed).
    """

    def __init__(self, locked_denoiser, control_net):
        super().__init__()
        self.locked = locked_denoiser
        self.control_net = control_net
        self._condition_img = None

    def set_condition(self, condition_img):
        """Set the grayscale condition image for subsequent forward passes.

        Args:
            condition_img: tensor of shape (1, 1, H, W) or (B, 1, H, W).
                           Automatically expanded to match the batch dim.
        """
        self._condition_img = condition_img

    def forward(self, x, t, cond=None):
        assert self._condition_img is not None, (
            "Call set_condition(img) before forward / sampling"
        )
        cond_img = self._condition_img
        if cond_img.shape[0] != x.shape[0]:
            cond_img = cond_img.expand(x.shape[0], -1, -1, -1)

        controls = self.control_net(x, t, cond_img)
        return self.locked(x, t, cond=cond, controls=controls)
