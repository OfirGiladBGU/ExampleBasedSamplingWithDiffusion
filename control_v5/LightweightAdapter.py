"""Image-GECCO early-fusion wrapper for control_v5.

Architecture
------------
Two lightweight conv layers extract a feature map from the conditioning image.
At each denoising step, features are sampled at the current noisy-offset
positions via ``F.grid_sample`` (GECCO: Geometry-Aware Feature Conditioning
with Offsets).  The sampled features are *concatenated* to the noisy offsets
and fed into the U-Net via its very first convolution (early fusion / channel
concatenation).  No parallel U-Net branch is required.

Grid centers are derived dynamically from ``offsets_t.shape[-1]``, so the
model generalises to any offset grid resolution at inference time
(e.g. 32×32 trained → 48×48 at test time).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class ImageGECCOWrapper(nn.Module):
    """Early-fusion image-conditioned denoiser wrapper.

    On construction this module widens the wrapped denoiser's first convolution
    in-place to accept ``(2 + gecco_ch)`` input channels instead of the
    original 2, copying the pretrained weights and zero-initialising the new
    channels for a stable warm-up.

    Parameters
    ----------
    denoiser : DenoiserModel
        Pretrained base U-Net denoiser.
    gecco_ch : int
        Number of GECCO feature channels extracted from the conditioning image.
        Default: 8.
    """

    def __init__(self, denoiser, gecco_ch: int = 8):
        super().__init__()
        self.denoiser = denoiser
        self.gecco_ch = gecco_ch
        self._cached_image = None

        # Lightweight feature extractor: (B, 1, H, W) → (B, gecco_ch, H, W)
        self.gecco_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, gecco_ch, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Widen denoiser.conv1 from (2 → ch) to (2 + gecco_ch → ch).
        # denoiser.conv1 is Conv2dSame; the actual nn.Conv2d lives at .net[1].
        old_conv = denoiser.conv1.net[1]
        out_ch = old_conv.out_channels
        ks = old_conv.kernel_size
        has_bias = old_conv.bias is not None

        new_conv = nn.Conv2d(2 + gecco_ch, out_ch, kernel_size=ks, bias=has_bias)
        with torch.no_grad():
            new_conv.weight[:, :2, :, :].copy_(old_conv.weight.data)
            new_conv.weight[:, 2:, :, :].zero_()
            if has_bias:
                new_conv.bias.data.copy_(old_conv.bias.data)

        denoiser.conv1.net[1] = new_conv

    # ------------------------------------------------------------------
    def set_condition(self, high_res_image: torch.Tensor) -> None:
        """Cache the conditioning image for upcoming forward calls.

        Parameters
        ----------
        high_res_image : Tensor, shape ``(B, 1, H, W)`` or ``(1, 1, H, W)``
        """
        self._cached_image = high_res_image

    # ------------------------------------------------------------------
    def compute_gecco(
        self, offsets_t: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        """Sample GECCO features at positions defined by ``offsets_t``.

        Grid centers are derived dynamically from ``offsets_t.shape[-1]`` so
        the wrapper is resolution-agnostic: train on 32×32, run on 48×48.

        Parameters
        ----------
        offsets_t : Tensor, shape ``(B, 2, G, G)``
        image : Tensor, shape ``(B, 1, H, W)``

        Returns
        -------
        Tensor, shape ``(B, gecco_ch, G, G)``
        """
        B, _, G, _ = offsets_t.shape
        device, dtype = offsets_t.device, offsets_t.dtype

        xs = torch.linspace(0.5 / G, 1.0 - 0.5 / G, G, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(xs, xs, indexing="ij")
        grid_centers = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, G, G)

        positions = grid_centers + offsets_t / G            # (B, 2, G, G) in [0, 1]
        sample_coords = positions.permute(0, 2, 3, 1) * 2.0 - 1.0  # (B, G, G, 2) in [-1, 1]

        gecco_map = self.gecco_extractor(image)             # (B, gecco_ch, H, W)
        return F.grid_sample(
            gecco_map,
            sample_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (B, gecco_ch, G, G)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond=None
    ) -> torch.Tensor:
        """Denoise ``x`` conditioned on the cached image.

        Parameters
        ----------
        x : Tensor, shape ``(B, 2, G, G)`` — noisy offset grid
        t : Tensor, shape ``(B,)`` — diffusion timestep
        cond : ignored (kept for API compatibility with DiffusionModel)
        """
        assert self._cached_image is not None, (
            "Call set_condition(high_res_image) before forward()"
        )
        img = self._cached_image
        if img.shape[0] != x.shape[0]:
            img = img.expand(x.shape[0], -1, -1, -1)

        gecco = self.compute_gecco(x, img)
        x_aug = torch.cat([x, gecco], dim=1)   # (B, 2 + gecco_ch, G, G)
        return self.denoiser(x_aug, t, cond=cond)
