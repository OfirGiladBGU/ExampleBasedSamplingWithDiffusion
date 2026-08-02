"""Descriptor-field-conditioned ControlNet for multi-oracle style control (Milestone M2).

Conditions `control_v4`'s DynamicControlNet on a `(K, G, G)` field of measured point-distribution
descriptors -- `descriptor_fields.CONDITIONING_KEYS`, normalised to [0,1] -- so the requested style
can vary ACROSS THE IMAGE, not just per image.

Why spatial FiLM and not an extra hint channel
-----------------------------------------------
The scalar-axis predecessor (`control_v4_mix/DynamicControlNetStyle.py`) mapped one number through
an MLP and added it to `temb`. That cannot work here: `temb` is a per-sample vector, so anything
injected there is spatially constant, and "GBN-like here, Floyd-Steinberg-like there" becomes
unreachable -- which is the entire contribution.

The obvious spatial alternative is to concatenate the descriptor onto `hint_parts` beside
`target_density_map`. That is cheap and position-free, but it makes the descriptor 5 input channels
competing against rho, and rho is enormously predictive of the target. The plan's three named
failure modes -- snapping, muddy averaging, and metric-response-without-visual-change -- are all the
model UNDER-USING the conditioning. Additive conditioning at a single early site is exactly what a
network learns to ignore when a stronger correlate is available.

So the descriptor modulates features MULTIPLICATIVELY at every control resblock:

    x <- x * (1 + scale(D)) + shift(D)

Multiplicative gating cannot be bypassed by ignoring an input channel -- it scales whatever the
block computed. FiLM is also the standard mechanism for steering generation by a continuous
parameter, which is precisely the request here.

Three properties keep this inside the plan's constraints:

  * POSITION-FREE. The descriptor encoder is fully convolutional and the per-level resampling is
    `adaptive_avg_pool2d` to whatever spatial size the trunk activation has. No learned absolute
    positional embedding is introduced, so the G-transfer property (train at 32, sample at 48/64)
    survives -- the same pooling adapts to any G.
  * RHO IS UNTOUCHED. Density stays a hard condition on its own path; style is a residual control.
    An edit to the input density map would be separable and would destroy the contribution.
  * IDENTITY AT INIT. Every FiLM head is zero-initialised, so `scale = 0`, `shift = 0`, and the
    network is bit-identical to the unconditioned baseline at step 0. It then LEARNS to use the
    descriptor rather than being perturbed into it -- the same safety property the zero-init MLP
    gave the scalar version.

`film_activity()` reports how far the learned scales actually depart from 1, so "is the model using
the descriptor at all?" is a measurement rather than an assumption -- the analogue of
`gate_activity_diagnostic.py` for the existing sigmoid gate.

`--descriptor-inject hint` is kept as an ablation: it concatenates the field onto the hint stack
instead, giving a clean A/B against the FiLM path without touching anything else.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Layers import get_timestep_embedding
from control_v4.DynamicControlNet import (
    DynamicControlNet,
    DynamicControlledDenoiser,
)


class _GatedFiLMHead(nn.Module):
    """conv(f) scaled by a learnable scalar initialised to zero -- see _film_head's note."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2 * out_ch, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, f):
        return self.gate * self.conv(f)


def _resblock_out_channels(block):
    """Output channel count of a control encoder block -- introspected exactly as the injection
    layers do in DynamicControlNet.__init__, so the two always agree."""
    resblock = list(block.children())[0]
    return resblock.conv2.net[1].weight.shape[0]


class DynamicControlNetDescriptor(DynamicControlNet):
    def __init__(self, *args, descriptor_channels=5, descriptor_hidden=64,
                 descriptor_inject="film", **kwargs):
        if descriptor_inject not in ("film", "hint", "both"):
            raise ValueError(f"descriptor_inject must be film|hint|both, got {descriptor_inject}")
        self._descriptor_hint = descriptor_inject in ("hint", "both")
        self._descriptor_channels = int(descriptor_channels)
        super().__init__(*args, **kwargs)
        self.descriptor_inject = descriptor_inject

        # Dilated trunk, mirroring input_hint_block's receptive-field growth: a descriptor cell is
        # a statistic over a ~5-cell window, so the encoder needs to see a comparable neighbourhood
        # for its features to mean anything.
        self.desc_encoder = nn.Sequential(
            nn.Conv2d(self._descriptor_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, descriptor_hidden, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(descriptor_hidden, descriptor_hidden, 3, padding=1),
            nn.SiLU(),
        )

        if descriptor_inject in ("film", "both"):
            self.film = nn.ModuleList()
            for level_layers in self.ctrl_encoder_layers:
                level = nn.ModuleList()
                for block in level_layers:
                    level.append(self._film_head(descriptor_hidden,
                                                 _resblock_out_channels(block)))
                self.film.append(level)
            self.film_middle = self._film_head(
                descriptor_hidden, _resblock_out_channels(self.ctrl_middle))

        # Widen the hint encoder only for the ablation path.
        if self._descriptor_hint:
            old = self.input_hint_block[0]
            new = nn.Conv2d(old.in_channels + self._descriptor_channels, old.out_channels,
                            kernel_size=old.kernel_size, padding=old.padding)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, : old.in_channels] = old.weight
                new.bias.copy_(old.bias)          # new channels start at zero -> identity at init
            self.input_hint_block[0] = new

    @staticmethod
    def _film_head(in_ch, out_ch):
        """1x1 conv -> (scale, shift), gated by a zero-init scalar.

        NOT a zero-init conv, and the difference is not cosmetic. ControlNet's injection layers are
        already zero-initialised by design, so at step 0 `dL/dx = 0` through them and NOTHING
        upstream receives gradient -- the baseline's own control encoder included. Zero-initialising
        the FiLM conv on top of that stacks a second dead stage, and `desc_encoder` a third: the
        descriptor branch could only start learning after injections woke the heads, which then woke
        the encoder. Measured: 0/20 descriptor tensors received gradient at init.

        A zero-init scalar GATE over a normally-initialised conv gives the same identity at step 0
        (`gate = 0` => scale = shift = 0) while keeping the gradient path alive, because
        `dL/dgate = dL/dout . conv(f)` is non-zero as soon as anything downstream is. The gate wakes
        in one step and the conv and encoder are immediately live behind it. This is the
        zero-init-residual / LayerScale trick, and it strictly dominates a zero-init conv here.
        """
        return _GatedFiLMHead(in_ch, out_ch)

    @staticmethod
    def _apply_film(x, head, feat):
        """x * (1 + scale) + shift, with the descriptor features resampled to x's resolution.

        `adaptive_avg_pool2d` rather than a fixed stride: the control branch halves resolution per
        level, and at inference G may be 48 or 64 instead of 32. Pooling to whatever `x` actually is
        keeps this correct for every level and every grid size without hard-coding either.
        """
        f = feat if feat.shape[-2:] == x.shape[-2:] else \
            F.adaptive_avg_pool2d(feat, x.shape[-2:])
        scale, shift = head(f).chunk(2, dim=1)
        return x * (1.0 + scale) + shift

    def forward(
        self,
        offsets_t,
        t,
        high_res_image,
        target_density_map,
        high_res_sdf=None,
        target_sdf_map=None,
        target_smart_init_map=None,
        descriptor_field=None,
        timing_breakdown=None,
    ):
        B, _, H, W = offsets_t.shape
        cuda_timing_enabled = (
            timing_breakdown is not None
            and torch.cuda.is_available()
            and offsets_t.is_cuda
        )

        # ── descriptor features (shared by every FiLM head) ────────────────
        desc_feat = None
        if descriptor_field is not None:
            d = descriptor_field
            if d.shape[-2:] != (H, W):
                d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            desc_feat = self._time_block(
                timing_breakdown, "ctrl.desc_encode",
                lambda: self.desc_encoder(d), cuda_timing_enabled)

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
                timing_breakdown, "ctrl.grid_gen",
                lambda: self._build_coord_grid(H, W, offsets_t.device), cuda_timing_enabled)
            hint_parts.append(coord_grid.expand(B, -1, -1, -1))
        if self.enable_gecco:
            hint_parts.append(self.compute_gecco_features(
                offsets_t, high_res_image, high_res_sdf, timing_breakdown=timing_breakdown))
        if self._descriptor_hint:
            if descriptor_field is None:
                hint_parts.append(offsets_t.new_zeros(B, self._descriptor_channels, H, W))
            else:
                hint_parts.append(d)

        hint_input = torch.cat(hint_parts, dim=1)
        hint = self._time_block(
            timing_breakdown, "ctrl.hint_encode",
            lambda: self.input_hint_block(hint_input), cuda_timing_enabled)

        x = self._time_block(
            timing_breakdown, "ctrl.conv1",
            lambda: self.ctrl_conv1(offsets_t), cuda_timing_enabled)
        x = x + hint

        def build_temb():
            t_emb = get_timestep_embedding(t, self.ch * 4)
            t_emb = self.ctrl_dense1(t_emb)
            return self.ctrl_dense2(t_emb)

        temb = self._time_block(timing_breakdown, "ctrl.temb", build_temb, cuda_timing_enabled)

        use_film = desc_feat is not None and self.descriptor_inject in ("film", "both")

        encoder_controls = []
        for lvl, (enc_layer, dns, level_inj) in enumerate(zip(
                self.ctrl_encoder_layers, self.ctrl_downsamp_layers, self.injections)):
            current_enc = []
            for bi, (layer, inj) in enumerate(zip(enc_layer, level_inj)):
                x = self._time_block(
                    timing_breakdown, "ctrl.down_blocks",
                    lambda layer=layer, x=x: layer(x, temb, None), cuda_timing_enabled)
                if use_film:
                    # Modulate BEFORE the injection, so the control signal handed to the frozen
                    # base already carries the descriptor's influence.
                    x = self._time_block(
                        timing_breakdown, "ctrl.film",
                        lambda x=x, h=self.film[lvl][bi]: self._apply_film(x, h, desc_feat),
                        cuda_timing_enabled)
                current_enc.append(self._time_block(
                    timing_breakdown, "ctrl.injections",
                    lambda inj=inj, x=x: inj(x), cuda_timing_enabled))
            encoder_controls.append(current_enc[::-1])

            x = self._time_block(
                timing_breakdown, "ctrl.down_blocks",
                lambda dns=dns, x=x: dns(x, temb, None), cuda_timing_enabled)

        x = self._time_block(
            timing_breakdown, "ctrl.mid_block",
            lambda: self.ctrl_middle(x, temb, None), cuda_timing_enabled)
        if use_film:
            x = self._apply_film(x, self.film_middle, desc_feat)

        middle_control = self._time_block(
            timing_breakdown, "ctrl.injections",
            lambda: self.inject_middle(x), cuda_timing_enabled)

        return (encoder_controls, middle_control)

    # ── diagnostics ──────────────────────────────────────────────────────

    @torch.no_grad()
    def film_activity(self, descriptor_field, grid_size=None):
        """How far the FiLM modulation departs from identity, per level.

        At initialisation every value is 0 by construction. If these stay ~0 after training the
        model is IGNORING the descriptor, which is the plan's "metric response without visual
        change" failure arriving early and cheaply -- catch it here rather than in an eval panel.
        """
        if self.descriptor_inject not in ("film", "both"):
            return {}
        d = descriptor_field
        if grid_size is not None and d.shape[-1] != grid_size:
            d = F.interpolate(d, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        feat = self.desc_encoder(d)
        out = {}
        for lvl, level in enumerate(self.film):
            for bi, head in enumerate(level):
                scale, shift = head(feat).chunk(2, dim=1)
                out[f"L{lvl}.{bi}"] = {"abs_scale": float(scale.abs().mean()),
                                       "abs_shift": float(shift.abs().mean())}
        scale, shift = self.film_middle(feat).chunk(2, dim=1)
        out["middle"] = {"abs_scale": float(scale.abs().mean()),
                         "abs_shift": float(shift.abs().mean())}
        out["mean_abs_scale"] = float(
            sum(v["abs_scale"] for k, v in out.items() if isinstance(v, dict)) /
            max(sum(1 for v in out.values() if isinstance(v, dict)), 1))
        return out

    def descriptor_parameters(self):
        """Parameters introduced by descriptor conditioning (for separate LR / weight decay)."""
        mods = [self.desc_encoder]
        if self.descriptor_inject in ("film", "both"):
            mods += [self.film, self.film_middle]
        for m in mods:
            for p in m.parameters():
                yield p


class DynamicControlledDescriptorDenoiser(DynamicControlledDenoiser):
    """Inference wrapper threading the descriptor field into the control branch."""

    def set_condition(self, high_res_image, high_res_sdf, target_density_map, target_sdf_map,
                      smart_init_grid, descriptor_field=None):
        super().set_condition(high_res_image, high_res_sdf, target_density_map, target_sdf_map,
                              smart_init_grid)
        self._descriptor_field = descriptor_field

    def forward(self, x, t, cond=None, timing_breakdown=None):
        assert self._high_res_image is not None, "Call set_condition(..., descriptor_field=) first"
        hrs = self._high_res_image
        hrs_sdf = self._high_res_sdf
        tgt = self._target_density
        tgt_sdf = self._target_sdf
        smart = self._smart_init_grid
        desc = getattr(self, "_descriptor_field", None)

        def _expand(v):
            return v if v is None or v.shape[0] == x.shape[0] else v.expand(x.shape[0], -1, -1, -1)

        controls = self.control(
            x, t, _expand(hrs), _expand(tgt),
            high_res_sdf=_expand(hrs_sdf) if self.control.sdf_features else None,
            target_sdf_map=_expand(tgt_sdf) if self.control.sdf_features else None,
            target_smart_init_map=_expand(smart) if self.control.smart_init_features else None,
            descriptor_field=_expand(desc),
            timing_breakdown=timing_breakdown,
        )
        return self.locked(x, t, cond=cond, controls=controls, timing_breakdown=timing_breakdown)
