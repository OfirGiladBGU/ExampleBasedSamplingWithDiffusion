"""ControlNet V4 -- truncated-control conditioning with Smart Init support.

Active injection path uses AdaptiveGateInjection (revert of V3.2) so control
signals are sigmoid-gated 1x1 projections, initialized near zero at step 1.

Hint input channels: offsets_t(2) + target_density(1) + target_sdf(1) +
smart_init_grid(1) + coord_grid(2) = 7 (plus optional GECCO channels).
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Layers import get_timestep_embedding

### INJECTION MODULES ###


class ZeroConvInjection(nn.Module):
    """Simple zero-initialized 1x1 convolution injection (standard ControlNet behavior).
    
    Directly adds the control signal without gating: ``transform(ctrl)``.
    """

    def __init__(self, channels):
        super().__init__()
        self.transform = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

    def forward(self, ctrl_feature):
        return self.transform(ctrl_feature)


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



### DYNAMIC CONTROL NET ###


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
            either AdaptiveGateInjection (sigmoid-gated) or ZeroConvInjection (simple zero conv),
         ready for additive injection into the frozen U-Net.

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
    enable_adaptive_gate_injection : bool
        If True, use AdaptiveGateInjection (sigmoid-gated 1x1 convolutions).
        If False, use simple ZeroConvInjection (standard ControlNet zero convolutions).
    """

    def __init__(
        self,
        denoiser,
        grid_size=32,
        enable_gecco=False,
        gecco_channels=16,
        enable_adaptive_gate_injection=True,
        smart_init_features=False,
        sdf_features=False,
        batch_coords_features=False,
    ):
        super().__init__()
        self.ch = denoiser.ch
        self.grid_size = grid_size
        self.enable_gecco = enable_gecco
        self.gecco_channels = gecco_channels
        self.enable_adaptive_gate_injection = bool(enable_adaptive_gate_injection)
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

        # ── injection layers (adaptive gated projection or simple zero conv) ──────────────
        InjectionClass = AdaptiveGateInjection if self.enable_adaptive_gate_injection else ZeroConvInjection
        self.injections = nn.ModuleList()
        for level_layers in self.ctrl_encoder_layers:
            level_inj = nn.ModuleList()
            for block in level_layers:
                resblock = list(block.children())[0]
                out_ch = resblock.conv2.net[1].weight.shape[0]
                level_inj.append(InjectionClass(out_ch))
            self.injections.append(level_inj)

        middle_first_resblock = list(self.ctrl_middle.children())[0]
        middle_ch = middle_first_resblock.conv2.net[1].weight.shape[0]
        self.inject_middle = InjectionClass(middle_ch)

    def _time_block(self, timing_breakdown, key, fn, enabled):
        if timing_breakdown is None or not enabled:
            return fn()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        timing_breakdown.setdefault(key, []).append((start_event, end_event))
        return result

    def forward(
        self,
        offsets_t,
        t,
        high_res_image,
        target_density_map,
        high_res_sdf=None,
        target_sdf_map=None,
        target_smart_init_map=None,
        timing_breakdown=None,
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
        cuda_timing_enabled = (
            timing_breakdown is not None
            and torch.cuda.is_available()
            and offsets_t.is_cuda
        )

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
            coord_grid = self._time_block(
                timing_breakdown,
                "ctrl.grid_gen",
                lambda: self._build_coord_grid(H, W, offsets_t.device),
                cuda_timing_enabled,
            )
            hint_parts.append(coord_grid.expand(B, -1, -1, -1))

        if self.enable_gecco:
            gecco_dynamic = self.compute_gecco_features(offsets_t, high_res_image, high_res_sdf, timing_breakdown=timing_breakdown)
            hint_parts.append(gecco_dynamic)

        hint_input = torch.cat(hint_parts, dim=1)
        hint = self._time_block(
            timing_breakdown,
            "ctrl.hint_encode",
            lambda: self.input_hint_block(hint_input),
            cuda_timing_enabled,
        )

        # Time conv1
        x = self._time_block(
            timing_breakdown,
            "ctrl.conv1",
            lambda: self.ctrl_conv1(offsets_t),
            cuda_timing_enabled
        )
        x = x + hint

        # Time timestep embedding
        def build_temb():
            t_emb = get_timestep_embedding(t, self.ch * 4)
            t_emb = self.ctrl_dense1(t_emb)
            return self.ctrl_dense2(t_emb)
        
        temb = self._time_block(
            timing_breakdown,
            "ctrl.temb",
            build_temb,
            cuda_timing_enabled
        )

        encoder_controls = []
        for enc_layer, dns, level_inj in zip(
            self.ctrl_encoder_layers,
            self.ctrl_downsamp_layers,
            self.injections,
        ):
            current_enc = []
            for layer, inj in zip(enc_layer, level_inj):
                # Time the block math
                x = self._time_block(
                    timing_breakdown,
                    "ctrl.down_blocks",
                    lambda layer=layer, x=x: layer(x, temb, None),
                    cuda_timing_enabled
                )
                # Time the injection projection separately
                current_enc.append(
                    self._time_block(
                        timing_breakdown,
                        "ctrl.injections",
                        lambda inj=inj, x=x: inj(x),
                        cuda_timing_enabled,
                    )
                )
            encoder_controls.append(current_enc[::-1])
            
            # Time the downsample
            x = self._time_block(
                timing_breakdown,
                "ctrl.down_blocks",
                lambda dns=dns, x=x: dns(x, temb, None),
                cuda_timing_enabled
            )

        x = self._time_block(
            timing_breakdown,
            "ctrl.mid_block",
            lambda: self.ctrl_middle(x, temb, None),
            cuda_timing_enabled
        )
        
        middle_control = self._time_block(
            timing_breakdown,
            "ctrl.injections",
            lambda: self.inject_middle(x),
            cuda_timing_enabled,
        )

        return (encoder_controls, middle_control)

    def _build_coord_grid(self, H, W, device):
        lin_y = torch.linspace(-1, 1, H, device=device)
        lin_x = torch.linspace(-1, 1, W, device=device)
        gy2d, gx2d = torch.meshgrid(lin_y, lin_x, indexing="ij")
        return torch.stack([gx2d, gy2d], dim=0).unsqueeze(0)
    
    def _build_grid_centers(self, H, W, device):
        cx = torch.arange(W, dtype=torch.float32, device=device) / W + 0.5 / W
        cy = torch.arange(H, dtype=torch.float32, device=device) / H + 0.5 / H
        gx, gy = torch.meshgrid(cx, cy, indexing="xy")
        return torch.stack([gx, gy], dim=0).unsqueeze(0)

    def compute_gecco_features(self, offsets_t, high_res_image, high_res_sdf, timing_breakdown=None):
        """Sample SDF-aware high-res GECCO features at current noisy positions."""
        if high_res_image is None:
            raise ValueError("high_res_image is required when enable_gecco=True")
        if self.sdf_features and high_res_sdf is None:
            raise ValueError("high_res_sdf is required when sdf_features=True and enable_gecco=True")

        # Recompute grid_centers for the actual spatial size (supports grid sizes
        # different from the one used at training time).
        _, _, H, W = offsets_t.shape
        cuda_timing_enabled = (
            timing_breakdown is not None
            and torch.cuda.is_available()
            and offsets_t.is_cuda
        )
        grid_centers = self._time_block(
            timing_breakdown,
            "ctrl.gecco_grid",
            lambda: self._build_grid_centers(H, W, offsets_t.device),
            cuda_timing_enabled,
        )

        positions = grid_centers + offsets_t / H
        sample_coords = positions.permute(0, 2, 3, 1) * 2.0 - 1.0

        gecco_input = high_res_image if not self.sdf_features else torch.cat([high_res_image, high_res_sdf], dim=1)
        gecco_feats_hr = self._time_block(
            timing_breakdown,
            "ctrl.gecco_extract",
            lambda: self.gecco_extractor(gecco_input),
            cuda_timing_enabled,
        )
        grid_sampled = self._time_block(
            timing_breakdown,
            "ctrl.gecco_sample",
            lambda: F.grid_sample(
                gecco_feats_hr,
                sample_coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ),
            cuda_timing_enabled,
        )
        return grid_sampled

    @staticmethod
    def _extract_control_state_dict(ctrl_state):
        """Extract the underlying control state-dict from a checkpoint-like object.

        Accepts either a dict with one of the keys ('control_net','model_state_dict','state_dict')
        or a raw state-dict mapping parameter names to tensors.
        Raises informative errors on mismatch.
        """
        if not isinstance(ctrl_state, dict):
            raise TypeError(f"Control checkpoint must be a dict, got {type(ctrl_state)}")

        for key in ("control_net", "model_state_dict", "state_dict"):
            value = ctrl_state.get(key)
            if isinstance(value, dict):
                return value

        if ctrl_state and all(hasattr(v, "shape") for v in ctrl_state.values()):
            return ctrl_state

        keys_preview = ", ".join(list(ctrl_state.keys())[:8])
        raise KeyError(
            "Could not find control weights in checkpoint. "
            f"Expected one of ['control_net', 'model_state_dict', 'state_dict']; found keys: [{keys_preview}]"
        )

    def safe_load_state_dict(self, state, strict=False):
        """Load weights into *this* model safely and run validation.

        Parameters
        - state: a loaded checkpoint dict or a raw state-dict mapping parameter names to tensors.
        - strict: passed to :meth:`load_state_dict`
        - raise_on_missing: if True, raise when missing keys are present
        - verbose: if True, print missing/unexpected key summaries

        This method does not return a value; it raises on fatal problems.
        """
        if isinstance(state, dict):
            ckpt = self._extract_control_state_dict(state)
        else:
            ckpt = state

        # --- MINIMAL FIX: Prevent ZeroConv <-> AdaptiveGate mismatch ---
        # Check if the checkpoint contains any 'gate' weights in the injection layers
        ckpt_has_gates = any(".gate." in k for k in ckpt.keys() if "inject" in k)
        
        if self.enable_adaptive_gate_injection and not ckpt_has_gates:
            raise RuntimeError(
                "CRITICAL MISMATCH: Model is initialized with AdaptiveGateInjection, "
                "but the loaded checkpoint was trained using ZeroConvInjection (no gate weights)."
            )
        elif not self.enable_adaptive_gate_injection and ckpt_has_gates:
            raise RuntimeError(
                "CRITICAL MISMATCH: Model is initialized with ZeroConvInjection, "
                "but the loaded checkpoint was trained using AdaptiveGateInjection (contains gate weights)."
            )
        # ---------------------------------------------------------------

        self.load_state_dict(ckpt, strict=strict)


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

    def _time_block(self, timing_breakdown, key, fn, enabled):
        if timing_breakdown is None or not enabled:
            return fn()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        timing_breakdown.setdefault(key, []).append((start_event, end_event))
        return result

    def forward(self, x, t, cond=None, timing_breakdown=None):
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

        cuda_timing_enabled = (
            timing_breakdown is not None
            and torch.cuda.is_available()
            and x.is_cuda
        )

        controls = self._time_block(
            timing_breakdown,
            "controlnet_forward",
            lambda: self.control(
                x, t, hrs, tgt,
                high_res_sdf=hrs_sdf if self.control.sdf_features else None,
                target_sdf_map=tgt_sdf if self.control.sdf_features else None,
                target_smart_init_map=smart if self.control.smart_init_features else None,
                timing_breakdown=timing_breakdown,
            ),
            cuda_timing_enabled,
        )
        return self._time_block(
            timing_breakdown,
            "unet_forward",
            lambda: self.locked(x, t, cond=cond, controls=controls, timing_breakdown=timing_breakdown),
            cuda_timing_enabled,
        )
