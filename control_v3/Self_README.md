# Control V3.7 (GECCO + AdaptiveGate + Min-SNR) flow:

Key differences from earlier V3 variants:
- SDF conditioning is removed.
- Target density is continuous (not binarized).
- GECCO dynamic features are used in overfit by default (`--enable-gecco`).
- Injection uses `AdaptiveGateInjection` (V3.2 revert).
- Overfit and train use Min-SNR-gamma weighted denoising loss.
- Sampling uses full reverse diffusion with optional RePaint micro-loops (`--resample-jumps`).

**GECCO Dynamic Feature Computation (when enabled, runs every step):**

- **Noisy offsets:** `offsets_t` (B, 2, 32, 32).
- **Position recovery:** `positions = grid_centers + offsets_t / grid_size` -> (B, 2, 32, 32) in [0, 1].
- **Coordinate transform:** `sample_coords = positions * 2 - 1` -> [-1, 1] for `grid_sample`.
- **Feature extraction:** `gecco_feats_hr = gecco_extractor(high_res_img)` -> (B, 16, H, W) by default.
- **Sampling:** `gecco_dynamic = F.grid_sample(gecco_feats_hr, sample_coords)` -> (B, 16, 32, 32).

**Condition Flow:**

- **Input 1:** Noisy offsets `offsets_t` -> (B, 2, 32, 32).
- **Input 2:** Continuous target density `target_density` -> (B, 1, 32, 32).
- **Input 3:** Static coordinate grid `coord_grid` -> (B, 2, 32, 32).
- **Input 4 (optional):** GECCO dynamic features -> (B, 16, 32, 32).
- **Concatenation:**
  - GECCO on: `cat([offsets_t, target_density, coord_grid, gecco_dynamic])` -> (B, 21, 32, 32).
  - GECCO off: `cat([offsets_t, target_density, coord_grid])` -> (B, 5, 32, 32).
- **Hint Encoder:** 3-layer dilated CNN -> `32 -> 64(d=2) -> 128(d=4)`.
- **Fusion:** `x = ctrl_conv1(offsets_t) + hint`.

**Control Encoder Flow:**

- Same deep-copied control encoder + middle from the frozen denoiser.
- **Injection:** `AdaptiveGateInjection` per skip + middle:
  - `inj = sigmoid(gate(ctrl)) * transform(ctrl)`.
  - Current init in V3.7 sets transform/gate weights to zero, so startup injection is near-zero.

**Training (overfit and full training sync):**

- **Frozen:** Base denoiser.
- **Trainable:** DynamicControlNet branch.
- **Loss:** Min-SNR-gamma weighted denoising MSE:
  - unreduced MSE -> per-sample mean
  - `snr = alpha_bar_t / (1 - alpha_bar_t)`
  - `weight = clamp(snr, max=gamma) / snr`
  - final loss = mean(per-sample-mse * weight)
- `gamma` is configurable (`--min-snr-gamma`, default 5.0; 0 disables weighting).

---

# Sampling:

- `DynamicControlledDenoiser` wraps frozen denoiser + DynamicControlNet.
- Call `set_condition(high_res_img, target_density)` once.
- Reverse diffusion runs on the 32x32 offset grid.
- RePaint-style resampling is available via `--resample-jumps` and runs across the full schedule.

---

# NOTE:

- V3.7 active stack = GECCO (optional, default-on in overfit), AdaptiveGateInjection, Min-SNR, full resampling.
- No SDF channel in conditioning.
- No target-density binarization in overfit/train synced path.
- High-res source image can be any spatial size; conditioning is projected to the 32x32 diffusion grid domain.
