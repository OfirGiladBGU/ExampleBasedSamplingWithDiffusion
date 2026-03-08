# Control V2 (Dynamic ControlNet) flow:

Key difference from V1: instead of a static 32x32 condition, V2 **dynamically recomputes local density at every denoising step** by sampling the full-resolution image at the current estimated point positions.

**Dynamic Density Computation (runs every step):**

* **Noisy offsets:** `offsets_t` (B, 2, 32, 32) in cell-relative units ~[-1, 1].
* **Position recovery:** `positions = grid_centers + offsets_t / 32` -> (B, 2, 32, 32) in [0, 1].
* **Coordinate transform:** `coords = positions * 2 - 1` -> [-1, 1] for `grid_sample`.
* **Sampling:** `dynamic_density = F.grid_sample(high_res_img, coords)` -> (B, 1, 32, 32).

**Condition Flow:**

* **Input 1:** Full-resolution grayscale image -> `high_res` (B, 1, 512, 512).
* **Input 2:** Area-downsampled density -> `target_density` (B, 1, 32, 32).
* **Input 3:** Dynamic density (computed above) -> `dynamic_density` (B, 1, 32, 32).
* **Concatenation:** `cat([offsets_t, target_density, dynamic_density])` -> (B, 4, 32, 32).
* **Hint Encoder:** 3-layer CNN (4ch -> 16 -> 32 -> 2ch) -> `hint` (B, 2, 32, 32).
* **Fusion:** `x = offsets_t + hint` (B, 2, 32, 32).

**Control Encoder Flow:**

* Same deep-copied encoder structure as V1 [128, 256, 384].
* **Injection:** Uses `AdaptiveGateInjection` instead of `ZeroConv2d`:
  * `output = sigmoid(gate_conv(ctrl)) * transform_conv(ctrl)`.
  * Gate bias init = -4.0 -> sigmoid(-4) ~ 0.018 (near-zero at start, same stability as ZeroConv).
  * Learns to adaptively scale control injection per spatial location (can dial down in crowded regions).

**Injection into Frozen UNet:**

* Same additive injection as V1 -- signals are pre-gated inside the control branch.
* Compatible with existing `controls` parameter in `DenoiserModel.forward`.

**Training:**

* **Frozen:** Base UNet (~27M params, locked).
* **Trainable:** DynamicControlNet (~16M params + gate params).
* **Data:** `DynamicStippleDataset` returns `(high_res_img, target_density, gt_offsets)`.
* **Loss:** MSE(noise_pred, noise).

---

# Sampling:

* `DynamicControlledDenoiser` wraps frozen denoiser + DynamicControlNet.
* Call `set_condition(high_res_img, target_density)` once.
* At **every** reverse-diffusion step, `compute_dynamic_density` re-queries the high-res image at current point positions -- the condition evolves as denoising progresses.

---

# NOTE:

- V2 keeps the same baseline UNet and the same diffusion loss as V1: `MSE(noise_pred, noise)`.
- Main change 1: the condition is now dynamic. At every denoising step, V2 computes `dynamic_density` by sampling the high-resolution image at the current estimated point positions.
- Main change 2: the hint encoder input is 4 channels instead of 1: `[offsets_t(2), target_density(1), dynamic_density(1)]`.
- Main change 3: `ZeroConv2d` is replaced with `AdaptiveGateInjection` (sigmoid-gated control), so the model can learn how much control to inject per location.
- The control outputs are still injected as `controls` (additive residuals on encoder skips + middle), exactly like V1 at the UNet interface.
- Final points are obtained only after the reverse diffusion loop: predict noise at each step -> update `x_t` down to `x_0` -> inverse OT -> `(N, 2)` points.
