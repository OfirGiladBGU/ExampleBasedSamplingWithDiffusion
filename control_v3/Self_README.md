# Control V3 (Wide-Bandwidth Dynamic ControlNet) flow:

Fixes 4 bottlenecks from V2 that caused the control signal to be ignored (loss flatlined ~0.17, outputs were uniform grids).

**What changed from V2:**

1. **Hint Encoder:** 4ch -> 2ch (before conv1) became 21ch -> 128ch (after conv1).
2. **GECCO Feature Extractor:** grid_sample now samples a learned 16ch feature map, not raw 1ch grayscale.
3. **Gate Bias:** -4.0 -> 0.0 (sigmoid=0.5, control flows freely from step 1).
4. **Coordinate Grid:** +2ch static meshgrid gives absolute spatial awareness.

---

**Dynamic Feature Computation (runs every step):**

* **Noisy offsets:** `offsets_t` (B, 2, 32, 32) in cell-relative units ~[-1, 1].
* **High-res features:** `high_res_feature_extractor(image)` -> (B, 16, 512, 512).
  * 2-layer CNN: Conv2d(1->8, k=3, pad=1) + ReLU + Conv2d(8->16, k=3, pad=1) + ReLU.
* **Position recovery:** `positions = grid_centers + offsets_t / 32` -> (B, 2, 32, 32) in [0, 1].
* **Coordinate transform:** `coords = positions * 2 - 1` -> [-1, 1] for grid_sample.
* **Feature sampling:** `dynamic_feats = F.grid_sample(high_res_feats, coords)` -> (B, 16, 32, 32).

**Condition Flow (21ch -> 128ch hint):**

* **Channel breakdown:**
  * offsets_t: 2ch
  * target_density (area-downsampled): 1ch
  * dynamic_feats (GECCO features): 16ch
  * coord_grid (static [-1,1] meshgrid): 2ch
  * **Total:** 21ch
* **Concatenation:** `cat([offsets_t, target_density, dynamic_feats, coord_grid])` -> (B, 21, 32, 32).
* **Hint Encoder:** 3-layer CNN (21ch -> 32 -> 64 -> 128ch) -> `hint` (B, 128, 32, 32).
* **Injection point:** `x = ctrl_conv1(offsets_t) + hint` (B, 128, 32, 32) -- hint added AFTER initial conv.

**Control Encoder Flow:**

* Same deep-copied encoder structure as V1/V2 [128, 256, 384].
* **Injection:** Uses `AdaptiveGateInjection` with gate bias = 0.0:
  * `output = sigmoid(gate_conv(ctrl)) * transform_conv(ctrl)`.
  * sigmoid(0) = 0.5 -- 50% of signal passes immediately, strong gradients from step 1.

**Injection into Frozen UNet:**

* Same additive injection as V1/V2 -- signals are pre-gated inside the control branch.
* Compatible with existing `controls` parameter in `DenoiserModel.forward`.

**Training:**

* **Frozen:** Base UNet (~27M params, locked).
* **Trainable:** DynamicControlNet V3 (encoder copies + hint encoder + feature extractor + gates).
* **Data:** Same `DynamicStippleDataset` as V2 returning `(high_res_img, target_density, gt_offsets)`.
* **Loss:** MSE(noise_pred, noise).

---

# Sampling:

* `DynamicControlledDenoiser` wraps frozen denoiser + DynamicControlNet V3.
* Call `set_condition(high_res_img, target_density)` once.
* At **every** reverse-diffusion step, `compute_dynamic_features` re-queries the learned feature map at current point positions -- the condition evolves as denoising progresses.
