# Control V3.2 ("Shock the System") flow:

Evolves from V3.0 through two critical fixes:

**V3.0 -> V3.1 "Static Anchor":** Removed GECCO-style dynamic feature sampling
(`F.grid_sample` with a 5x5-receptive-field CNN). The chaotic per-step features
acted as "poison" for high-frequency images, causing the optimizer to shut the
adaptive gates and collapse output to the base model's uniform grid.

**V3.1 -> V3.2 "Shock the System":** Replaced zero-initialized `AdaptiveGateInjection`
with `StandardInjection` (Kaiming-initialized 1x1 conv). The zero output at step 1
let the frozen U-Net learn a uniform-grid shortcut before the ControlNet could
influence it, trapping training in a local minimum (~0.18 MSE).

---

**Condition Flow (5ch -> 128ch hint):**

* **Channel breakdown:**
  * offsets_t: 2ch (current noisy offset grid)
  * target_density (area-downsampled source image at 32x32): 1ch
  * coord_grid (static [-1,1] meshgrid for spatial awareness): 2ch
  * **Total:** 5ch
* **No dynamic sampling** -- no `F.grid_sample`, no high-res feature extractor.
* **Concatenation:** `cat([offsets_t, target_density, coord_grid])` -> (B, 5, 32, 32).
* **Hint Encoder:** 3-layer CNN (5ch -> 32 -> 64 -> 128ch) with SiLU activations -> `hint` (B, 128, 32, 32).
* **Injection point:** `x = ctrl_conv1(offsets_t) + hint` (B, 128, 32, 32) -- hint added AFTER initial conv.

**Control Encoder Flow:**

* Same deep-copied encoder structure as V1/V2 [128, 256, 384].
* **Injection:** Uses `StandardInjection` -- a plain `nn.Conv2d(ch, ch, 1)` with default
  Kaiming initialization. Non-zero control signal from step 1 forces the U-Net to
  account for the image condition immediately.

**Injection into Frozen UNet:**

* Same additive injection as V1/V2 -- signals are transformed by `StandardInjection`
  inside the control branch, then added to encoder skip connections and middle block.
* Compatible with existing `controls` parameter in `DenoiserModel.forward`.

**Training:**

* **Frozen:** Base UNet (locked, set to `eval()` to disable dropout).
* **Trainable:** DynamicControlNet V3.2 (encoder copies + hint encoder + injection layers).
* **Data:** `DynamicStippleDataset` returning `(high_res_img, target_density, gt_offsets)`.
  `high_res_img` is passed through for call-signature compatibility but unused internally.
* **Loss:** MSE(noise_pred, noise).
* **Default LR:** 5e-4 (bumped from 1e-4 to help escape local minima with non-zero init).

---

# Sampling:

* `DynamicControlledDenoiser` wraps frozen denoiser + DynamicControlNet V3.2.
* Call `set_condition(high_res_img, target_density)` once.
* The condition is **static** -- same density map and coordinate grid at every
  reverse-diffusion step. The frozen U-Net's wide internal receptive field handles
  spatial routing from the 32x32 density map.
