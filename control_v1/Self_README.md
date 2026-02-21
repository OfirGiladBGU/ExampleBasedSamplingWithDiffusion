# Control V1 (ControlNet) flow:

Adds a trainable ControlNet adapter to the frozen base UNet. Given a grayscale image, the model generates stipple patterns whose density follows the image tones.

**Condition Flow:**

- **Input:** Grayscale source image resized to 32x32 -> `cond_img` (B, 1, 32, 32).
- **Hint Encoder:** 3-layer CNN (1ch -> 16 -> 32 -> 2ch) -> `hint` (B, 2, 32, 32).
- **Fusion:** Element-wise add: `x = x_noisy + hint` (B, 2, 32, 32).

**Control Encoder Flow:**

- **Architecture:** Deep-copy of frozen UNet's encoder + middle blocks (trainable).
- **Time Embedding:** Same sinusoidal -> dense1 -> dense2 -> `temb` (B, 512).
- **Encoder:** Same 3-level structure [128, 256, 384]. At each ResBlock output -> `ZeroConv2d` (1x1 conv init to zero) -> control signal.
- **Middle:** Same 2 ResBlocks -> `ZeroConv2d` -> middle control signal.
- **Output:** `(encoder_controls, middle_control)` -- 7 tensors total (6 encoder + 1 middle).

**Injection into Frozen UNet:**

- Frozen UNet runs its encoder + middle normally on `x_noisy`.
- After middle: `x = x + middle_control`.
- After each encoder skip: `encoder[i][j] = encoder[i][j] + encoder_controls[i][j]`.
- Decoder then uses the modified skip connections.
- Output: `noise_pred` (B, 2, 32, 32).

**Training:**

- **Frozen:** Base UNet denoiser (~27M params, locked).
- **Trainable:** ControlNet branch (~16M params).
- **Data:** `StippleDataset` returns `(cond_img, gt_offsets)` -- condition is resized to 32x32.
- **Loss:** MSE(noise_pred, noise) -- same as base model.

---

# Sampling:

- `ControlledDenoiser` wraps frozen denoiser + ControlNet as a drop-in replacement.
- Call `set_condition(img)` once, then run standard `p_sample_loop`.
- ControlNet runs in parallel with frozen encoder at every denoising step.

