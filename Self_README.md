# Base Diffusion Model flow:

**Data Representation (Optimal Transport):**

* **Input:** 1024 points in [0,1]^2 -> OT assignment to 32x32 grid -> `offsets = (point - grid_center) * 32` -> `x_0` (B, 2, 32, 32) in ~[-1, 1].
* **Inverse:** `point = offset / 32 + grid_center` -> back to [0,1]^2.

**Training Flow (forward diffusion + noise prediction):**

* **GT offsets:** `x_0` (B, 2, 32, 32).
* **Noise schedule:** 1000 linear timesteps, beta in [1e-4, 1e-2].
* **Forward diffusion:** `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise`.
* **Prediction:** `noise_pred = DenoiserModel(x_t, t, cond)` (B, 2, 32, 32).
* **Loss:** MSE(noise_pred, noise).

**DenoiserModel (UNet) flow:**

* **Input:** Noisy offsets -> `x_t` (B, 2, 32, 32).
* **Time Embedding:** Sinusoidal embedding of t -> dense1 -> dense2 -> `temb` (B, 512).
* **Optional Cond:** Class label (e.g. GBN/SOT/LDBN) -> Identity -> LayerNorm -> Linear -> `cond_emb` (B, 16, 1, 1), concatenated channel-wise.
* **Initial Conv:** `conv1` (2ch -> 128ch) at 32x32.
* **Encoder:** 3 levels, 2 ResBlocks each. Channels [128, 256, 384]. Spatial: 32 -> 16 -> 8.
* **Middle:** 2 ResBlocks at 384ch, 8x8.
* **Decoder:** Mirrors encoder. Each block receives skip connection (channel-cat with encoder output).
* **Output:** GroupNorm -> SiLU -> Conv -> `noise_pred` (B, 2, 32, 32).

**Sampling Flow (reverse diffusion):**

* **Start:** Pure Gaussian noise (B, 2, 32, 32).
* **Loop:** For t = T-1, ..., 0: predict noise, compute posterior mean + variance, sample x_{t-1}.
* **Final:** `x_0` offsets -> inverse OT -> point set (B, 1024, 2) in [0,1]^2.

---

# Config (GBN):

* UNet: `ch=128`, `ch_mult=[1,2,3]`, `num_res=2`, no attention. ~27M params.
* Diffusion: 1000 steps, linear beta, MSE loss.
* Trained unconditionally on GBN (Greedy Blue Noise) point sets.
