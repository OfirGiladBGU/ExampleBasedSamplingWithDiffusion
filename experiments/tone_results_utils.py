"""Shared pieces for the tone-correction experiment.

Library only -- no entry point and no argument parser. Imported by the numbered
tone_results_stage_*.py scripts; run those, not this.

The experiment asks a question a per-density optimizer cannot answer: *what density should I
ask for, so that the rendered stipple looks like the image I actually want?* Rendered stipples
have finite-radius dots, so ink overlaps and dark regions saturate -- the classical "dot gain"
problem, normally patched with a hand-tuned transfer curve. Because our sampler is a network,
we can instead solve for the pre-compensated density directly:

    rho* = argmin_rho  || Render(Sampler(rho)) - I ||^2 + lambda * R(rho)

with gradients running loss -> rendered image -> point positions -> denoiser -> rho.

What is differentiated, and what is not
---------------------------------------
Differentiated: the density channels of the conditioning (the high-resolution image and the
area-pooled target density), through the ControlNet and the locked denoiser, to the point
coordinates, through the splat, to the rendered image.

Held fixed, deliberately:
  * the SDF channels. They encode the silhouette; this experiment changes tone, not shape, so
    freezing them stops the optimizer quietly redrawing the object. They also come from a
    thresholded distance transform, which carries no useful gradient.
  * the Smart-Init point set, which comes from rejection sampling. It is not differentiable,
    it plays the role of the noise seed here, and it is drawn once from the ORIGINAL image so
    that every configuration under comparison starts from the same initialization.

Both restrictions are stated rather than hidden. The claim is that gradients flow through the
sampler -- they do -- not that every stage of the pipeline is differentiable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# control_v4 owns the model; this package only borrows from it.
from control_v4.DynamicControlNet import DynamicControlledDenoiser          # noqa: E402
from control_v4.sample_control import (_grid_centers_flat,                  # noqa: E402
                                       _offsets_to_coords_gpu, load_condition,
                                       load_pipeline)
from control_v4.smart_init import (                                         # noqa: E402
    generate_smart_init_points_from_density,
    render_smart_init_grid,
    smart_init_points_to_offsets,
)

EPS = 1e-6


# ── differentiable rendering ────────────────────────────────────────────────

def splat_points(coords, out_res, sigma_px, weights=None, chunk=65536):
    """Render (B,N,2) coords in [0,1]^2 into a (B,1,R,R) ink map of Gaussian dots.

    Evaluated directly against a pixel grid rather than by scatter-add, so the gradient path
    to `coords` is short and obvious. Pixels are processed in chunks because the intermediate
    is (B, R*R, N): at R=384 and N=1024 that is ~150M entries in one go.
    """
    B, N, _ = coords.shape
    dev, dt = coords.device, coords.dtype
    ax = (torch.arange(out_res, device=dev, dtype=dt) + 0.5) / out_res
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).reshape(-1, 2)                 # (R*R,2)

    sigma = max(float(sigma_px), 1e-3) / float(out_res)
    inv = -0.5 / (sigma * sigma)
    out = []
    for a in range(0, grid.shape[0], chunk):
        g = grid[a:a + chunk][None, :, None, :]                          # (1,c,1,2)
        d2 = ((g - coords[:, None, :, :]) ** 2).sum(-1)                  # (B,c,N)
        k = torch.exp(inv * d2)
        if weights is not None:
            k = k * weights[:, None, :]
        out.append(k.sum(-1))                                            # (B,c)
    return torch.cat(out, dim=1).reshape(B, 1, out_res, out_res)


def ink_to_darkness(ink, gain):
    """Ink density -> perceived darkness in [0,1), saturating as dots overlap.

    Beer-Lambert: darkness = 1 - exp(-gain * ink). Two dots on the same spot darken the paper
    by less than twice as much, which is exactly the dot gain this experiment compensates for.
    Smooth everywhere, so it does not obstruct the gradient.
    """
    return 1.0 - torch.exp(-float(gain) * ink.clamp_min(0.0))


def render_darkness(coords, out_res, sigma_px, gain, weights=None):
    return ink_to_darkness(splat_points(coords, out_res, sigma_px, weights), gain)


# ── the quantity being optimized ────────────────────────────────────────────

class DensityField(torch.nn.Module):
    """rho = sigmoid(logit(rho0) + delta): stays in (0,1), and starts exactly at rho0.

    param="field" gives delta one value per cell of a coarse grid, bilinearly upsampled. This
    is our method: it can compensate tone locally.
    param="curve" ties delta to a piecewise-linear function of the input density alone, i.e.
    the classical global transfer curve. Fitting it with the same optimizer against the same
    objective makes it a fair baseline rather than a straw man.
    """

    def __init__(self, image_01, param="field", field_res=64, knots=17, device="cpu"):
        super().__init__()
        self.param = param
        img = torch.as_tensor(np.asarray(image_01), dtype=torch.float32, device=device)[None, None]
        rho0 = (1.0 - img).clamp(EPS, 1 - EPS)          # density = darkness = 1 - intensity
        self.register_buffer("rho0", rho0)
        self.register_buffer("logit0", torch.log(rho0 / (1 - rho0)))
        if param == "field":
            self.delta = torch.nn.Parameter(torch.zeros(1, 1, field_res, field_res, device=device))
        elif param == "curve":
            self.delta = torch.nn.Parameter(torch.zeros(knots, device=device))
        else:
            raise ValueError(f"param must be 'field' or 'curve'; got {param!r}")

    def forward(self):
        if self.param == "field":
            d = F.interpolate(self.delta, size=self.rho0.shape[-2:], mode="bilinear",
                              align_corners=False)
        else:
            # look the curve up at each pixel's own density, as a 1-D LUT under grid_sample
            x = self.rho0 * 2.0 - 1.0                                     # (1,1,H,W)
            g = torch.stack([x, torch.zeros_like(x)], dim=-1).squeeze(1)  # (1,H,W,2)
            d = F.grid_sample(self.delta.view(1, 1, 1, -1), g, mode="bilinear",
                              align_corners=True, padding_mode="border")
        return torch.sigmoid(self.logit0 + d).clamp(EPS, 1 - EPS)

    def regularizer(self):
        """Keep the correction small and smooth: it is a tone fix, not a redrawing."""
        if self.param == "curve":
            tv = (self.delta[1:] - self.delta[:-1]).pow(2).mean()
        else:
            tv = ((self.delta[:, :, 1:] - self.delta[:, :, :-1]).pow(2).mean()
                  + (self.delta[:, :, :, 1:] - self.delta[:, :, :, :-1]).pow(2).mean())
        return self.delta.pow(2).mean(), tv


# ── the sampler, wrapped so gradients reach the conditioning ────────────────

@dataclass
class SamplerCfg:
    grid_size: int = 32
    eval_timesteps: int = 1000
    truncation_ratio: float = 0.5
    sdf_features: bool = False
    smart_init_features: bool = False
    sdf_truncate_px: float = 8.0
    smart_init_seed: int = 42
    device: str = "cuda"
    mode: str = "onestep"       # "onestep" | "unrolled"
    unroll_steps: int = 8       # gradient-carrying steps at the end of the reverse process
    t_probe: int = 120          # onestep: the timestep at which x0 is estimated
    seed: int = 0


class DifferentiableStippler(torch.nn.Module):
    """rho (1,1,H,W) -> point coordinates (1,N,2), differentiably.

    Two gradient paths, both running the real trained network:

    "onestep"  -- noise the fixed init to one low timestep, run the denoiser once, take the
                  closed-form x0 estimate. This is precisely the path the Component-2 KDE loss
                  already differentiates through during training, so it is known to produce
                  usable gradients. One forward and one backward per optimizer step.
    "unrolled" -- run the truncated reverse process, keeping gradients only for the final
                  `unroll_steps` steps. Faithful to what inference actually does, at
                  proportionally more memory.
    """

    def __init__(self, diffusion, control_net, cfg: SamplerCfg):
        super().__init__()
        self.cfg = cfg
        self.diffusion = diffusion
        # load_pipeline leaves diffusion.model as the raw denoiser; capture it before the
        # controlled wrapper is installed, otherwise this picks up the wrapper itself.
        self.base_denoiser = diffusion.model
        self.controlled = DynamicControlledDenoiser(self.base_denoiser, control_net)
        for p in self.controlled.parameters():
            p.requires_grad_(False)
        self.controlled.eval()
        diffusion.set_num_timesteps(cfg.eval_timesteps)
        diffusion.eval()
        self.register_buffer("_centers", _grid_centers(cfg.grid_size, cfg.device))
        self._ctx = None

    def prepare(self, image_path):
        """Build everything that does NOT depend on rho, once, from the original image."""
        cfg = self.cfg
        image_01, _, _, high_res_sdf, target_sdf = load_condition(
            image_path, cfg.grid_size, cfg.device,
            sdf_features=cfg.sdf_features, sdf_truncate_px=cfg.sdf_truncate_px)

        pts = generate_smart_init_points_from_density(
            image_01, n_points=cfg.grid_size * cfg.grid_size, seed=cfg.smart_init_seed)
        x_init = torch.from_numpy(smart_init_points_to_offsets(pts)).unsqueeze(0).to(cfg.device)

        smart_grid = None
        if cfg.smart_init_features:
            smart_grid = torch.from_numpy(
                render_smart_init_grid(pts, grid_size=cfg.grid_size)).unsqueeze(0).to(cfg.device)

        # Common random numbers: the diffusion noise is drawn ONCE and reused at every
        # optimizer step, so the objective is a deterministic function of rho. Resampling it
        # per step makes the objective stochastic, and at the gradient magnitudes seen here
        # the noise swamps the signal.
        g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        noise = torch.randn(x_init.shape, device=cfg.device, dtype=x_init.dtype, generator=g)

        self._ctx = dict(image_01=image_01, high_res_sdf=high_res_sdf, target_sdf=target_sdf,
                         x_init=x_init, smart_grid=smart_grid, noise=noise)
        return image_01

    def _set_condition(self, rho):
        """rho is DENSITY (1 = full ink). control_v4 conditions on INTENSITY (0 = ink), as
        stated in smart_init.generate_smart_init_points_from_density, so invert here. Passing
        density directly makes the model stipple the background."""
        c = self._ctx
        image = 1.0 - rho
        low = F.interpolate(image, size=(self.cfg.grid_size,) * 2, mode="area")
        self.controlled.set_condition(image, c["high_res_sdf"], low,
                                      c["target_sdf"], c["smart_grid"])

    def forward(self, rho, generator=None):
        if self._ctx is None:
            raise RuntimeError("call prepare(image_path) before forward()")
        cfg, d = self.cfg, self.diffusion
        self._set_condition(rho)
        x_init = self._ctx["x_init"]

        if cfg.mode == "onestep":
            t = torch.full((1,), int(cfg.t_probe), dtype=torch.int64, device=cfg.device)
            alpha = d.alphas_cumprod[int(cfg.t_probe)]
            x_t = alpha.sqrt() * x_init + (1.0 - alpha).sqrt() * self._ctx["noise"]
            eps = self.controlled(x_t, t)
            x0 = d.predict_xstart_from_noise(x_t, t, eps)
            return offsets_to_coords(x0, cfg.grid_size, self._centers)

        if cfg.mode == "unrolled":
            t_start = int(np.clip(int(cfg.eval_timesteps * cfg.truncation_ratio), 1,
                                  max(cfg.eval_timesteps - 1, 1)))
            alpha = d.alphas_cumprod[t_start]
            img = alpha.sqrt() * x_init + (1.0 - alpha).sqrt() * self._ctx["noise"]
            # Only the tail carries gradients: the early steps set global structure, and
            # holding them under no_grad keeps memory proportional to unroll_steps.
            grad_from = max(0, cfg.unroll_steps - 1)
            prev = d.model
            d.model = self.controlled
            # p_sample draws fresh noise at every step from the global RNG, so an identical
            # rho would otherwise give a different trajectory each call and the objective
            # would be stochastic. Re-seeding here fixes the whole trajectory (common random
            # numbers), which is what makes the descent curve interpretable.
            torch.manual_seed(int(cfg.seed))
            try:
                for i in reversed(range(t_start)):
                    t = torch.full((1,), i, dtype=torch.int64, device=cfg.device)
                    if i > grad_from:
                        with torch.no_grad():
                            img = d.p_sample(img, cond=None, t=t, clip_denoised=d.sample_clip,
                                             with_sampling=True)
                    else:
                        img = d.p_sample(img, cond=None, t=t, clip_denoised=d.sample_clip,
                                         with_sampling=True)
            finally:
                d.model = prev
            return offsets_to_coords(img, cfg.grid_size, self._centers)

        raise ValueError(f"mode must be 'onestep' or 'unrolled'; got {cfg.mode!r}")


def build_stippler(args, device):
    """load_pipeline + wrapper, with the argument names control_v4 already uses."""
    diffusion, control_net = load_pipeline(
        args.base_config_path, args.base_ckpt_path, args.control_ckpt_path,
        args.grid_size, args.enable_gecco, args.enable_adaptive_gate_injection,
        args.smart_init_features, args.sdf_features, args.batch_coords_features, device)
    cfg = SamplerCfg(
        grid_size=args.grid_size, eval_timesteps=args.eval_timesteps,
        truncation_ratio=args.infer_truncation_ratio, sdf_features=args.sdf_features,
        smart_init_features=args.smart_init_features, sdf_truncate_px=args.sdf_truncate_px,
        smart_init_seed=args.smart_init_seed, device=device, mode=args.mode,
        unroll_steps=args.unroll_steps, t_probe=args.t_probe, seed=args.seed)
    return DifferentiableStippler(diffusion, control_net, cfg).to(device)


def _grid_centers(grid_size, device):
    return _grid_centers_flat(grid_size, device, torch.float32)


def offsets_to_coords(offsets, grid_size, centers):
    """(B,2,G,G) grid offsets -> (B,G*G,2) coordinates in [0,1]^2.

    Delegates to control_v4 so the convention cannot drift. The scale matters: offsets are in
    GRID-CELL units, so they are divided by grid_size before being added to the cell centres.
    Adding them raw overshoots the domain and clamps every point onto the border.
    """
    return _offsets_to_coords_gpu(offsets, grid_size, centers)


# ── metrics ────────────────────────────────────────────────────────────────

def psnr(a, b, data_range=1.0):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10.0 * float(np.log10(data_range ** 2 / mse))


def ssim(a, b, data_range=1.0, win=7):
    """Windowed SSIM with a uniform filter, so scikit-image is not a dependency."""
    from scipy.ndimage import uniform_filter
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    n = win * win
    saa = (uniform_filter(a * a, win) - mu_a * mu_a) * n / (n - 1)
    sbb = (uniform_filter(b * b, win) - mu_b * mu_b) * n / (n - 1)
    sab = (uniform_filter(a * b, win) - mu_a * mu_b) * n / (n - 1)
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(s.mean())
