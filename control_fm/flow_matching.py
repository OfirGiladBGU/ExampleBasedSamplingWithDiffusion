"""Flow Matching (Optimal-Transport / linear path) core for the control_fm baseline.

Self-contained replacement for the DDPM ``DiffusionModel`` process: rectified-flow
/ conditional-OT Flow Matching. No pretrained diffusion baseline, no beta schedule,
no ancestral sampler, no dependency on models.Diffusion / utils.Config.ParseSampleConfig.

Convention
----------
    t in [0, 1],   t = 0  -> data,   t = 1 -> noise
    x_t = (1 - t) * x_data + t * eps          (linear / OT interpolant)
    v   = d x_t / d t = eps - x_data          (constant target velocity along the path)

The network predicts v(x_t, t). Sampling integrates the ODE backward from t = 1
(pure noise) to t = 0 (data). Because the OT path is linear, the target velocity is
constant, so an Euler step is exact when the model is perfect -- which is exactly what
makes the overfit sanity check converge with very few steps.
"""

import json

import torch
import torch.nn as nn


def build_velocity_network(config_path, device="cpu"):
    """Instantiate the velocity U-Net directly from a model config (no checkpoint).

    Reuses the existing DenoiserModel architecture as the velocity field. Only the
    ``model`` block of the JSON config is used; the diffusion block and the pretrained
    checkpoint are intentionally ignored.
    """
    with open(config_path) as f:
        config = json.load(f)
    model_cfg = dict(config["model"])
    model_cfg.pop("cond", None)  # conditioning is handled by the control branch
    from models.Denoiser import DenoiserModel

    model = DenoiserModel(**model_cfg)
    return model.to(device)


class FlowMatching(nn.Module):
    """Stateless Flow-Matching helper: interpolant, velocity target, and ODE sampler.

    Coupling
    --------
    ``coupling="gaussian"`` (default): the source endpoint (t=1) is N(0, I) -- standard
    Flow Matching that generates data from pure noise.

    ``coupling="smartinit"``: the source endpoint (t=1) is the *smart-init offsets*, so the
    flow learns to transport the (clumpy) smart-init point cloud to the (blue-noise) data.
    This is the OT/bridge coupling -- the model learns to *move* the smart-init to the target
    rather than to denoise. ``source_jitter_px`` adds light Gaussian jitter (in offset/pixel
    units) to the source during training, widening the single cached smart-init into a small
    cloud so the model generalizes to different smart-inits at inference.

    Both couplings reuse the same linear interpolant and velocity target; only the source
    endpoint fed as ``eps`` changes.
    """

    def __init__(self, device="cpu", t_scale=1000.0, eps_t=1e-4,
                 coupling="gaussian", source_jitter_px=0.0):
        super().__init__()
        self.device = device
        self.t_scale = float(t_scale)
        self.eps_t = float(eps_t)
        self.coupling = str(coupling)
        self.source_jitter_px = float(source_jitter_px)
        if self.coupling not in ("gaussian", "smartinit"):
            raise ValueError(f"Unknown coupling '{self.coupling}' (expected 'gaussian' or 'smartinit')")

    # ── coupling / source endpoint ──────────────────────────────────
    def sample_source(self, x_data, smart_init=None):
        """Return the source endpoint (t=1) of the interpolant.

        gaussian  -> N(0, I) with the shape of ``x_data``.
        smartinit -> ``smart_init`` offsets (+ optional pixel jitter), broadcast to x_data.
        """
        if self.coupling == "smartinit":
            if smart_init is None:
                raise ValueError("coupling='smartinit' requires smart_init offsets as the source")
            src = smart_init.to(device=x_data.device, dtype=x_data.dtype)
            if src.shape != x_data.shape:
                src = src.expand_as(x_data).contiguous()
            if self.source_jitter_px > 0.0:
                src = src + torch.randn_like(src) * self.source_jitter_px
            return src
        return torch.randn_like(x_data)

    def start_state(self, shape, device=None, smart_init=None):
        """Initial ODE state at t_start for sampling (no jitter -- use the real smart-init)."""
        device = device or self.device
        if self.coupling == "smartinit":
            if smart_init is None:
                raise ValueError("coupling='smartinit' requires smart_init offsets to start sampling")
            src = smart_init.to(device)
            if list(src.shape) != list(shape):
                src = src.expand(*shape).contiguous()
            return src
        return torch.randn(shape, device=device)

    # ── training-time quantities ────────────────────────────────────
    def sample_t(self, batch_size, device=None):
        """Uniform t in (eps_t, 1 - eps_t) -- endpoints avoided for stability."""
        device = device or self.device
        return torch.rand(batch_size, device=device) * (1.0 - 2.0 * self.eps_t) + self.eps_t

    @staticmethod
    def _broadcast(t, ndim):
        return t.view(t.shape[0], *([1] * (ndim - 1)))

    def interpolate(self, x_data, eps, t):
        """x_t = (1 - t) * x_data + t * eps."""
        tt = self._broadcast(t, x_data.ndim)
        return (1.0 - tt) * x_data + tt * eps

    def velocity_target(self, x_data, eps):
        """Constant OT-path velocity: eps - x_data."""
        return eps - x_data

    def net_t(self, t):
        """Scale continuous t in [0, 1] into the range the sinusoidal embedding expects.

        get_timestep_embedding was designed for integer diffusion steps (~0..1000), so a
        raw t in [0, 1] barely moves the high-frequency bands. Scaling restores resolution.
        """
        return t * self.t_scale

    # ── sampling ────────────────────────────────────────────────────
    @torch.no_grad()
    def ode_sample(self, velocity_fn, shape, device=None, n_steps=50,
                   method="euler", t_start=1.0, x_start=None, show_tqdm=False):
        """Integrate dx/dt = v backward from t_start (noise) to 0 (data).

        Parameters
        ----------
        velocity_fn : callable(x, t_net) -> Tensor
            Predicted velocity. ``t_net`` is already scaled via ``net_t`` (matching the
            training-time argument passed to the network).
        shape : sequence[int]
            Shape of the sample tensor (used only when ``x_start`` is None).
        x_start : Tensor or None
            Optional initial state at ``t_start``. When None, starts from N(0, I).
        """
        device = device or self.device
        x = x_start.to(device) if x_start is not None else torch.randn(shape, device=device)

        ts = torch.linspace(float(t_start), 0.0, n_steps + 1, device=device)
        iterator = range(n_steps)
        if show_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_steps, desc="fm-ode")

        for i in iterator:
            t_cur = ts[i]
            t_next = ts[i + 1]
            dt = t_cur - t_next  # positive
            bt = torch.full((x.shape[0],), float(t_cur), device=device)
            v = velocity_fn(x, self.net_t(bt))
            if method == "heun":
                x_euler = x - dt * v
                bt_next = torch.full((x.shape[0],), float(t_next), device=device)
                v_next = velocity_fn(x_euler, self.net_t(bt_next))
                x = x - dt * 0.5 * (v + v_next)
            else:  # euler
                x = x - dt * v
        return x
