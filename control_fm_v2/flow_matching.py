"""Flow Matching core for control_fm_v2 -- config-file free, with Tier-1 upgrades.

Differences from control_fm (v1)
--------------------------------
1. **No config.json dependency.** The architecture is defined by ``MODEL_CONFIG`` below.
   ``build_velocity_network`` takes no path. The old ``config/GBN/config.json`` also
   carried dead ``diffusion`` / ``train`` / ``eval`` blocks; none were ever read.
2. **logit-normal t sampling** (Tier 1.2), selectable and reversible.
3. **min-SNR loss weighting** (Tier 1.3), derived for *this* interpolant, not copied
   from DDPM epsilon-loss.
4. **Optional bottleneck attention** (Tier 2), attached post-hoc; see ``attention.py``.

Convention (unchanged from v1)
------------------------------
    t in [0, 1],   t = 0 -> data,   t = 1 -> source
    x_t = (1 - t) * x_data + t * eps          (linear / OT interpolant)
    v   = d x_t / d t = eps - x_data          (constant target velocity along the path)

Deterministic (ODE) vs stochastic (SDE) sampling is selected by ``eta`` at inference
time only; ``eta = 0`` is the probability-flow ODE, ``eta = 1`` the canonical reverse SDE.
Stochastic sampling needs a Gaussian component in the source (sigma > 0).
"""

import math

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Architecture definition  (control_fm_v2 has no config.json)
# ─────────────────────────────────────────────────────────────────────────────
#
# Only parameters that actually DO something are exposed here. ``DenoiserModel`` also
# demands two more, but both are inert leftovers of the DDPM lineage:
#
#   attn_layers  must stay []    -- high-res attention is out of scope: quadratic in the
#                                   full token count, and it would wreck grid transfer.
#   attn_middle  must stay False -- DenoiserModel's own block is single-head and its output
#                                   projection (NIN = nn.Linear) is NOT zero-initialised.
#
# They are hard-wired inside ``_denoiser_kwargs()`` and deliberately kept OUT of
# MODEL_CONFIG, so nobody tunes a knob that does nothing.
#
# Tier-2 bottleneck attention lives in MODEL_CONFIG as ``attn_middle`` / ``attn_heads``.
# That refers to control_fm_v2's OWN zero-init, position-encoding-free block
# (see attention.py) -- not to DenoiserModel's.
#
#   num_channels : state channels -- the 2-channel OT offset field (dx, dy).
#                  The `concat` conditioner widens this to 2 + n_cond.
#   out_ch       : the predicted velocity has the same shape as the state -> 2.
#   ch / ch_mult : 128 -> [128, 256, 384]; 3 levels, 2 downsamples (32 -> 8).
#
MODEL_CONFIG = {
    # trunk
    "num_channels": 2,
    "out_ch": 2,
    "ch": 128,
    "ch_mult": [1, 2, 3],
    "num_res": 2,
    "dropout": 0.1,
    "resamp_with_conv": True,
    # Tier 2 -- control_fm_v2's bottleneck attention (NOT DenoiserModel's)
    "attn_middle": True,
    "attn_heads": 4,
}


def get_model_config(overrides=None):
    """Return a mutable copy of MODEL_CONFIG, optionally overridden."""
    cfg = dict(MODEL_CONFIG)
    cfg["ch_mult"] = list(cfg["ch_mult"])
    if overrides:
        cfg.update(overrides)
    return cfg


def _denoiser_kwargs(cfg):
    """MODEL_CONFIG -> DenoiserModel(**kwargs), injecting the two inert-but-required flags."""
    return {
        "num_channels": int(cfg["num_channels"]),
        "out_ch": int(cfg["out_ch"]),
        "ch": int(cfg["ch"]),
        "ch_mult": list(cfg["ch_mult"]),
        "num_res": int(cfg["num_res"]),
        "dropout": float(cfg["dropout"]),
        "resamp_with_conv": bool(cfg["resamp_with_conv"]),
        "attn_layers": [],     # never any high-res attention
        "attn_middle": False,  # v2 attaches its own zero-init multi-head block instead
    }


def resolve_attention(cfg=None, attn_middle=None, attn_heads=None):
    """Fall back to MODEL_CONFIG when the caller passes None."""
    cfg = cfg or MODEL_CONFIG
    am = cfg["attn_middle"] if attn_middle is None else bool(attn_middle)
    ah = cfg["attn_heads"] if attn_heads is None else int(attn_heads)
    return am, ah


def build_velocity_network(device="cpu", model_config=None, attn_middle=None, attn_heads=None):
    """Instantiate the velocity U-Net from MODEL_CONFIG (no config file, no checkpoint).

    ``attn_middle`` / ``attn_heads`` default to MODEL_CONFIG when left as None.

    NOTE ordering: attention is attached to ``denoiser.middle`` here, and
    ``DynamicControlNet`` deep-copies that middle -- so always build the network BEFORE the
    control branch, or the control branch silently lacks attention.
    """
    cfg = get_model_config(model_config)
    attn_middle, attn_heads = resolve_attention(cfg, attn_middle, attn_heads)

    from models.Denoiser import DenoiserModel

    model = DenoiserModel(**_denoiser_kwargs(cfg))
    if attn_middle:
        from control_fm_v2.attention import attach_bottleneck_attention

        attach_bottleneck_attention(model, num_heads=attn_heads)
    return model.to(device)


class FlowMatching(nn.Module):
    """Interpolant, velocity target, timestep distribution, loss weighting, ODE/SDE sampler.

    Coupling
    --------
    ``coupling="gaussian"``: source endpoint (t=1) is N(0, I) -- standard Flow Matching.
    ``coupling="smartinit"``: source endpoint is the smart-init offsets (+ optional jitter),
    turning the flow into an OT bridge that *moves* the clumpy point cloud to the target.

    Timestep distribution (Tier 1.2)
    --------------------------------
    ``t_dist="uniform"``    : t ~ U(eps_t, 1 - eps_t)                       (v1 behaviour)
    ``t_dist="logitnormal"``: u ~ N(m, s); t = sigmoid(u), clamped          (SD3)

    Loss weighting (Tier 1.3)
    -------------------------
    See ``loss_weight``. ``min_snr_gamma <= 0`` disables it (plain velocity MSE).
    """

    def __init__(self, device="cpu", t_scale=1000.0, eps_t=1e-4,
                 coupling="gaussian", source_jitter_px=0.0,
                 t_dist="uniform", t_logitnorm_m=0.0, t_logitnorm_s=1.0,
                 min_snr_gamma=0.0, snr_weight_normalize=True):
        super().__init__()
        self.device = device
        self.t_scale = float(t_scale)
        self.eps_t = float(eps_t)
        self.coupling = str(coupling)
        self.source_jitter_px = float(source_jitter_px)
        self.t_dist = str(t_dist)
        self.t_logitnorm_m = float(t_logitnorm_m)
        self.t_logitnorm_s = float(t_logitnorm_s)
        self.min_snr_gamma = float(min_snr_gamma)
        self.snr_weight_normalize = bool(snr_weight_normalize)

        if self.coupling not in ("gaussian", "smartinit"):
            raise ValueError(f"Unknown coupling '{self.coupling}' (expected 'gaussian' or 'smartinit')")
        if self.t_dist not in ("uniform", "logitnormal"):
            raise ValueError(f"Unknown t_dist '{self.t_dist}' (expected 'uniform' or 'logitnormal')")
        if self.min_snr_gamma > 0.0 and self.coupling != "gaussian":
            raise ValueError(
                "min_snr_gamma > 0 is only defined for coupling='gaussian'. The SNR of the "
                "interpolant assumes the source is N(0, I); with the smartinit coupling the "
                "source is a data-dependent point cloud and 'signal-to-noise ratio' has no "
                "meaning. Set min_snr_gamma=0 or switch coupling."
            )

    # ── coupling / source endpoint ──────────────────────────────────
    def sample_source(self, x_data, smart_init=None):
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
        """Draw t according to ``t_dist``, clamped to (eps_t, 1 - eps_t)."""
        device = device or self.device
        lo, hi = self.eps_t, 1.0 - self.eps_t
        if self.t_dist == "logitnormal":
            # SD3: concentrate training mass on the harder mid-range of the path.
            u = torch.randn(batch_size, device=device) * self.t_logitnorm_s + self.t_logitnorm_m
            return torch.sigmoid(u).clamp(lo, hi)
        return torch.rand(batch_size, device=device) * (hi - lo) + lo

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

    def x0_from_velocity(self, x_t, t, v):
        """Exact one-step decode of the clean endpoint: x0 = x_t - t * v.

        Algebraic inverse of ``interpolate`` -- no ODE solve. Cheap enough to run every
        training step, which is what makes the geometry loss in ``geometry_loss.py``
        affordable. Note ``dx0/dv = -t``: any loss placed on x0 has a gradient that scales
        with t and vanishes at t = 0. Valid for BOTH couplings (t = 0 is the data endpoint
        either way), unlike ``loss_weight``, which needs a Gaussian source.
        """
        tt = self._broadcast(t, x_t.ndim)
        return x_t - tt * v

    def net_t(self, t):
        """Scale continuous t into the range the sinusoidal embedding expects (~0..1000)."""
        return t * self.t_scale

    # ── Tier 1.3: loss weighting ────────────────────────────────────
    def loss_weight(self, t):
        """Per-sample weight applied to the **velocity** MSE. Returns ones when disabled.

        Derivation (do NOT copy the DDPM epsilon-loss weighting -- the interpolant differs)
        -----------------------------------------------------------------------------------
        For the linear interpolant with a Gaussian source,
            x_t = alpha_t * x0 + sigma_t * eps,   alpha_t = 1 - t,  sigma_t = t
            SNR(t) = alpha_t^2 / sigma_t^2 = (1 - t)^2 / t^2

        Min-SNR (Hang et al. 2023) is defined on the **x0-prediction** loss:
            L = min(SNR, gamma) * ||x0_hat - x0||^2

        We regress the velocity, not x0. From the interpolant, x0 = x_t - t * v exactly, so
        an error dv in the predicted velocity maps to dx0 = -t * dv, hence
            MSE_x0 = t^2 * MSE_v

        Substituting gives the weight to apply to the velocity MSE:
            w(t) = t^2 * min(SNR(t), gamma) = min( (1 - t)^2 , gamma * t^2 )

        Sanity: gamma -> inf recovers w = (1 - t)^2 (the pure x0 loss expressed in v-space);
        w -> 0 at both endpoints, concentrating gradient on the mid-path -- qualitatively the
        same region logit-normal t sampling targets. The two are therefore partly redundant;
        ablate them separately.

        Caveat: because w -> 0 as t -> 1, the model receives little gradient at the pure-noise
        end. With gaussian coupling that is exactly where sampling *starts*, so this can hurt.
        The guide flags this as the least certain of the Tier-1 changes -- revert if it
        destabilises.

        ``snr_weight_normalize`` rescales the weights to mean 1 over the batch so that
        enabling this does not silently change the effective learning rate.
        """
        if self.min_snr_gamma <= 0.0:
            return torch.ones_like(t)
        w = torch.minimum((1.0 - t) ** 2, self.min_snr_gamma * t ** 2)
        if self.snr_weight_normalize:
            w = w / w.mean().clamp(min=1e-12)
        return w

    # ── score recovery (needed for stochastic sampling) ─────────────
    def _source_stats(self, x, source_mean=None, source_std=None):
        """Return ``(s, sigma)``: the deterministic offset and Gaussian scale of the source."""
        if self.coupling == "smartinit":
            if source_mean is None:
                raise ValueError(
                    "Stochastic sampling with coupling='smartinit' needs source_mean "
                    "(the smart-init offsets used as the ODE start state)."
                )
            s = source_mean.to(device=x.device, dtype=x.dtype)
            if s.shape != x.shape:
                s = s.expand_as(x)
            sigma = self.source_jitter_px if source_std is None else source_std
        else:
            if source_mean is None:
                s = torch.zeros((), device=x.device, dtype=x.dtype)
            else:
                s = source_mean.to(device=x.device, dtype=x.dtype)
                if s.shape != x.shape:
                    s = s.expand_as(x)
            sigma = 1.0 if source_std is None else source_std
        return s, float(sigma)

    def score_from_velocity(self, x, t, v, source_mean=None, source_std=None):
        """grad_x log p_t(x) recovered in closed form from the predicted velocity.

            sigma * xi = x_t - s + (1 - t) * v
            score      = -(x_t - s + (1-t) v) / (t * sigma^2)
        """
        s, sigma = self._source_stats(x, source_mean, source_std)
        if sigma <= 0.0:
            raise ValueError("score is undefined for a source with zero Gaussian scale (sigma = 0)")
        tt = self._broadcast(t, x.ndim) if (torch.is_tensor(t) and t.ndim > 0) else t
        return -(x - s + (1.0 - tt) * v) / (tt * sigma * sigma)

    def _require_stochastic_source(self, x, source_mean, source_std):
        s, sigma = self._source_stats(x, source_mean, source_std)
        if sigma <= 0.0:
            raise ValueError(
                "Stochastic sampling (eta > 0) needs a Gaussian component in the source, but "
                "sigma = 0.\n"
                "  coupling='smartinit' with source_jitter_px = 0 gives a fully deterministic\n"
                "  interpolant: p_t has no Gaussian smoothing, so the score does not exist.\n"
                "  Fixes: (a) sample a gaussian-coupled checkpoint, (b) pass source_std equal to\n"
                "  the --fm-source-jitter-px the model was TRAINED with, or (c) use eta = 0."
            )
        return s, sigma

    # ── sampling ────────────────────────────────────────────────────
    @torch.no_grad()
    def ode_sample(self, velocity_fn, shape, device=None, n_steps=50,
                   method="euler", t_start=1.0, x_start=None, show_tqdm=False,
                   eta=0.0, source_mean=None, source_std=None):
        """Integrate from t_start (source side) down to 0 (data side).

        eta = 0 -> deterministic probability-flow ODE (euler / heun).
        eta > 0 -> reverse SDE via Euler-Maruyama (``method`` ignored). eta = 1 is canonical.
        """
        device = device or self.device
        x = x_start.to(device) if x_start is not None else torch.randn(shape, device=device)

        eta = float(eta)
        s = sigma = None
        if eta > 0.0:
            s, sigma = self._require_stochastic_source(x, source_mean, source_std)

        ts = torch.linspace(float(t_start), 0.0, n_steps + 1, device=device)
        iterator = range(n_steps)
        if show_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_steps, desc=("fm-sde" if eta > 0.0 else "fm-ode"))

        for i in iterator:
            t_cur = ts[i]
            t_next = ts[i + 1]
            dt = t_cur - t_next  # positive
            bt = torch.full((x.shape[0],), float(t_cur), device=device)
            v = velocity_fn(x, self.net_t(bt))

            if eta > 0.0 and i < n_steps - 1:
                # Reverse SDE, Euler-Maruyama, g^2 = 2*eta*t*sigma^2. The (t*sigma^2) cancels
                # against the score denominator -> no 1/t singularity.
                corr = eta * (x - s + (1.0 - t_cur) * v)
                noise_scale = sigma * math.sqrt(2.0 * eta * float(t_cur) * float(dt))
                x = x - dt * v - dt * corr + noise_scale * torch.randn_like(x)
            elif method == "heun":
                x_euler = x - dt * v
                bt_next = torch.full((x.shape[0],), float(t_next), device=device)
                v_next = velocity_fn(x_euler, self.net_t(bt_next))
                x = x - dt * 0.5 * (v + v_next)
            else:  # euler
                x = x - dt * v
        return x

    @torch.no_grad()
    def sde_sample(self, velocity_fn, shape, eta=1.0, **kwargs):
        """Stochastic (reverse-SDE) sampling. Alias for ``ode_sample(eta > 0)``."""
        kwargs.pop("eta", None)
        return self.ode_sample(velocity_fn, shape, eta=eta, **kwargs)