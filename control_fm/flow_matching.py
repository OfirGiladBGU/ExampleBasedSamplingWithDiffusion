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

Deterministic (ODE) vs stochastic (SDE) sampling
------------------------------------------------
The same trained velocity field can be sampled either way. ``eta`` selects it, at
inference time only, with no retraining and no config change:

    eta = 0   -> probability-flow ODE (the original deterministic Euler/Heun solve)
    eta = 1   -> the canonical reverse SDE for this path (DDPM-like stochasticity)

The reverse SDE is  dx = [v - (g^2 / 2) * score] dt + g dw,  with the natural diffusion
coefficient  g^2 = 2 * eta * t * sigma^2  for this interpolant. The score is recovered
in closed form from the predicted velocity (see ``score_from_velocity``), so no extra
network and no retraining are required.

IMPORTANT -- stochastic sampling requires a Gaussian component in the source.
With ``coupling="gaussian"`` the source is N(0, I) (sigma = 1) and everything is well
defined. With ``coupling="smartinit"`` the source is the smart-init offsets plus
``source_jitter_px`` * N(0, I). If that jitter is 0 the interpolant path is fully
deterministic, p_t carries no Gaussian smoothing, the score does not exist, and SDE
sampling is undefined -- ``ode_sample`` raises rather than silently pushing the state
onto off-path regions the velocity field was never trained on.
"""

import json
import math

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


def load_denoiser_base_weights(denoiser, ckpt_path, use_raw=False, strict=True,
                               verbose=True):
    """Load a TRAINED baseline FM velocity U-Net into ``denoiser`` (a DenoiserModel).

    Two-stage path: the baseline is trained by ``train.py`` + ``utils/Eval.compute``, which
    writes ``{"diffu": <FlowMatchingModel state_dict>, "raw": <raw state_dict>}`` (the "raw"
    key is only present when EMA-for-eval was on, i.e. the flow process). ``FlowMatchingModel``
    wraps the DenoiserModel as ``self.model``, so its keys carry a leading ``model.`` prefix;
    a bare DenoiserModel has none. We strip that prefix so the tensors line up.

    Parameters
    ----------
    use_raw : select the non-EMA weights (``"raw"``). Default takes ``"diffu"`` -- the EMA
        weights the baseline actually evaluated and shipped.
    strict : raise on ANY missing/unexpected key after prefix-stripping. Leave True: a
        silent partial load here means the "frozen base" is partly random, which is exactly
        the class of bug this whole exercise is about.

    Returns the same ``denoiser`` (loaded in place).
    """
    import torch

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"base checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and ("diffu" in ckpt or "raw" in ckpt):
        key = "raw" if use_raw else "diffu"
        if key not in ckpt:
            fallback = "diffu" if "diffu" in ckpt else "raw"
            if verbose:
                print(f"  base ckpt has no '{key}' key; falling back to '{fallback}'")
            key = fallback
        sd = ckpt[key]
    elif isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "keys"):
        sd = ckpt["model"]
    else:
        sd = ckpt  # assume a bare state_dict

    stripped = {}
    for k, v in sd.items():
        stripped[k[len("model."):] if k.startswith("model.") else k] = v

    result = denoiser.load_state_dict(stripped, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if verbose:
        src = "raw (non-EMA)" if use_raw else "diffu (EMA)"
        print(f"  loaded frozen FM base from {ckpt_path}  [weights: {src}]")
        print(f"    matched {len(stripped) - len(unexpected)} / {len(stripped)} tensors; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    if strict and (missing or unexpected):
        raise RuntimeError(
            "base-weight load is not exact -- refusing to proceed with a "
            "partly-random 'frozen' base. missing=%s unexpected=%s. "
            "The baseline model block must match --base_config_path's model "
            "block; pass --no-base-strict only if you understand why the keys "
            "differ." % (missing[:8], unexpected[:8])
        )
    return denoiser


class FlowMatching(nn.Module):
    """Stateless Flow-Matching helper: interpolant, velocity target, and ODE/SDE sampler.

    Coupling
    --------
    ``coupling="gaussian"`` (default): the source endpoint (t=1) is N(0, I) -- standard
    Flow Matching that generates data from pure noise.

    ``coupling="smartinit"``: the source endpoint (t=1) is the *smart-init offsets*, so the
    flow learns to transport the (clumpy) smart-init point cloud to the (blue-noise) data.
    This is the OT/bridge coupling -- the model learns to *move* the smart-init to the target
    rather than to denoise. ``source_jitter_px`` adds light Gaussian jitter (in offset/pixel
    units) to the source during training, widening the single cached smart-init into a small
    cloud so the model generalizes to different smart-inits at inference. It is also the
    quantity that makes stochastic (SDE) sampling well defined for this coupling.

    Both couplings reuse the same linear interpolant and velocity target; only the source
    endpoint fed as ``eps`` changes.
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
        # Ported from control_fm_v2: logit-normal t sampling (Tier 1.2) and min-SNR
        # velocity-loss weighting (Tier 1.3). Both were absent here; min_snr_gamma was
        # parsed by the trainer and silently ignored.
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
                "min_snr_gamma > 0 is only defined for coupling='gaussian'. The interpolant "
                "SNR assumes the source is N(0, I); with the smartinit coupling the source is "
                "a data-dependent point cloud and 'signal-to-noise ratio' has no meaning. "
                "Set --min-snr-gamma 0 or use --fm-coupling gaussian."
            )

    # -- coupling / source endpoint ----------------------------------
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

    # -- training-time quantities ------------------------------------
    def sample_t(self, batch_size, device=None):
        """Draw t according to ``t_dist``, clamped to (eps_t, 1 - eps_t).

        uniform     -> t ~ U(eps_t, 1 - eps_t)   (original behaviour)
        logitnormal -> u ~ N(m, s); t = sigmoid(u)  (SD3; concentrates mass mid-path)
        """
        device = device or self.device
        lo, hi = self.eps_t, 1.0 - self.eps_t
        if self.t_dist == "logitnormal":
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

    def loss_weight(self, t):
        """Per-sample weight on the VELOCITY MSE (min-SNR). Ones when disabled.

        Derived for THIS interpolant -- do not copy the DDPM epsilon weighting. With the
        linear Gaussian-source interpolant alpha_t=1-t, sigma_t=t, SNR=(1-t)^2/t^2. Min-SNR
        is defined on the x0 loss; since x0 = x_t - t v, a velocity error dv maps to
        dx0 = -t dv, so MSE_x0 = t^2 MSE_v and the weight to apply to the velocity MSE is

            w(t) = t^2 * min(SNR, gamma) = min( (1 - t)^2 , gamma * t^2 ).

        Only defined for gaussian coupling (enforced in __init__). ``snr_weight_normalize``
        rescales to batch-mean 1 so enabling it does not move the effective LR.
        """
        if self.min_snr_gamma <= 0.0:
            return torch.ones_like(t)
        w = torch.minimum((1.0 - t) ** 2, self.min_snr_gamma * t ** 2)
        if self.snr_weight_normalize:
            w = w / w.mean().clamp(min=1e-12)
        return w

    def net_t(self, t):
        """Scale continuous t in [0, 1] into the range the sinusoidal embedding expects.

        get_timestep_embedding was designed for integer diffusion steps (~0..1000), so a
        raw t in [0, 1] barely moves the high-frequency bands. Scaling restores resolution.
        """
        return t * self.t_scale

    # -- score recovery (the machinery stochastic sampling needs) -----
    def _source_stats(self, x, source_mean=None, source_std=None):
        """Return ``(s, sigma)``: the deterministic offset and Gaussian scale of the source.

        gaussian  -> s = 0,          sigma = 1
        smartinit -> s = smart-init, sigma = source_jitter_px (or an explicit override)
        """
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
        """grad_x log p_t(x), recovered in closed form from the predicted velocity.

        With  x_t = (1-t) x_0 + t (s + sigma * xi),  xi ~ N(0, I),  v = (s + sigma*xi) - x_0:

            sigma * xi = x_t - s + (1 - t) * v
            score      = -E[xi | x_t] / (t * sigma) = -(x_t - s + (1-t) v) / (t * sigma^2)

        Gaussian coupling is the special case s = 0, sigma = 1, which reduces to the
        familiar  score = -(x + (1-t) v) / t.

        Note this has a 1/t singularity; the samplers below use an algebraically
        equivalent form in which the (t * sigma^2) cancels against g^2 / 2.
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
                "  - coupling='smartinit' with source_jitter_px = 0 gives a fully deterministic\n"
                "    interpolant: p_t has no Gaussian smoothing, so the score does not exist and\n"
                "    injecting noise pushes the state onto off-path regions the velocity field\n"
                "    never saw during training.\n"
                "  Fixes: (a) sample a gaussian-coupled checkpoint, (b) pass source_std equal to\n"
                "  the --fm-source-jitter-px the model was TRAINED with, or (c) use eta = 0."
            )
        return s, sigma

    # -- sampling ----------------------------------------------------
    @torch.no_grad()
    def ode_sample(self, velocity_fn, shape, device=None, n_steps=50,
                   method="euler", t_start=1.0, x_start=None, show_tqdm=False,
                   eta=0.0, source_mean=None, source_std=None):
        """Integrate from t_start (source side) down to 0 (data side).

        Parameters
        ----------
        velocity_fn : callable(x, t_net) -> Tensor
            Predicted velocity. ``t_net`` is already scaled via ``net_t`` (matching the
            training-time argument passed to the network).
        shape : sequence[int]
            Shape of the sample tensor (used only when ``x_start`` is None).
        x_start : Tensor or None
            Optional initial state at ``t_start``. When None, starts from N(0, I).
        eta : float
            0 -> deterministic probability-flow ODE (unchanged legacy behaviour).
            > 0 -> reverse SDE integrated with Euler-Maruyama; ``method`` is ignored
            because Heun is not defined for the stochastic branch. eta = 1 is the
            canonical reverse SDE (maximum DDPM-like stochasticity for this path).
        source_mean, source_std : Tensor / float or None
            Source statistics used to recover the score. Defaults: (0, 1) for gaussian
            coupling; (smart-init offsets, ``self.source_jitter_px``) for smartinit --
            in which case ``source_mean`` must be supplied explicitly.
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

            # The final step always lands deterministically on the data side.
            if eta > 0.0 and i < n_steps - 1:
                # Reverse SDE, Euler-Maruyama, with g^2 = 2 * eta * t * sigma^2:
                #   x <- x - dt*v + (dt/2)*g^2*score + g*sqrt(dt)*z
                # Substituting score = -(x - s + (1-t)v) / (t*sigma^2) makes the
                # (t * sigma^2) cancel, so there is no 1/t singularity and the
                # correction term vanishes as t -> 0 (where x -> x_0 and v -> s - x_0).
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
        """Stochastic (reverse-SDE) sampling. Convenience alias for ``ode_sample(eta>0)``.

        eta = 1.0 is the canonical reverse SDE whose marginals match the ODE exactly.
        """
        kwargs.pop("eta", None)
        return self.ode_sample(velocity_fn, shape, eta=eta, **kwargs)