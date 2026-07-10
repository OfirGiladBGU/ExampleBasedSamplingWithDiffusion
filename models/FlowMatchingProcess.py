"""Flow-Matching process -- a drop-in twin of ``models.Diffusion.DiffusionModel``.

Purpose
-------
Lets the *existing* repo baseline (``train.py`` / ``sample.py`` / ``utils.Trainer`` /
``utils.Eval``) train a Flow-Matching model instead of a DDPM, selected purely by the
config file: use a ``"flow"`` block instead of a ``"diffusion"`` block. Nothing else in
the pipeline changes -- the dataset, the multi-scale loop, the eval hooks and the
checkpoint format are all process-agnostic, and ``DenoiserModel`` is unchanged (Flow
Matching alters the training *target* and the *sampler*, not the architecture).

Convention (identical to control_fm)
------------------------------------
    t in [0, 1],   t = 0 -> data,   t = 1 -> noise
    x_t = (1 - t) * x_0 + t * eps          (linear / OT interpolant)
    v   = dx_t/dt = eps - x_0              (constant target velocity)

The network predicts v(x_t, t, cond). Sampling integrates dx/dt = v from t=1 to t=0.

Interface contract (what the rest of the repo actually calls)
-------------------------------------------------------------
    .model  .device  .num_timesteps
    forward(x_start, t, noise=None, x_cond=None) -> scalar loss     (utils/Trainer)
    sample_timesteps(bs)                                            (utils/Trainer)
    p_sample_loop(shape, img=None, cond=None,
                  with_tqdm=True, with_sampling=True)               (utils/Eval, sample.py)
    set_num_timesteps(T)  reset_timesteps()                         (utils/Eval, sample.py)
    noise_fn(shape)                                                 (utils/Eval)
    state_dict()                                                    (utils/Eval, Trainer.save)

Deliberate differences from DiffusionModel
------------------------------------------
* No betas / alphas / posterior buffers -- FM has no noise schedule.
* ``set_num_timesteps(T)`` sets the number of **ODE integration steps**, not a subset of a
  beta schedule. There is no accuracy cliff at small T: a 20-50 step Euler/Heun solve is
  usually plenty, so an eval config of ``"ts": [20, 50]`` makes more sense than [50, 1000].
* ``with_sampling`` has no FM analogue (it selects DDPM ancestral noise). It is accepted for
  signature compatibility and IGNORED: this class always integrates the deterministic
  probability-flow ODE. The stochastic counterpart is the reverse SDE (eta > 0), which is a
  separate, explicit choice -- not something to trigger off a DDPM-shaped flag.
* ``p_sample_loop`` is grad-transparent (no ``@torch.no_grad()``), exactly like
  ``DiffusionModel.p_sample_loop``, so ``Trainer.sample_with_grad`` keeps working.
"""

import torch
import torch.nn as nn


class FlowMatchingModel(nn.Module):
    def __init__(self, model, device, loss="mse", t_scale=1000.0, eps_t=1e-4,
                 t_dist="uniform", t_logitnorm_m=0.0, t_logitnorm_s=1.0,
                 steps=50, ode_method="euler"):
        super().__init__()
        assert loss in ("mse", "mae"), f"Unknown loss {loss!r}"
        assert t_dist in ("uniform", "logitnormal"), f"Unknown t_dist {t_dist!r}"
        assert ode_method in ("euler", "heun"), f"Unknown ode_method {ode_method!r}"

        # 'device' is a plain attribute (not a buffer) -- utils/Eval reads model.device,
        # matching DiffusionModel's convention.
        self.device = device
        self.model = model

        self.loss = loss
        self.t_scale = float(t_scale)
        self.eps_t = float(eps_t)
        self.t_dist = t_dist
        self.t_logitnorm_m = float(t_logitnorm_m)
        self.t_logitnorm_s = float(t_logitnorm_s)
        self.ode_method = ode_method

        # ODE step count. `num_timesteps` keeps the name the rest of the repo expects.
        self.default_steps = int(steps)
        self.num_timesteps = int(steps)

        self.to(self.device)

    # ── schedule-shaped API (repurposed as ODE step count) ──────────
    def set_num_timesteps(self, T):
        self.num_timesteps = max(1, int(T))

    def reset_timesteps(self):
        self.num_timesteps = self.default_steps

    def set_subsequence(self, subsequence=None):
        # No beta schedule to subset; kept so callers written against DiffusionModel do not
        # explode. The step count is what matters, and it is set via set_num_timesteps().
        return

    # ── basics ──────────────────────────────────────────────────────
    def noise_fn(self, shape):
        return torch.randn(shape, dtype=torch.float32).to(self.device)

    def sample_timesteps(self, batch_size):
        """Continuous t in (eps_t, 1 - eps_t).

        Trainer draws this ONCE per iteration and reuses it across every scale, so the
        shared-timestep-across-scales property of the baseline is preserved.
        """
        lo, hi = self.eps_t, 1.0 - self.eps_t
        if self.t_dist == "logitnormal":
            # SD3: concentrate training on the harder middle of the path.
            u = torch.randn(batch_size, device=self.device) * self.t_logitnorm_s + self.t_logitnorm_m
            return torch.sigmoid(u).clamp(lo, hi)
        return torch.rand(batch_size, device=self.device) * (hi - lo) + lo

    @staticmethod
    def _broadcast(t, ndim):
        return t.view(t.shape[0], *([1] * (ndim - 1)))

    def net_t(self, t):
        """get_timestep_embedding was built for integer DDPM steps (~0..1000); rescale."""
        return t * self.t_scale

    # ── training ────────────────────────────────────────────────────
    def forward(self, x_start, t, noise=None, x_cond=None):
        assert t.shape[0] == x_start.shape[0]
        if noise is None:
            noise = self.noise_fn(x_start.shape)

        tt = self._broadcast(t.to(x_start.dtype), x_start.ndim)
        x_t = (1.0 - tt) * x_start + tt * noise
        target = noise - x_start                       # constant OT-path velocity

        predictions = self.model(x_t, self.net_t(t), x_cond)

        errs = predictions - target
        if self.loss == "mse":
            errs = torch.square(errs)
        else:
            errs = torch.abs(errs)

        loss = errs.view(errs.shape[0], -1).mean(1)
        return loss.mean()

    # ── sampling ────────────────────────────────────────────────────
    # NOTE: deliberately NOT @torch.no_grad(), matching DiffusionModel.p_sample_loop.
    # utils/Trainer.sample_with_grad() backprops through the sampler; a no_grad decorator
    # here would silently detach it. Every read-only caller already wraps in no_grad.
    def p_sample_loop(self, shape, img=None, cond=None, with_tqdm=True, with_sampling=True):
        """Integrate dx/dt = v backward from t=1 (noise) to t=0 (data).

        ``with_sampling`` is accepted for DiffusionModel signature parity and ignored.
        """
        if img is None:
            img = self.noise_fn(shape)
        else:
            shape = img.shape
        img = img.to(self.device)

        if cond is not None:
            cond = cond.to(self.device)

        n_steps = int(self.num_timesteps)
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)

        iterator = range(n_steps)
        if with_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_steps, desc="fm-ode")

        for i in iterator:
            t_cur, t_next = ts[i], ts[i + 1]
            dt = t_cur - t_next  # positive
            bt = torch.full((img.shape[0],), float(t_cur), device=self.device)
            v = self.model(img, self.net_t(bt), cond)

            if self.ode_method == "heun" and i < n_steps - 1:
                x_euler = img - dt * v
                bt_next = torch.full((img.shape[0],), float(t_next), device=self.device)
                v_next = self.model(x_euler, self.net_t(bt_next), cond)
                img = img - dt * 0.5 * (v + v_next)
            else:
                img = img - dt * v

        return img