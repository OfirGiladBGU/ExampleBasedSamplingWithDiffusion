"""Grad-enabled Flow-Matching ODE unroll -- the objective-driven core.

``FlowMatching.ode_sample`` is ``@torch.no_grad`` (eval/inference only). GT-free training
needs gradients to flow through the sampler so the composite objective computed on the
FINAL generated points can update the velocity field. ``velocity_fn`` is any callable
``(x, t_net) -> v`` (a conditioner or a controlled-denoiser closure).
"""

import torch
import torch.utils.checkpoint  # module-level so it never shadows `torch` inside the fn


def unroll_ode(fm, velocity_fn, shape, device, n_steps, method="euler",
               x_start=None, grad_checkpoint=False):
    x = x_start if x_start is not None else torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        dt = t_cur - t_next
        bt = torch.full((x.shape[0],), float(t_cur), device=device)
        t_net = fm.net_t(bt)
        if grad_checkpoint:
            v = torch.utils.checkpoint.checkpoint(velocity_fn, x, t_net, use_reentrant=False)
        else:
            v = velocity_fn(x, t_net)
        if method == "heun":
            x_euler = x - dt * v
            bt2 = torch.full((x.shape[0],), float(t_next), device=device)
            v2 = velocity_fn(x_euler, fm.net_t(bt2))
            x = x - dt * 0.5 * (v + v2)
        else:
            x = x - dt * v
    return x
