"""Weight EMA for control_fm_v2 (Tier 1.1).

Why not reuse ``utils/Trainer.py``
----------------------------------
That trainer uses ``ema_pytorch.EMA``, which (a) is an external dependency and (b) wraps a
single ``nn.Module``. The control_fm training loop optimises **two** modules jointly (the
control branch and the base denoiser, since ``FREEZE_DENOISER=False``) and needs an explicit
swap-in/swap-out so that *sampling, geometry metrics and checkpoints* all come from the EMA
weights. A small self-contained implementation is simpler and dependency-free.

The classic bug this guards against
-----------------------------------
Maintaining EMA weights but still **sampling from the raw weights**. Use the
``averaged()`` context manager around every eval / sample / save site.
"""

import contextlib

import torch


class WeightEMA:
    """Exponential moving average over the state dicts of one or more modules.

    Parameters
    ----------
    modules : dict[str, nn.Module]
        Named modules to track, e.g. ``{"control_net": cnet, "denoiser": unet}``.
        Names become the keys of ``state_dict()`` so they can be stored in a checkpoint.
    decay : float
        EMA decay. 0.9999 suits late-stage smoothing over long runs; lower it (0.999) for
        short runs, otherwise the average lags the weights and EMA samples look *worse*.
    warmup_steps : int
        Below this step count the shadow is hard-copied from the live weights (decay=0),
        so early random weights never poison the average.
    """

    def __init__(self, modules, decay=0.9999, warmup_steps=0):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.modules = dict(modules)
        self.shadow = {
            name: {
                key: value.detach().clone()
                for key, value in module.state_dict().items()
                if torch.is_floating_point(value)
            }
            for name, module in self.modules.items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, step=None):
        """Call once after every accepted ``optimizer.step()``."""
        step = self.num_updates if step is None else int(step)
        decay = 0.0 if step < self.warmup_steps else self.decay
        for name, module in self.modules.items():
            live = module.state_dict()
            shadow = self.shadow[name]
            for key, s in shadow.items():
                p = live[key]
                if decay == 0.0:
                    s.copy_(p.detach())
                else:
                    s.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        self.num_updates += 1

    @contextlib.contextmanager
    def averaged(self):
        """Temporarily swap the EMA weights into the live modules.

        Use around sampling, geometry scoring and checkpoint writing::

            with ema.averaged():
                pred = sample_eval_batch(...)
        """
        backup = {
            name: {k: v.detach().clone() for k, v in module.state_dict().items()}
            for name, module in self.modules.items()
        }
        try:
            for name, module in self.modules.items():
                module.load_state_dict(self.shadow[name], strict=False)
            yield
        finally:
            for name, module in self.modules.items():
                module.load_state_dict(backup[name], strict=False)

    def state_dict(self):
        return {"decay": self.decay, "warmup_steps": self.warmup_steps,
                "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        if not state:
            return
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow", {})
        for name, tensors in shadow.items():
            if name not in self.shadow:
                continue
            for key, value in tensors.items():
                if key in self.shadow[name]:
                    self.shadow[name][key].copy_(value)