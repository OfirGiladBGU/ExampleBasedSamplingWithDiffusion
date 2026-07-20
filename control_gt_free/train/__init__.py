# Ensure torch.utils.checkpoint is importable for train.unroll.unroll_ode(grad_checkpoint=True).
# (torch.utils.checkpoint is not always auto-imported by `import torch`.)
try:
    import torch.utils.checkpoint  # noqa: F401
except Exception:
    pass
