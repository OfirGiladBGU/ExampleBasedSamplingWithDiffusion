"""
train_free_v2: Sinkhorn-Guided Diffusion Posterior Sampling

Zero-shot point cloud generation with no training required.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from . import sinkhorn_loss
from . import guided_sample_dps
from . import utils_guidance

__all__ = [
    'sinkhorn_loss',
    'guided_sample_dps',
    'utils_guidance',
]
