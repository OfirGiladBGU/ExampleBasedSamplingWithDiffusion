"""Tier-2: bottleneck self-attention for control_fm_v2 (position-encoding free).

Why a new module instead of flipping ``attn_middle`` on ``DenoiserModel``
----------------------------------------------------------------------
``models/Layers.AttentionBlock`` already exists and is *almost* what we want -- it has
no positional encoding, so it is grid-transfer safe. But it is (a) **single-head** and
(b) its output projection is ``NIN = nn.Linear`` with default init, i.e. **not
zero-initialised**. The upgrade guide asks for 4 heads and a zero-init output so the
block starts as an exact identity, matching the codebase's zero-init warm-start
philosophy (zero-conv / adaptive gates / SPADE gamma-beta / out_conv).

Editing ``models/Layers.py`` would also change ``control_v4`` and ``control_gt_free``,
which share it. So control_fm_v2 defines its own block and *attaches* it to an already
built network. No shared model code is touched.

Constraints honoured (see the guide's "non-negotiable constraints")
-------------------------------------------------------------------
* **Bottleneck only.** Never inserted at the high-res encoder/decoder levels.
* **No positional encoding of any kind.** Attention is permutation-equivariant over
  tokens; the convolutional trunk carries all spatial information. This is what keeps
  the model resolution-transferable (train at G=32, sample at G=8..112). A learned
  absolute position table would break that property and is explicitly forbidden.
* **Zero-init output projection**, so the block is an exact identity at step 0.
"""

import torch
import torch.nn as nn

from models.Layers import NormalizeLayer, SequentialWithEmbedding


class BottleneckAttention(nn.Module):
    """Multi-head self-attention over the bottleneck feature map. No positional encoding.

    The ``forward(x, emb, cond)`` signature matches ``ResnetBlock`` so the block can be
    dropped straight into a ``SequentialWithEmbedding``; ``emb`` and ``cond`` are accepted
    and ignored (the bottleneck attention is unconditional -- its job is purely global
    token mixing).

    Token count is ``h * w`` at the bottleneck: 8x8 = 64 tokens for a G=32 grid,
    24x24 = 576 for G=96. Cost stays negligible relative to the conv trunk.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = int(channels)
        self.num_heads = int(num_heads)

        self.norm = NormalizeLayer(channels)  # GroupNorm(1, C) -- matches the rest of the net
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True,
        )
        # Zero-init the output projection -> the residual branch contributes exactly 0 at
        # init, so inserting this block cannot perturb a trained network. It only starts
        # contributing once gradients flow.
        nn.init.zeros_(self.mha.out_proj.weight)
        if self.mha.out_proj.bias is not None:
            nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, x, emb=None, cond=None):
        b, c, h, w = x.shape
        z = self.norm(x)
        # (B, C, H, W) -> (B, H*W, C) tokens; no positional information added.
        z = z.flatten(2).transpose(1, 2)
        z, _ = self.mha(z, z, z, need_weights=False)
        z = z.transpose(1, 2).reshape(b, c, h, w)
        return x + z


def attach_bottleneck_attention(denoiser, num_heads=4):
    """Insert a ``BottleneckAttention`` between the two middle ResnetBlocks, in place.

    ``DenoiserModel.middle`` is ``SequentialWithEmbedding(ResnetBlock, ResnetBlock)`` when
    ``attn_middle=False`` (which control_fm_v2 always passes -- we attach our own block
    instead of the single-head one ``DenoiserModel`` would build).

    IMPORTANT ordering constraint
    -----------------------------
    ``DynamicControlNet`` does ``copy.deepcopy(denoiser.middle)``. Call this function
    **before** constructing the control branch so the control middle gets its own
    (independently trained) attention block. Attaching afterwards would leave the control
    branch attention-free -- a silent asymmetry.

    Returns the number of channels the attention was built for (for assertions/logging).
    """
    kids = list(denoiser.middle.children())
    if len(kids) != 2:
        raise RuntimeError(
            f"Expected denoiser.middle to hold exactly 2 ResnetBlocks (attn_middle=False), "
            f"found {len(kids)}. Refusing to attach attention over an unknown middle block."
        )
    first = kids[0]
    channels = first.conv2.net[1].weight.shape[0]  # out-channels of the first middle ResnetBlock
    denoiser.middle = SequentialWithEmbedding(
        kids[0], BottleneckAttention(channels, num_heads=num_heads), kids[1],
    )
    return channels