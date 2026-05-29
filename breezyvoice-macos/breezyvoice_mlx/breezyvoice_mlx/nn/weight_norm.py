"""Weight normalization for MLX.

MLX has no built-in equivalent to ``torch.nn.utils.weight_norm`` (see
ml-explore/mlx#1888, PR #1921 closed unmerged). BreezyVoice's HiFiGAN vocoder
(``cosyvoice/hifigan/generator.py``) and F0 predictor
(``cosyvoice/hifigan/f0_predictor.py``) apply weight_norm to ~21 conv layers.

KEY INSIGHT — we usually don't need a *runtime* WeightNorm layer at all.
PyTorch weight_norm reparameterizes a weight ``w`` as::

    w = g * v / ||v||

where ``||v||`` is the L2 norm of ``v`` computed over every dimension *except*
``dim`` (default ``dim=0``). BreezyVoice calls ``remove_weight_norm()`` before
inference, which fuses ``g`` and ``v`` back into a single ``w``. So for an
inference-only MLX port the right move is to **fuse at weight-conversion time**
(``fuse_weight_norm`` below) and ship plain conv weights — no custom layer,
zero runtime cost, bit-for-bit equivalent to the PyTorch ``remove_weight_norm``
path.

The ``WeightNorm`` ``nn.Module`` is provided only for the rare cases where you
want to keep the reparameterization live (e.g. fine-tuning on-device).

NOTE on layout: this operates in the *PyTorch* weight layout
(Conv1d weight = ``(out_channels, in_channels, kernel)``, ``dim=0``). Convert
the fused weight to MLX's Conv1d layout ``(out_channels, kernel, in_channels)``
*after* fusing — see ``tools/convert_weights.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _norm_except_dim(v: mx.array, dim: int) -> mx.array:
    """L2 norm of ``v`` over all axes except ``dim``, keeping shape broadcastable.

    Mirrors ``torch._weight_norm`` / ``torch.norm_except_dim(v, 2, dim)``.
    Returns an array whose only non-1 axis is ``dim`` (e.g. ``(out, 1, 1)`` for a
    Conv1d weight with ``dim=0``).
    """
    if dim == -1:
        return mx.sqrt(mx.sum(v * v))
    axes = tuple(a for a in range(v.ndim) if a != dim)
    return mx.sqrt(mx.sum(v * v, axis=axes, keepdims=True))


def fuse_weight_norm(g: mx.array, v: mx.array, dim: int = 0) -> mx.array:
    """Fuse a (``weight_g``, ``weight_v``) pair into the effective weight ``w``.

    Equivalent to PyTorch's ``remove_weight_norm``: ``w = g * v / ||v||``.

    Args:
        g: the ``weight_g`` tensor. PyTorch stores it with the ``dim`` axis kept
           and all others size-1 (e.g. ``(out, 1, 1)`` for Conv1d ``dim=0``).
        v: the ``weight_v`` tensor (full weight shape).
        dim: the dimension preserved by the norm (PyTorch default ``0``).

    Returns:
        The fused weight, same shape as ``v``, still in PyTorch layout.
    """
    return g * (v / _norm_except_dim(v, dim))


class WeightNorm(nn.Module):
    """Runtime weight-norm reparameterization wrapper for an MLX conv/linear layer.

    Prefer ``fuse_weight_norm`` for inference. Use this only when you need the
    live ``g``/``v`` split (training / on-device fine-tuning).

    The wrapped module must expose a ``weight`` attribute; ``dim`` is given in
    that module's own layout.
    """

    def __init__(self, module: nn.Module, dim: int = 0):
        super().__init__()
        self.module = module
        self.dim = dim
        w = module.weight
        self.weight_g = _norm_except_dim(w, dim)
        self.weight_v = w
        # The fused weight is recomputed each forward; drop the original to avoid
        # it shadowing the reparameterization.
        del self.module.weight

    def __call__(self, *args, **kwargs):
        self.module.weight = fuse_weight_norm(self.weight_g, self.weight_v, self.dim)
        return self.module(*args, **kwargs)
