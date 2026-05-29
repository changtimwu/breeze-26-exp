"""F0 predictor (mel -> F0 contour).  [PORT STATUS: STUB — MEDIUM, weight_norm]

PyTorch source: ../BreezyVoice/cosyvoice/hifigan/f0_predictor.py (56 lines)
Class: ConvRNNF0Predictor — 5x weight_norm(Conv1d) + ELU, then a final conv.

weight_norm -> fuse at conversion time. Otherwise pure Conv1d + ELU, easy.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ConvRNNF0Predictor(nn.Module):  # TODO: implement
    """MLX port of ConvRNNF0Predictor. NOT YET IMPLEMENTED."""

    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        raise NotImplementedError("ConvRNNF0Predictor MLX port pending.")

    def __call__(self, x: mx.array) -> mx.array:  # pragma: no cover
        raise NotImplementedError
