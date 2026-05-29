"""Length regulator.  [PORT STATUS: IMPLEMENTED — parity-tested]

MLX port of InterpolateRegulator from
../BreezyVoice/cosyvoice/flow/length_regulator.py

Nearest-interpolate the encoder output along time to the target mel length, then
refine with [Conv1d, GroupNorm, Mish] x len(sampling_ratios) + a final 1x1 Conv1d.
`model` is a list mirroring the torch nn.Sequential so checkpoint keys
(model.0=Conv, model.1=GroupNorm, model.2=Mish, ... model.{last}=Conv) line up.
Runs in MLX-native (B, T, C) layout (torch runs (B, C, T); equivalent).
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def _nearest_interp(x: mx.array, out_len: int) -> mx.array:
    """Nearest resample along time axis=1. x: (B, T, C) -> (B, out_len, C).
    Matches torch F.interpolate(mode='nearest'): src = floor(i * T/out_len)."""
    in_len = x.shape[1]
    idx = (mx.arange(out_len) * in_len // out_len).astype(mx.int32)
    return mx.take(x, idx, axis=1)


class InterpolateRegulator(nn.Module):
    def __init__(self, channels: int, sampling_ratios: Tuple,
                 out_channels: int = None, groups: int = 1):
        super().__init__()
        out_channels = out_channels or channels
        model = []
        for _ in sampling_ratios:
            model.append(nn.Conv1d(channels, channels, 3, 1, 1))
            model.append(nn.GroupNorm(groups, channels, pytorch_compatible=True))
            model.append(nn.Mish())
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = model

    def __call__(self, x: mx.array, ylens: mx.array) -> Tuple[mx.array, mx.array]:
        # x: (B, T, D); interpolate over time, then conv/norm/act stack.
        out_len = int(mx.max(ylens).item())
        h = _nearest_interp(x, out_len)          # (B, out_len, C)
        for layer in self.model:
            h = layer(h)
        return h, ylens
