"""Position-wise feed-forward.  [PORT STATUS: IMPLEMENTED]

MLX port of PositionwiseFeedForward from
../BreezyVoice/cosyvoice/transformer/positionwise_feed_forward.py

w_2(activation(w_1(x))). BreezyVoice's ConformerEncoder default
activation_type='swish' (not overridden in cosyvoice.yaml) -> SiLU.
Weight names w_1/w_2 match the torch module.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float = 0.0,
                 activation=nn.SiLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.activation = activation

    def __call__(self, xs: mx.array) -> mx.array:
        return self.w_2(self.activation(self.w_1(xs)))
