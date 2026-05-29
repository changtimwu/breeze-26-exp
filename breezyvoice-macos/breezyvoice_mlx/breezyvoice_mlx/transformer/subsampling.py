"""Input layer.  [PORT STATUS: IMPLEMENTED]

MLX port of LinearNoSubsampling from
../BreezyVoice/cosyvoice/transformer/subsampling.py — BreezyVoice uses
input_layer='linear' for both the text encoder and the flow encoder.

Sequential(Linear(idim, odim), LayerNorm(odim), Dropout). Weight names match
the torch nn.Sequential: out.0.{weight,bias} (Linear), out.1.{weight,bias}
(LayerNorm) — preserved here via an explicit `out` submodule list.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .embedding import EspnetRelPositionalEncoding


class LinearNoSubsampling(nn.Module):
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc: EspnetRelPositionalEncoding):
        super().__init__()
        # Mirror torch.nn.Sequential(Linear, LayerNorm, Dropout) so checkpoint
        # keys out.0.* / out.1.* line up.
        self.out = [nn.Linear(idim, odim), nn.LayerNorm(odim, eps=1e-5)]
        self.pos_enc = pos_enc
        self.right_context = 0
        self.subsampling_rate = 1

    def __call__(self, x: mx.array, x_mask: mx.array, offset: int = 0
                 ) -> Tuple[mx.array, mx.array, mx.array]:
        x = self.out[1](self.out[0](x))   # Linear -> LayerNorm (Dropout is no-op at eval)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
