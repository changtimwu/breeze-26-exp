"""Conformer encoder layer.  [PORT STATUS: IMPLEMENTED]

MLX port of ConformerEncoderLayer from
../BreezyVoice/cosyvoice/transformer/encoder_layer.py

BreezyVoice's encoders set macaron_style=False and use_cnn_module=False, so the
macaron FFN and conv module are absent: each layer is just
  norm_mha -> rel-pos self-attn -> residual ; norm_ff -> FFN -> residual
with ff_scale=1.0 and no norm_final. This implementation supports only that
configuration (the conv/macaron branches are intentionally omitted — add them
back if a future config enables them). LayerNorm names norm_mha/norm_ff match
the checkpoint.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class ConformerEncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module,
                 dropout_rate: float = 0.0, normalize_before: bool = True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        self.ff_scale = 1.0  # no macaron
        self.size = size
        self.normalize_before = normalize_before

    def __call__(self, x: mx.array, mask: mx.array, pos_emb: mx.array,
                 mask_pad: Optional[mx.array] = None,
                 att_cache: Optional[mx.array] = None,
                 cnn_cache: Optional[mx.array] = None
                 ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        # multi-headed self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb=pos_emb,
                                              cache=att_cache)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)

        # feed forward
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm_ff(x)

        return x, mask, new_att_cache, cnn_cache
