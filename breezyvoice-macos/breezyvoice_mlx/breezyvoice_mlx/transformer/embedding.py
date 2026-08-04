"""Positional encoding.  [PORT STATUS: IMPLEMENTED]

MLX port of the EspnetRelPositionalEncoding from
../BreezyVoice/cosyvoice/transformer/embedding.py — the only pos-enc type
BreezyVoice uses (cosyvoice.yaml: pos_enc_layer_type='rel_pos_espnet').

The encoding table `pe` has length 2*max_len-1 (positive half flipped, then
negative half) to support the Transformer-XL rel-shift trick. It is a constant
(not learned), recomputed here at init — no weights to load.
"""

from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _build_pe(d_model: int, max_len: int) -> np.ndarray:
    """Replicate EspnetRelPositionalEncoding.extend_pe (numpy, init-time only)."""
    pe_positive = np.zeros((max_len, d_model), dtype=np.float32)
    pe_negative = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32)
                      * -(math.log(10000.0) / d_model))
    pe_positive[:, 0::2] = np.sin(position * div_term)
    pe_positive[:, 1::2] = np.cos(position * div_term)
    pe_negative[:, 0::2] = np.sin(-1 * position * div_term)
    pe_negative[:, 1::2] = np.cos(-1 * position * div_term)
    pe_positive = np.flip(pe_positive, axis=0)[None]      # (1, max_len, d)
    pe_negative = pe_negative[1:][None]                   # (1, max_len-1, d)
    return np.concatenate([pe_positive, pe_negative], axis=1)  # (1, 2*max_len-1, d)


class EspnetRelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        # Constant table; freeze so load_weights/quantization leave it alone.
        self._pe = mx.array(_build_pe(d_model, max_len))
        self.freeze(keys=["_pe"], recurse=False)

    def position_encoding(self, size: int) -> mx.array:
        """Centered slice of length 2*size-1 (matches torch source)."""
        center = self._pe.shape[1] // 2
        return self._pe[:, center - size + 1 : center + size]

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        x = x * self.xscale
        pos_emb = self.position_encoding(x.shape[1])
        return x, pos_emb
