"""Conformer encoder (non-streaming inference path).  [PORT STATUS: IMPLEMENTED]

MLX port of ConformerEncoder / BaseEncoder.forward from
../BreezyVoice/cosyvoice/transformer/encoder.py

Shared by the LLM text encoder and the flow encoder (both ConformerEncoder with
input_layer='linear', pos_enc='rel_pos_espnet', rel_selfattn, no cnn module, no
macaron). Only the full-utterance forward() is ported — the streaming
forward_chunk path is omitted (inference uses full context).

Chunk masking (matches wenet add_optional_chunk_mask):
  * static_chunk_size > 0  (text encoder = 1) -> causal chunk mask
  * static_chunk_size == 0 (flow encoder)     -> padding mask only
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import RelPositionMultiHeadedAttention
from .embedding import EspnetRelPositionalEncoding
from .encoder_layer import ConformerEncoderLayer, TransformerEncoderLayer
from .positionwise_feed_forward import PositionwiseFeedForward
from .subsampling import LegacyLinearNoSubsampling, LinearNoSubsampling


def make_pad_mask(lengths: mx.array, max_len: int = 0) -> mx.array:
    """(B,) lengths -> (B, max_len) bool, True at PADDED positions."""
    b = lengths.shape[0]
    max_len = max_len if max_len > 0 else int(mx.max(lengths).item())
    seq = mx.arange(max_len).reshape(1, max_len)
    return seq >= lengths.reshape(b, 1)


def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1
                          ) -> mx.array:
    """(size, size) bool. True = attend. Matches wenet subsequent_chunk_mask."""
    idx = mx.arange(size)
    # ending[i] = min(size, (i//chunk_size + 1) * chunk_size)
    ending = mx.minimum((idx // chunk_size + 1) * chunk_size, size).reshape(size, 1)
    if num_left_chunks < 0:
        start = mx.zeros((size, 1))
    else:
        start = mx.maximum((idx // chunk_size - num_left_chunks) * chunk_size, 0
                           ).reshape(size, 1)
    cols = mx.arange(size).reshape(1, size)
    return (cols >= start) & (cols < ending)


class ConformerEncoder(nn.Module):
    def __init__(self, input_size: int, output_size: int = 256,
                 attention_heads: int = 4, linear_units: int = 2048,
                 num_blocks: int = 6, dropout_rate: float = 0.0,
                 positional_dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 normalize_before: bool = True, static_chunk_size: int = 0,
                 key_bias: bool = True, **_ignored):
        super().__init__()
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size

        pos_enc = EspnetRelPositionalEncoding(output_size, positional_dropout_rate)
        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, pos_enc)
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.encoders = [
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(attention_heads, output_size,
                                                attention_dropout_rate, key_bias),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate,
                                        nn.SiLU()),
                dropout_rate, normalize_before,
            )
            for _ in range(num_blocks)
        ]

    def output_size(self) -> int:
        return self._output_size

    def __call__(self, xs: mx.array, xs_lens: mx.array,
                 decoding_chunk_size: int = 0, num_decoding_left_chunks: int = -1
                 ) -> Tuple[mx.array, mx.array]:
        T = xs.shape[1]
        masks = (~make_pad_mask(xs_lens, T))[:, None, :]   # (B, 1, T) keep=True
        xs, pos_emb, masks = self.embed(xs, masks)

        if self.static_chunk_size > 0:
            chunk = subsequent_chunk_mask(xs.shape[1], self.static_chunk_size,
                                          num_decoding_left_chunks)[None]  # (1,T,T)
            chunk_masks = masks & chunk                     # (B, T, T)
        else:
            chunk_masks = masks                             # (B, 1, T)

        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_chunk(self, xs: mx.array, offset: int = 0,
                      required_cache_size: int = -1,
                      att_cache: mx.array = None, cnn_cache: mx.array = None,
                      att_mask: mx.array = None):
        """KV-cached single-chunk forward (used by the AR decode loop).

        att_cache: (n_layers, head, cache_t, 2*d_k) or None on the first step.
        Returns (xs, new_att_cache). required_cache_size<0 keeps the full cache
        (BreezyVoice's LLM loop passes -1). EspnetRelPositionalEncoding ignores
        `offset`, so we don't thread it into position_encoding.
        """
        xs, _, _ = self.embed(xs, None, offset)            # input layer + xscale
        cache_t1 = 0 if att_cache is None else att_cache.shape[2]
        attention_key_size = cache_t1 + xs.shape[1]
        pos_emb = self.embed.position_encoding(attention_key_size)
        next_cache_start = 0 if required_cache_size < 0 \
            else max(attention_key_size - required_cache_size, 0)

        new_caches = []
        for i, layer in enumerate(self.encoders):
            layer_cache = None if att_cache is None else att_cache[i:i + 1]
            xs, _, new_att_cache, _ = layer(xs, att_mask, pos_emb,
                                            att_cache=layer_cache)
            new_caches.append(new_att_cache[:, :, next_cache_start:, :])
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, mx.concatenate(new_caches, axis=0)


class TransformerEncoder(nn.Module):
    """Transformer encoder — BreezyVoice's LLM backbone (self.llm).

    Same building blocks as ConformerEncoder but TransformerEncoderLayer
    (norm1/norm2), input_layer='linear_legacy' (extra ReLU), and ReLU FFN
    (activation_type='relu' default). static_chunk_size=1 -> causal.
    """

    def __init__(self, input_size: int, output_size: int = 256,
                 attention_heads: int = 4, linear_units: int = 2048,
                 num_blocks: int = 6, dropout_rate: float = 0.0,
                 positional_dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 normalize_before: bool = True, static_chunk_size: int = 0,
                 key_bias: bool = True, **_ignored):
        super().__init__()
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size

        pos_enc = EspnetRelPositionalEncoding(output_size, positional_dropout_rate)
        self.embed = LegacyLinearNoSubsampling(input_size, output_size,
                                               dropout_rate, pos_enc)
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.encoders = [
            TransformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(attention_heads, output_size,
                                                attention_dropout_rate, key_bias),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate,
                                        nn.ReLU()),
                dropout_rate, normalize_before,
            )
            for _ in range(num_blocks)
        ]

    def output_size(self) -> int:
        return self._output_size

    # full-utterance forward + KV-cached forward_chunk share ConformerEncoder's
    # implementations verbatim (same layer call signature); reuse by aliasing.
    __call__ = ConformerEncoder.__call__
    forward_chunk = ConformerEncoder.forward_chunk
