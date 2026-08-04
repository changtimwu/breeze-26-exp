"""Multi-head attention (shared by Conformer encoder + transformer LM).
[PORT STATUS: IMPLEMENTED — parity-tested vs PyTorch]

MLX port of ../BreezyVoice/cosyvoice/transformer/attention.py
Classes: MultiHeadedAttention, RelPositionMultiHeadedAttention

BreezyVoice's config (cosyvoice.yaml) uses selfattention_layer_type='rel_selfattn'
+ pos_enc_layer_type='rel_pos_espnet', so RelPositionMultiHeadedAttention is the
one actually exercised. The vanilla MultiHeadedAttention is the base class.

Weight names match the PyTorch module exactly (linear_q/k/v/out.{weight,bias},
linear_pos.weight, pos_bias_u, pos_bias_v) so converted checkpoints load directly.
MLX nn.Linear uses the same (out, in) weight layout as torch.nn.Linear — no
transpose needed for these (only conv layers need layout fixes).

Masking convention (matches torch source): `mask` is 1/True for KEEP, 0/False for
PAD; padded positions get -inf before softmax then 0 after. A None or width-0
mask means "no masking".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0,
                 key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout_rate = dropout_rate

    def forward_qkv(self, query: mx.array, key: mx.array, value: mx.array
                    ) -> Tuple[mx.array, mx.array, mx.array]:
        n_batch = query.shape[0]
        q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_k)
        q = mx.transpose(q, (0, 2, 1, 3))  # (B, h, T1, d_k)
        k = mx.transpose(k, (0, 2, 1, 3))  # (B, h, T2, d_k)
        v = mx.transpose(v, (0, 2, 1, 3))  # (B, h, T2, d_k)
        return q, k, v

    def forward_attention(self, value: mx.array, scores: mx.array,
                          mask: Optional[mx.array] = None) -> mx.array:
        n_batch = value.shape[0]
        if mask is not None and mask.shape[2] > 0:
            mask = mx.expand_dims(mask, 1)              # (B, 1, *, T2)
            mask = mask[:, :, :, : scores.shape[-1]]
            keep = mask != 0
            scores = mx.where(keep, scores, -float("inf"))
            attn = mx.softmax(scores, axis=-1)
            attn = mx.where(keep, attn, 0.0)
        else:
            attn = mx.softmax(scores, axis=-1)
        x = attn @ value                                 # (B, h, T1, d_k)
        x = mx.transpose(x, (0, 2, 1, 3))                # (B, T1, h, d_k)
        x = x.reshape(n_batch, -1, self.h * self.d_k)    # (B, T1, d_model)
        return self.linear_out(x)

    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 mask: Optional[mx.array] = None, pos_emb: Optional[mx.array] = None,
                 cache: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        q, k, v = self.forward_qkv(query, key, value)
        if cache is not None and cache.shape[0] > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)
        new_cache = mx.concatenate([k, v], axis=-1)
        scores = (q @ mx.swapaxes(k, -2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-head attention with relative positional encoding (Transformer-XL)."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0,
                 key_bias: bool = True):
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # Loaded from checkpoint; init values here are placeholders.
        self.pos_bias_u = mx.zeros((self.h, self.d_k))
        self.pos_bias_v = mx.zeros((self.h, self.d_k))

    def rel_shift(self, x: mx.array) -> mx.array:
        """(B, head, T1, 2*T1-1) -> (B, head, T1, T1). Matches torch source."""
        b, h, t1, n = x.shape
        zero_pad = mx.zeros((b, h, t1, 1), dtype=x.dtype)
        x_padded = mx.concatenate([zero_pad, x], axis=-1)
        x_padded = x_padded.reshape(b, h, n + 1, t1)
        x = x_padded[:, :, 1:].reshape(b, h, t1, n)[:, :, :, : n // 2 + 1]
        return x

    def __call__(self, query: mx.array, key: mx.array, value: mx.array,
                 mask: Optional[mx.array] = None, pos_emb: Optional[mx.array] = None,
                 cache: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        q, k, v = self.forward_qkv(query, key, value)
        q = mx.transpose(q, (0, 2, 1, 3))                # (B, T1, h, d_k)

        if cache is not None and cache.shape[0] > 0:
            key_cache, value_cache = mx.split(cache, 2, axis=-1)
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)
        new_cache = mx.concatenate([k, v], axis=-1)

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).reshape(n_batch_pos, -1, self.h, self.d_k)
        p = mx.transpose(p, (0, 2, 1, 3))                # (B, h, T1, d_k)

        q_with_bias_u = mx.transpose(q + self.pos_bias_u, (0, 2, 1, 3))  # (B,h,T1,d_k)
        q_with_bias_v = mx.transpose(q + self.pos_bias_v, (0, 2, 1, 3))  # (B,h,T1,d_k)

        matrix_ac = q_with_bias_u @ mx.swapaxes(k, -2, -1)
        matrix_bd = q_with_bias_v @ mx.swapaxes(p, -2, -1)
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache
