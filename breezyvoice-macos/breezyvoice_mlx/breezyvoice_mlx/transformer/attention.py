"""Multi-head attention (shared by Conformer encoder + transformer decoder).
[PORT STATUS: STUB — EASY: pure matmul/softmax, no custom kernels]

PyTorch source: ../BreezyVoice/cosyvoice/transformer/attention.py (326 lines)
Classes: MultiHeadedAttention, RelPositionMultiHeadedAttention

Port notes:
  * Use mx.fast.scaled_dot_product_attention for the vanilla path.
  * RelPositionMultiHeadedAttention needs the rel-shift trick (matrix_bd) —
    port the index/pad-shift exactly; it's the one fiddly bit.
  * Supports incremental decoding via (k,v) cache — return updated cache like
    the PyTorch forward(..., cache) signature so llm.py's AR loop can thread it.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MultiHeadedAttention(nn.Module):  # TODO: implement
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0):
        super().__init__()
        raise NotImplementedError("MultiHeadedAttention MLX port pending.")


class RelPositionMultiHeadedAttention(MultiHeadedAttention):  # TODO: implement
    pass
