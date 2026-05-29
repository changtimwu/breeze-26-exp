"""Conformer text encoder.  [PORT STATUS: STUB — MEDIUM]

PyTorch source: ../BreezyVoice/cosyvoice/transformer/encoder.py (472 lines)
Classes: BaseEncoder, ConformerEncoder (used as TransformerLM.text_encoder
and MaskedDiffWithXvec.encoder).

Port notes:
  * Building blocks: Conv2d/linear subsampling, positional encoding
    (abs/rel/scaled), N x ConformerEncoderLayer (self-attn + conv module + FFN).
  * Skip the dynamic/static *chunking* + gradient-checkpointing paths — those
    are for streaming/training. Inference uses decoding_chunk_size=1,
    num_decoding_left_chunks=-1 (full-context). A non-chunked forward is enough.
  * output_size() must match so downstream affine layers line up.
"""

from __future__ import annotations

import mlx.nn as nn


class ConformerEncoder(nn.Module):  # TODO: implement
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("ConformerEncoder MLX port pending.")

    def output_size(self) -> int:  # pragma: no cover
        raise NotImplementedError
