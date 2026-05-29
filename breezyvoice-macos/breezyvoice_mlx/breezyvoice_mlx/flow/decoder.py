"""UNet1D flow-matching decoder (the CFM estimator).  [PORT STATUS: STUB — MEDIUM]

PyTorch source: ../BreezyVoice/cosyvoice/flow/decoder.py (222 lines)
Class: ConditionalDecoder (UNet1D: down/mid/up ResNet1D + Transformer blocks)

Building blocks come from third_party/Matcha-TTS:
  matcha/models/components/decoder.py    -> SinusoidalPosEmb, Block1D, ResnetBlock1D,
                                            Downsample1D, Upsample1D, TimestepEmbedding
  matcha/models/components/transformer.py-> BasicTransformerBlock

NOTE: this is BreezyVoice's CosyVoice2 UNet1D estimator, distinct from
CosyVoice3's DiT. mlx-audio-plus's DiT is NOT a drop-in — port the UNet1D here.
einops pack/rearrange -> mx.reshape / mx.transpose. Conv1d layout fix via converter.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ConditionalDecoder(nn.Module):  # TODO: implement
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("ConditionalDecoder (UNet1D) MLX port pending.")

    def __call__(self, x, mask, mu, t, spks=None, cond=None) -> mx.array:  # pragma: no cover
        raise NotImplementedError
