"""HiFiGAN-NSF vocoder (mel -> waveform).  [PORT STATUS: STUB — HARD]

PyTorch source: ../BreezyVoice/cosyvoice/hifigan/generator.py (391 lines)
Class: HiFTGenerator (+ SineGen, SourceModuleHnNSF, ResBlock, Snake)

Porting notes / blockers:
  * weight_norm — ~16 uses. FUSE at conversion time (breezyvoice_mlx.nn.fuse_weight_norm),
    so the MLX modules below use plain mlx.nn.Conv1d. See nn/weight_norm.py.
  * STFT/iSTFT — generator._stft/_istft use torch.stft / torch.istft with a Hann
    window (n_fft from istft_params). MLX has mx.fft but no windowed STFT helper;
    implement framing + window + rfft / overlap-add manually. See nn/stft.py (TODO).
  * Snake activation — x + (1/a) * sin(a*x)^2 ; trivial in MLX.
  * Conv1d layout — PyTorch (out,in,k) -> MLX (out,k,in); handled by converter.
  * NSF source: SineGen F0->harmonics, f0_upsamp (nn.Upsample) -> use mx repeat.

forward(mel) -> waveform ; inference(mel) is just forward under no-grad.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class HiFTGenerator(nn.Module):  # TODO: implement
    """MLX port of HiFTGenerator. NOT YET IMPLEMENTED."""

    def __init__(self, **config):
        super().__init__()
        raise NotImplementedError(
            "HiFTGenerator MLX port pending — see docstring. "
            "Port order: ResBlock + Snake -> NSF source -> STFT/iSTFT -> assemble."
        )

    def __call__(self, mel: mx.array) -> mx.array:  # pragma: no cover
        raise NotImplementedError

    def inference(self, mel: mx.array) -> mx.array:  # pragma: no cover
        return self(mel)
