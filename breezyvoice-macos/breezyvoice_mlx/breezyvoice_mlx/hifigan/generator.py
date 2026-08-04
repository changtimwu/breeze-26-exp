"""HiFiGAN-NSF vocoder (mel -> waveform).  [PORT STATUS: IMPLEMENTED — decode parity-tested]

BreezyVoice's vocoder is HiFTGenerator (HiFiGAN-NSF + ISTFTNet:
../BreezyVoice/cosyvoice/hifigan/generator.py). mlx-audio-plus ships a faithful
MLX HiFTGenerator with identical default config (22050 Hz, upsample [8,8],
istft n_fft=16/hop=4, NSF source, STFT/iSTFT, Snake-ResBlocks). We reuse it and
the matching ConvRNNF0Predictor, and provide remap_hift_weights to load a
BreezyVoice PyTorch checkpoint: fuse weight_norm (conv_pre/conv_post/ups/resblocks/
source_resblocks), fix Conv1d/ConvTranspose1d layout, and collapse the F0
predictor's Sequential index gaps.

The deterministic decode path (conv stack + STFT/iSTFT + Snake ResBlocks) is
parity-tested vs the torch source (tests/test_hifigan_parity.py). The NSF source
(SineGen) is inherently stochastic (random phase + noise), so the full forward is
not bit-reproducible across frameworks by design.
"""

from __future__ import annotations

import re

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.s3gen.hifigan import HiFTGenerator as _MlxHiFTGenerator

from ..nn.weight_norm import fuse_weight_norm
from .f0_predictor import ConvRNNF0Predictor  # re-export


class HiFTGenerator(_MlxHiFTGenerator):
    """BreezyVoice HiFT vocoder.

    Overrides mlx-audio-plus's decode() to fix a bug: the FINAL leaky_relu before
    conv_post must use slope 0.01 (torch HiFTGenerator uses bare `F.leaky_relu(x)`,
    default 0.01), not lrelu_slope (0.1). The upstream code used 0.1 there, which —
    via the exp() magnitude path — produced ~6x too-loud, heavily-clipped audio.
    With slope 0.01 the decode matches the torch source to ~4e-3 on real weights.
    """

    def decode(self, x: mx.array, s: mx.array) -> mx.array:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = mx.concatenate([s_stft_real, s_stft_imag], axis=1)

        x = mx.swapaxes(self.conv_pre(mx.swapaxes(x, 1, 2)), 1, 2)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = mx.swapaxes(self.ups[i](mx.swapaxes(x, 1, 2)), 1, 2)
            if i == self.num_upsamples - 1:
                x = mx.concatenate([x[:, :, 1:2], x], axis=2)
            si = mx.swapaxes(self.source_downs[i](mx.swapaxes(s_stft, 1, 2)), 1, 2)
            si = self.source_resblocks[i](si)
            x = x + si
            start = i * self.num_kernels
            x = mx.mean(mx.stack([self.resblocks[start + j](x)
                                  for j in range(self.num_kernels)], axis=0), axis=0)

        x = nn.leaky_relu(x, negative_slope=0.01)  # FINAL: torch F.leaky_relu default
        x = mx.swapaxes(self.conv_post(mx.swapaxes(x, 1, 2)), 1, 2)
        half = self.istft_params["n_fft"] // 2 + 1
        magnitude = mx.exp(x[:, :half, :])
        phase = mx.sin(x[:, half:, :])
        x = self._istft(magnitude, phase)
        return mx.clip(x, -self.audio_limit, self.audio_limit)


def _is_conv_transpose(key: str) -> bool:
    return re.search(r"(^|\.)ups\.\d+\.weight$", key) is not None


def remap_hift_weights(torch_state: dict) -> list:
    """BreezyVoice torch HiFTGenerator state_dict -> MLX (weight_norm fuse +
    conv layout + F0 condnet index collapse)."""
    g, v, rest = {}, {}, []
    for k, t in torch_state.items():
        a = mx.array(np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                                dtype=np.float32))
        if k.endswith(".weight_g"):
            g[k[:-9]] = a
        elif k.endswith(".weight_v"):
            v[k[:-9]] = a
        else:
            rest.append((k, a))
    for base in g:
        rest.append((base + ".weight", fuse_weight_norm(g[base], v[base], dim=0)))

    out = []
    for k, a in rest:
        # collapse f0_predictor.condnet.{2i} -> .{i}
        m = re.match(r"(f0_predictor\.condnet\.)(\d+)\.(weight|bias)$", k)
        if m:
            k = f"{m.group(1)}{int(m.group(2)) // 2}.{m.group(3)}"
        if k.endswith(".weight") and a.ndim == 3:
            if _is_conv_transpose(k):   # ConvTranspose1d (in,out,k)->(out,k,in)
                a = mx.transpose(a, (1, 2, 0))
            else:                        # Conv1d (out,in,k)->(out,k,in)
                a = mx.transpose(a, (0, 2, 1))
        out.append((k, a))
    return out


__all__ = ["HiFTGenerator", "ConvRNNF0Predictor", "remap_hift_weights"]
