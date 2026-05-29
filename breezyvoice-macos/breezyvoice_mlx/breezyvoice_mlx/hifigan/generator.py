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

from mlx_audio.codec.models.s3gen.hifigan import HiFTGenerator

from ..nn.weight_norm import fuse_weight_norm
from .f0_predictor import ConvRNNF0Predictor  # re-export


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
