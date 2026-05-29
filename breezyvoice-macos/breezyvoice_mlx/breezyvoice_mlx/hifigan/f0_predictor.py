"""F0 predictor (mel -> F0 contour).  [PORT STATUS: IMPLEMENTED — parity-tested]

MLX port of ConvRNNF0Predictor from
../BreezyVoice/cosyvoice/hifigan/f0_predictor.py: 5x weight_norm(Conv1d)+ELU then
a Linear classifier, abs() output. mlx-audio-plus ships a faithful MLX
ConvRNNF0Predictor (matching config); we reuse it and provide remap_f0_weights to
fuse weight_norm + fix conv layout + collapse the nn.Sequential index gaps
(torch condnet.0/2/4/6/8 -> MLX list condnet.0..4).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlx_audio.codec.models.s3gen.f0_predictor import ConvRNNF0Predictor

from ..nn.weight_norm import fuse_weight_norm


def remap_f0_weights(torch_state: dict, prefix: str = "") -> list:
    """BreezyVoice torch ConvRNNF0Predictor state_dict -> MLX (key+layout)."""
    g, v, out = {}, {}, []
    for k, t in torch_state.items():
        a = mx.array(np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                                dtype=np.float32))
        if k.endswith(".weight_g"):
            g[k[:-9]] = a
        elif k.endswith(".weight_v"):
            v[k[:-9]] = a
        else:
            out.append((k, a))
    # fuse weight_norm pairs -> .weight
    for base in g:
        out.append((base + ".weight", fuse_weight_norm(g[base], v[base], dim=0)))
    # rename condnet.{2i} -> condnet.{i}, transpose conv weights
    remapped = []
    for k, a in out:
        m = k
        import re
        mm = re.match(r"condnet\.(\d+)\.(weight|bias)$", k)
        if mm:
            m = f"condnet.{int(mm.group(1)) // 2}.{mm.group(2)}"
        if m.endswith(".weight") and a.ndim == 3:     # Conv1d (out,in,k)->(out,k,in)
            a = mx.transpose(a, (0, 2, 1))
        remapped.append((prefix + m, a))
    return remapped


__all__ = ["ConvRNNF0Predictor", "remap_f0_weights"]
