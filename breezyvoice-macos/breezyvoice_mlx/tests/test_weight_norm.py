"""Parity test: our MLX weight_norm fusion must match PyTorch's remove_weight_norm.

Run (in the breezyvoice-macos venv, which has torch + mlx):
    python -m pytest tests/test_weight_norm.py -q
or standalone:
    python tests/test_weight_norm.py
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from breezyvoice_mlx.nn.weight_norm import fuse_weight_norm


def _torch_reference(v_np, g_np, dim=0):
    import torch
    from torch.nn.utils import weight_norm, remove_weight_norm

    # Build a Conv1d, install weight_g/weight_v, then remove_weight_norm and read .weight.
    out_c, in_c, k = v_np.shape
    conv = torch.nn.Conv1d(in_c, out_c, k, bias=False)
    wn = weight_norm(conv, name="weight", dim=dim)
    with torch.no_grad():
        wn.weight_v.copy_(torch.from_numpy(v_np))
        wn.weight_g.copy_(torch.from_numpy(g_np))
    remove_weight_norm(wn)
    return wn.weight.detach().numpy()


def test_conv1d_weight_norm_matches_torch():
    rng = np.random.default_rng(0)
    out_c, in_c, k = 16, 8, 3
    v_np = rng.standard_normal((out_c, in_c, k)).astype(np.float32)
    # PyTorch weight_g for dim=0 has shape (out_c, 1, 1)
    g_np = rng.standard_normal((out_c, 1, 1)).astype(np.float32)

    ours = np.array(fuse_weight_norm(mx.array(g_np), mx.array(v_np), dim=0))

    try:
        ref = _torch_reference(v_np, g_np, dim=0)
    except ImportError:
        # No torch available — fall back to a hand-computed reference.
        norm = np.sqrt((v_np ** 2).sum(axis=(1, 2), keepdims=True))
        ref = g_np * v_np / norm

    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_conv1d_weight_norm_matches_torch()
    print("OK: MLX weight_norm fusion matches reference.")
