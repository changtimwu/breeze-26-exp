"""Parity: MLX ConvRNNF0Predictor vs BreezyVoice torch (validates weight_norm
fusion + key remap on a real module).

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_f0_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)

import torch  # noqa: E402
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor as TorchF0  # noqa: E402

from breezyvoice_mlx.hifigan.f0_predictor import ConvRNNF0Predictor, remap_f0_weights  # noqa: E402


def test_f0_parity():
    C = 80
    tf0 = TorchF0(num_class=1, in_channels=C, cond_channels=64).eval()
    mf0 = ConvRNNF0Predictor(num_class=1, in_channels=C, cond_channels=64)
    mf0.load_weights(remap_f0_weights(dict(tf0.state_dict())), strict=False)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, C, 12)).astype(np.float32)
    with torch.no_grad():
        t_out = tf0(torch.from_numpy(x))
    m_out = mf0(mx.array(x))
    t_out, m_out = t_out.numpy(), np.array(m_out)
    print("shapes:", m_out.shape, t_out.shape)
    np.testing.assert_allclose(m_out, t_out, rtol=1e-2, atol=3e-3)
    print("OK: ConvRNNF0Predictor parity (weight_norm fused)")


if __name__ == "__main__":
    test_f0_parity()
    print("F0 PARITY TEST PASSED")
