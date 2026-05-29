"""Parity: MLX ConditionalDecoder (UNet1D flow estimator) vs BreezyVoice's torch
ConditionalDecoder. Copies torch weights via remap_decoder_weights and compares a
forward pass (b=1, no padding -> mask all ones, so the attn-mask convention
difference is moot).

Requires diffusers + conformer (for the torch source import).

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_decoder_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)
sys.path.insert(0, os.path.join(BV, "third_party", "Matcha-TTS"))

import torch  # noqa: E402
from cosyvoice.flow.decoder import ConditionalDecoder as TorchDecoder  # noqa: E402

from breezyvoice_mlx.flow.decoder import ConditionalDecoder, remap_decoder_weights  # noqa: E402


def test_decoder_parity():
    cfg = dict(in_channels=8, out_channels=4, channels=[16, 16], dropout=0.0,
               attention_head_dim=8, n_blocks=1, num_mid_blocks=1, num_heads=2,
               act_fn="gelu")
    tdec = TorchDecoder(**cfg).eval()
    mdec = ConditionalDecoder(in_channels=8, out_channels=4, channels=(16, 16),
                              dropout=0.0, attention_head_dim=8, n_blocks=1,
                              num_mid_blocks=1, num_heads=2, act_fn="gelu")
    mdec.load_weights(remap_decoder_weights(dict(tdec.state_dict())), strict=False)

    B, T = 1, 8
    rng = np.random.default_rng(0)
    # decoder input x has in_channels - mu_channels... here forward concatenates
    # x and mu internally, so x has (in_channels - mu) ch. The torch decoder packs
    # [x, mu] -> in_channels=8, so x=4ch (out_channels) and mu=4ch.
    x = rng.standard_normal((B, 4, T)).astype(np.float32)
    mu = rng.standard_normal((B, 4, T)).astype(np.float32)
    mask = np.ones((B, 1, T), dtype=np.float32)
    t = np.array([0.3], dtype=np.float32)

    with torch.no_grad():
        t_out = tdec(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(mu),
                     torch.from_numpy(t), spks=None, cond=None)
    m_out = mdec(mx.array(x), mx.array(mask), mx.array(mu), mx.array(t),
                 spks=None, cond=None)

    t_out, m_out = t_out.numpy(), np.array(m_out)
    print("shapes:", m_out.shape, t_out.shape)
    np.testing.assert_allclose(m_out, t_out, rtol=2e-2, atol=5e-3)
    print("OK: ConditionalDecoder (UNet1D) parity")


if __name__ == "__main__":
    test_decoder_parity()
    print("DECODER PARITY TEST PASSED")
