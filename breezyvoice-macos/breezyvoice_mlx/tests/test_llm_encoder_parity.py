"""Parity: MLX TransformerEncoder (BreezyVoice's LLM backbone) vs PyTorch,
including the KV-cached forward_chunk path the autoregressive decode loop uses.

Config mirrors cosyvoice.yaml `llm`: input_layer='linear_legacy', rel_pos_espnet,
rel_selfattn, static_chunk_size=1, ReLU FFN.

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_llm_encoder_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)

import torch  # noqa: E402
from cosyvoice.transformer.encoder import TransformerEncoder as TorchTransformerEncoder  # noqa: E402

from breezyvoice_mlx.transformer.encoder import TransformerEncoder  # noqa: E402


def _build_pair():
    cfg = dict(input_size=32, output_size=32, attention_heads=4, linear_units=64,
               num_blocks=3, dropout_rate=0.0, positional_dropout_rate=0.0,
               attention_dropout_rate=0.0, normalize_before=True,
               static_chunk_size=1)
    tmod = TorchTransformerEncoder(
        input_layer="linear_legacy", pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn", **cfg).eval()
    mmod = TransformerEncoder(**cfg)
    items = {k: mx.array(v.detach().numpy()) for k, v in tmod.state_dict().items()}
    mmod.load_weights(list(items.items()), strict=False)
    return tmod, mmod


def test_full_forward_parity():
    tmod, mmod = _build_pair()
    B, T, D = 2, 8, 32
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    lens = np.array([T, T - 2], dtype=np.int32)
    with torch.no_grad():
        tout, _ = tmod(torch.from_numpy(x), torch.from_numpy(lens))
    mout, _ = mmod(mx.array(x), mx.array(lens))
    tout, mout = tout.numpy(), np.array(mout)
    for b in range(B):
        np.testing.assert_allclose(mout[b, : lens[b]], tout[b, : lens[b]],
                                   rtol=1e-2, atol=3e-3)
    print("OK: TransformerEncoder full forward parity")


def test_forward_chunk_ar_parity():
    """Simulate the AR loop: prefix step then a single-token decode step."""
    tmod, mmod = _build_pair()
    D = 32
    rng = np.random.default_rng(1)
    P = 6
    prefix = rng.standard_normal((1, P, D)).astype(np.float32)

    def tril(n):
        return torch.tril(torch.ones((1, n, n))).bool()

    # --- step 0: full prefix ---
    with torch.no_grad():
        t_y0, t_cache, _ = tmod.forward_chunk(
            torch.from_numpy(prefix), offset=0, required_cache_size=-1,
            att_cache=torch.zeros(0, 0, 0, 0), cnn_cache=torch.zeros(0, 0, 0, 0),
            att_mask=tril(P))
    m_y0, m_cache = mmod.forward_chunk(
        mx.array(prefix), offset=0, required_cache_size=-1, att_cache=None,
        att_mask=mx.array(np.tril(np.ones((1, P, P), dtype=np.float32))))
    np.testing.assert_allclose(np.array(m_y0), t_y0.numpy(), rtol=1e-2, atol=3e-3)
    assert tuple(t_cache.shape) == tuple(m_cache.shape), (t_cache.shape, m_cache.shape)
    print(f"OK: forward_chunk step0 parity (cache {tuple(m_cache.shape)})")

    # --- step 1: one new token, using the accumulated cache ---
    tok = rng.standard_normal((1, 1, D)).astype(np.float32)
    with torch.no_grad():
        t_y1, _, _ = tmod.forward_chunk(
            torch.from_numpy(tok), offset=0, required_cache_size=-1,
            att_cache=t_cache, cnn_cache=torch.zeros(0, 0, 0, 0), att_mask=tril(1))
    m_y1, _ = mmod.forward_chunk(
        mx.array(tok), offset=0, required_cache_size=-1, att_cache=m_cache,
        att_mask=mx.array(np.tril(np.ones((1, 1, 1), dtype=np.float32))))
    np.testing.assert_allclose(np.array(m_y1), t_y1.numpy(), rtol=1e-2, atol=3e-3)
    print("OK: forward_chunk step1 (cached decode) parity")


if __name__ == "__main__":
    test_full_forward_parity()
    test_forward_chunk_ar_parity()
    print("ALL LLM-ENCODER PARITY TESTS PASSED")
