"""Numerical parity: MLX attention must match BreezyVoice's PyTorch attention.

Loads identical weights into the torch source module and our MLX port, feeds the
same input, and asserts the outputs match. Covers both the vanilla and the
relative-position attention (the latter is what BreezyVoice actually uses).

Run (breezyvoice-macos venv has torch + mlx):
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_attention_parity.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import mlx.core as mx

from breezyvoice_mlx.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)

# Import the PyTorch source module directly by path (avoid importing the whole
# cosyvoice package, which pulls heavy deps).
BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
_src = os.path.join(BV, "cosyvoice", "transformer", "attention.py")
_spec = importlib.util.spec_from_file_location("bv_attention", _src)
bv = importlib.util.module_from_spec(_spec)
sys.modules["bv_attention"] = bv
_spec.loader.exec_module(bv)

import torch  # noqa: E402


def _sync_linear(mlx_mod, name, torch_lin):
    w = {f"{name}.weight": mx.array(torch_lin.weight.detach().numpy())}
    if torch_lin.bias is not None:
        w[f"{name}.bias"] = mx.array(torch_lin.bias.detach().numpy())
    return w


def test_mha_parity():
    n_head, n_feat, B, T = 4, 32, 2, 7
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, T, n_feat)).astype(np.float32)
    # mask: keep all but last 2 of key positions in batch elem 1
    mask_np = np.ones((B, 1, T), dtype=np.float32)
    mask_np[1, :, -2:] = 0

    tmod = bv.MultiHeadedAttention(n_head, n_feat, 0.0).eval()
    mmod = MultiHeadedAttention(n_head, n_feat, 0.0)

    flat = {}
    for nm in ["linear_q", "linear_k", "linear_v", "linear_out"]:
        flat.update(_sync_linear(mmod, nm, getattr(tmod, nm)))
    mmod.load_weights(list(flat.items()))

    with torch.no_grad():
        tx = torch.from_numpy(x)
        tmask = torch.from_numpy(mask_np).bool()
        tout, _ = tmod(tx, tx, tx, tmask)
    mout, _ = mmod(mx.array(x), mx.array(x), mx.array(x), mx.array(mask_np))

    np.testing.assert_allclose(np.array(mout), tout.numpy(), rtol=1e-2, atol=3e-3)
    print("OK: MultiHeadedAttention parity")


def test_rel_pos_mha_parity():
    n_head, n_feat, B, T = 4, 32, 2, 7
    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, T, n_feat)).astype(np.float32)
    # espnet rel-pos embedding length is 2*T-1
    pos = rng.standard_normal((1, 2 * T - 1, n_feat)).astype(np.float32)
    mask_np = np.ones((B, 1, T), dtype=np.float32)
    mask_np[0, :, -3:] = 0

    tmod = bv.RelPositionMultiHeadedAttention(n_head, n_feat, 0.0).eval()
    mmod = RelPositionMultiHeadedAttention(n_head, n_feat, 0.0)

    flat = {}
    for nm in ["linear_q", "linear_k", "linear_v", "linear_out"]:
        flat.update(_sync_linear(mmod, nm, getattr(tmod, nm)))
    flat["linear_pos.weight"] = mx.array(tmod.linear_pos.weight.detach().numpy())
    flat["pos_bias_u"] = mx.array(tmod.pos_bias_u.detach().numpy())
    flat["pos_bias_v"] = mx.array(tmod.pos_bias_v.detach().numpy())
    mmod.load_weights(list(flat.items()))

    with torch.no_grad():
        tx = torch.from_numpy(x)
        tmask = torch.from_numpy(mask_np).bool()
        tout, _ = tmod(tx, tx, tx, tmask, torch.from_numpy(pos))
    mout, _ = mmod(mx.array(x), mx.array(x), mx.array(x),
                   mx.array(mask_np), mx.array(pos))

    np.testing.assert_allclose(np.array(mout), tout.numpy(), rtol=1e-2, atol=3e-3)
    print("OK: RelPositionMultiHeadedAttention parity")


if __name__ == "__main__":
    test_mha_parity()
    test_rel_pos_mha_parity()
    print("ALL ATTENTION PARITY TESTS PASSED")
