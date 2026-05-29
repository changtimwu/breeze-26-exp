"""End-to-end parity: MLX ConformerEncoder vs BreezyVoice's PyTorch encoder.

Builds the torch ConformerEncoder (BreezyVoice's text-encoder config: rel_pos_espnet
+ rel_selfattn, no cnn module, no macaron, static_chunk_size=1 -> causal), copies
its weights into the MLX port, and asserts equal outputs on a padded batch. If the
full forward matches, every sub-module (pos-enc, subsampling, attention, FFN,
layernorms, causal chunk mask) is correct.

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_encoder_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)

import torch  # noqa: E402
from cosyvoice.transformer.encoder import ConformerEncoder as TorchConformerEncoder  # noqa: E402

from breezyvoice_mlx.transformer.encoder import ConformerEncoder  # noqa: E402


def _check(static_chunk_size: int, label: str):
    cfg = dict(input_size=16, output_size=32, attention_heads=4, linear_units=64,
               num_blocks=2, dropout_rate=0.0, positional_dropout_rate=0.0,
               attention_dropout_rate=0.0, normalize_before=True,
               static_chunk_size=static_chunk_size)

    tmod = TorchConformerEncoder(
        input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn", use_cnn_module=False,
        macaron_style=False, **cfg).eval()

    mmod = ConformerEncoder(**cfg)

    # Copy torch params -> mlx (names align; Linear/LayerNorm share layout).
    torch_items = {k: mx.array(v.detach().numpy()) for k, v in tmod.state_dict().items()}
    model_keys = {k for k, _ in mmod.parameters_flat()} if hasattr(mmod, "parameters_flat") else None
    mmod.load_weights(list(torch_items.items()), strict=False)

    # Strictness check: torch keys must cover every model param except the pe constant.
    def _flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flat(v, key + "."))
            elif isinstance(v, list):
                for i, e in enumerate(v):
                    if isinstance(e, dict):
                        out.update(_flat(e, f"{key}.{i}."))
            else:
                out[key] = v
        return out

    mlx_keys = set(_flat(mmod.parameters()).keys())
    missing = mlx_keys - set(torch_items) - {"embed.pos_enc._pe"}
    assert not missing, f"[{label}] model params not loaded from torch: {missing}"

    B, T, D = 2, 9, 16
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    lens = np.array([T, T - 3], dtype=np.int32)  # second item padded by 3

    with torch.no_grad():
        tout, tmask = tmod(torch.from_numpy(x), torch.from_numpy(lens))
    mout, mmask = mmod(mx.array(x), mx.array(lens))

    # Compare only valid (non-padded) positions of each batch element.
    tout = tout.numpy()
    mout = np.array(mout)
    for b in range(B):
        n = lens[b]
        np.testing.assert_allclose(mout[b, :n], tout[b, :n], rtol=1e-2, atol=3e-3)
    print(f"OK: ConformerEncoder parity [{label}]  out={mout.shape}")


def test_encoder_parity_causal():
    _check(static_chunk_size=1, label="text-encoder (causal chunk)")


def test_encoder_parity_full():
    _check(static_chunk_size=0, label="flow-encoder (full attention)")


if __name__ == "__main__":
    test_encoder_parity_causal()
    test_encoder_parity_full()
    print("ALL ENCODER PARITY TESTS PASSED")
