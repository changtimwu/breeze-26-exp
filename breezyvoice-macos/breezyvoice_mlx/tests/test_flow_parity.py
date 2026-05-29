"""Parity: MLX ConditionalCFM (Euler+CFG) and InterpolateRegulator vs PyTorch.

CFM: shared arithmetic estimator (no weights) + injected fixed noise -> the Euler
solve + classifier-free guidance must match exactly. Regulator: copy torch weights
(conv layout transposed) and compare on a fixed input.

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_flow_parity.py
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)
sys.path.insert(0, os.path.join(BV, "third_party", "Matcha-TTS"))

import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from cosyvoice.flow.length_regulator import InterpolateRegulator as TorchReg  # noqa: E402

from breezyvoice_mlx.flow.flow_matching import ConditionalCFM  # noqa: E402
from breezyvoice_mlx.flow.length_regulator import InterpolateRegulator  # noqa: E402

CFM_PARAMS = DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                         "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7,
                         "reg_loss_type": "l1"})


def test_cfm_euler_cfg_parity():
    C, T, n_steps = 8, 5, 10

    # Shared estimator formula (no params): -x + mu + spks_broadcast + t
    def t_est(x, mask, mu, t, spks, cond):
        s = spks.unsqueeze(-1) if spks is not None else 0.0
        return -0.3 * x + 0.2 * mu + 0.1 * s + 0.05 * t
    def m_est(x, mask, mu, t, spks, cond):
        s = mx.expand_dims(spks, -1) if spks is not None else 0.0
        return -0.3 * x + 0.2 * mu + 0.1 * s + 0.05 * t

    mcfm = ConditionalCFM(in_channels=C, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=C)
    mcfm.estimator = m_est
    cfg_rate = CFM_PARAMS.inference_cfg_rate

    rng = np.random.default_rng(0)
    mu = rng.standard_normal((1, C, T)).astype(np.float32)
    mask = np.ones((1, 1, T), dtype=np.float32)
    spks = rng.standard_normal((1, C)).astype(np.float32)
    z = rng.standard_normal((1, C, T)).astype(np.float32)

    # Reference: replicate cosyvoice/flow/flow_matching.py solve_euler in numpy
    # (the torch source can't be imported without conformer/diffusers).
    def np_est(x, mu_, spks_):
        s = spks_[..., None] if spks_ is not None else 0.0
        return -0.3 * x + 0.2 * mu_ + 0.1 * s + 0.05  # t added below per-step
    span = 1 - np.cos(np.linspace(0, 1, n_steps + 1, dtype=np.float32) * 0.5 * np.pi)
    x = z.copy(); t = span[0]; dt = span[1] - span[0]
    for step in range(1, len(span)):
        d = -0.3 * x + 0.2 * mu + 0.1 * spks[..., None] + 0.05 * t
        dc = -0.3 * x + 0.2 * np.zeros_like(mu) + 0.1 * np.zeros_like(spks)[..., None] + 0.05 * t
        d = (1.0 + cfg_rate) * d - cfg_rate * dc
        x = x + dt * d; t = t + dt
        if step < len(span) - 1:
            dt = span[step + 1] - t
    ref = x

    m_span = 1 - mx.cos(mx.linspace(0, 1, n_steps + 1) * 0.5 * np.pi)
    m_out = mcfm.solve_euler(mx.array(z), t_span=m_span, mu=mx.array(mu),
                             mask=mx.array(mask), spks=mx.array(spks), cond=None)
    np.testing.assert_allclose(np.array(m_out), ref, rtol=1e-3, atol=1e-4)
    print("OK: ConditionalCFM Euler+CFG parity")


def _remap_conv(items: dict) -> list:
    out = []
    for k, v in items.items():
        a = mx.array(v.detach().numpy())
        if k.endswith(".weight") and a.ndim == 3:   # Conv1d (out,in,k)->(out,k,in)
            a = mx.transpose(a, (0, 2, 1))
        out.append((k, a))
    return out


def test_length_regulator_parity():
    C, T = 8, 6
    treg = TorchReg(channels=C, sampling_ratios=[1, 1], out_channels=C, groups=1).eval()
    mreg = InterpolateRegulator(channels=C, sampling_ratios=[1, 1], out_channels=C, groups=1)
    mreg.load_weights(_remap_conv(dict(treg.state_dict())), strict=False)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, T, C)).astype(np.float32)
    ylens = np.array([10], dtype=np.int32)
    with torch.no_grad():
        t_out, _ = treg(torch.from_numpy(x), torch.from_numpy(ylens))
    m_out, _ = mreg(mx.array(x), mx.array(ylens))
    np.testing.assert_allclose(np.array(m_out), t_out.numpy(), rtol=1e-2, atol=3e-3)
    print(f"OK: InterpolateRegulator parity (out {tuple(t_out.shape)})")


if __name__ == "__main__":
    test_cfm_euler_cfg_parity()
    test_length_regulator_parity()
    print("ALL FLOW (CFM + REGULATOR) PARITY TESTS PASSED")
