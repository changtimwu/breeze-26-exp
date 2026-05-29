"""Conditional flow matching (CFM) ODE solver.  [PORT STATUS: IMPLEMENTED — parity-tested]

MLX port of ConditionalCFM from ../BreezyVoice/cosyvoice/flow/flow_matching.py
(extends Matcha BASECFM). Only the inference path (forward + solve_euler) is
ported — fixed-step Euler ODE solve with classifier-free guidance
(inference_cfg_rate=0.7). The estimator is the UNet1D ConditionalDecoder.

No learnable parameters of its own (just the estimator); nothing to load here.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class ConditionalCFM(nn.Module):
    def __init__(self, in_channels: int, cfm_params, n_spks: int = 1,
                 spk_emb_dim: int = 64, estimator: nn.Module = None):
        super().__init__()
        self.sigma_min = getattr(cfm_params, "sigma_min", 1e-6)
        self.t_scheduler = cfm_params.t_scheduler
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator

    def __call__(self, mu, mask, n_timesteps, temperature: float = 1.0,
                 spks=None, cond=None, z: Optional[mx.array] = None):
        """Forward diffusion. mu/mask/cond are (B, C, T); spks (B, spk_dim).
        `z` lets a caller inject fixed noise (used for deterministic parity)."""
        if z is None:
            z = mx.random.normal(mu.shape) * temperature
        t_span = mx.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(t_span * 0.5 * math.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        sol = []
        for step in range(1, t_span.shape[0]):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            if self.inference_cfg_rate > 0:
                cfg = self.estimator(
                    x, mask, mx.zeros_like(mu), t,
                    mx.zeros_like(spks) if spks is not None else None,
                    mx.zeros_like(cond) if cond is not None else None)
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt
                           - self.inference_cfg_rate * cfg)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < t_span.shape[0] - 1:
                dt = t_span[step + 1] - t
        return sol[-1]
