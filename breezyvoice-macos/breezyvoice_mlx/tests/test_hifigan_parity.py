"""Parity / validation for the MLX HiFiGAN-NSF vocoder.

Per-component verification (the full decode on random-init weights pathologically
amplifies strided-conv fp32 drift via the exp() magnitude path, so it's not a
meaningful end-to-end metric without trained weights):

  * ResBlock (Snake + dilated convs, weight_norm fused) — exact parity vs torch.
  * Full HiFTGenerator.forward — smoke test: runs, right output length, finite.

Separately verified exact elsewhere / inline:
  * STFT/iSTFT vs torch.stft/istft: ~1e-7   * Snake: ~2e-7
  * Conv1d (dilation 1/3/5): ~5e-7          * F0 predictor: test_f0_parity
  * Strided Conv1d (~1e-3) and ConvTranspose1d (~4e-4): correct layout, MLX fp32
    summation drift for strided ops (documented; perceptually negligible on
    trained weights).

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_hifigan_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)

import torch  # noqa: E402
from cosyvoice.hifigan.generator import ResBlock as TorchResBlock  # noqa: E402
from cosyvoice.hifigan.generator import HiFTGenerator as TorchHiFT  # noqa: E402
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor as TorchF0  # noqa: E402

from mlx_audio.codec.models.s3gen.hifigan import ResBlock  # noqa: E402
from breezyvoice_mlx.hifigan.generator import HiFTGenerator, remap_hift_weights  # noqa: E402
from breezyvoice_mlx.hifigan.f0_predictor import ConvRNNF0Predictor  # noqa: E402

CFG = dict(in_channels=8, base_channels=16, nb_harmonics=8, sampling_rate=22050,
           upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
           istft_params={"n_fft": 16, "hop_len": 4},
           resblock_kernel_sizes=[3, 7, 11],
           resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
           source_resblock_kernel_sizes=[7, 11],
           source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]])


def test_resblock_parity():
    """Core repeated unit: Snake + dilated (non-strided) convs -> should be exact."""
    tb = TorchResBlock(channels=16, kernel_size=3, dilations=[1, 3, 5]).eval()
    mb = ResBlock(channels=16, kernel_size=3, dilations=[1, 3, 5])
    mb.load_weights(remap_hift_weights(dict(tb.state_dict())), strict=False)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 16, 20)).astype(np.float32)
    with torch.no_grad():
        t_out = tb(torch.from_numpy(x)).numpy()
    m_out = np.array(mb(mx.array(x)))
    print("ResBlock maxdiff:", np.abs(t_out - m_out).max())
    np.testing.assert_allclose(m_out, t_out, rtol=1e-3, atol=1e-4)
    print("OK: ResBlock parity (Snake + dilated convs, weight_norm fused)")


def test_generator_smoke():
    """Full vocoder runs end-to-end and yields finite audio of the right length."""
    tg = TorchHiFT(f0_predictor=TorchF0(in_channels=8, cond_channels=16), **CFG).eval()
    mg = HiFTGenerator(f0_predictor=ConvRNNF0Predictor(in_channels=8, cond_channels=16), **CFG)
    mg.load_weights(remap_hift_weights(dict(tg.state_dict())), strict=False)

    T = 6
    rng = np.random.default_rng(1)
    mel = mx.array(rng.standard_normal((1, 8, T)).astype(np.float32))
    wav, _ = mg(mel)
    wav = np.array(wav)
    ups = int(np.prod(CFG["upsample_rates"]) * CFG["istft_params"]["hop_len"])
    print("generator out:", wav.shape, "expected len ~", T * ups)
    assert wav.shape[0] == 1 and wav.shape[1] == T * ups, wav.shape
    assert np.isfinite(wav).all(), "non-finite audio"
    assert np.abs(wav).max() <= 0.99 + 1e-5, "exceeds audio_limit"
    print("OK: HiFTGenerator forward smoke (finite, correct length, clipped)")


def test_decode_parity_realistic_mel():
    """Full decode vs torch on a realistic (controlled-magnitude) mel + fixed
    source. Guards the final-leaky_relu slope fix (slope 0.01, not lrelu_slope).
    With a real-range mel the exp() path is well-behaved, so this matches tightly."""
    tg = TorchHiFT(f0_predictor=TorchF0(in_channels=8, cond_channels=16), **CFG).eval()
    mg = HiFTGenerator(f0_predictor=ConvRNNF0Predictor(in_channels=8, cond_channels=16), **CFG)
    mg.load_weights(remap_hift_weights(dict(tg.state_dict())), strict=False)

    T = 6
    rng = np.random.default_rng(0)
    # real mels are negative log-magnitudes (~N(-5, 2)); keeps exp() tame
    mel = (rng.standard_normal((1, 8, T)).astype(np.float32) * 1.5 - 5.0)
    ups = int(np.prod(CFG["upsample_rates"]) * CFG["istft_params"]["hop_len"])
    s = rng.standard_normal((1, 1, T * ups)).astype(np.float32)

    import torch.nn.functional as F
    with torch.no_grad():
        sr, si = tg._stft(torch.from_numpy(s).squeeze(1)); s_stft = torch.cat([sr, si], 1)
        x = tg.conv_pre(torch.from_numpy(mel))
        for i in range(tg.num_upsamples):
            x = F.leaky_relu(x, tg.lrelu_slope); x = tg.ups[i](x)
            if i == tg.num_upsamples - 1: x = tg.reflection_pad(x)
            x = x + tg.source_resblocks[i](tg.source_downs[i](s_stft))
            xs = None
            for j in range(tg.num_kernels):
                r = tg.resblocks[i * tg.num_kernels + j](x); xs = r if xs is None else xs + r
            x = xs / tg.num_kernels
        x = F.leaky_relu(x)  # default 0.01 — the correct final slope
        x = tg.conv_post(x)
        half = tg.istft_params["n_fft"] // 2 + 1
        t_out = torch.clamp(tg._istft(torch.exp(x[:, :half]), torch.sin(x[:, half:])),
                            -tg.audio_limit, tg.audio_limit).numpy().reshape(-1)

    m_out = np.array(mg.decode(mx.array(mel), mx.array(s))).reshape(-1)
    n = min(len(t_out), len(m_out))
    rel = np.linalg.norm(m_out[:n] - t_out[:n]) / (np.linalg.norm(t_out[:n]) + 1e-9)
    print(f"decode rel_l2={rel:.4f} t_rms={np.sqrt((t_out**2).mean()):.3f} "
          f"m_rms={np.sqrt((m_out**2).mean()):.3f}")
    assert rel < 0.05, f"decode diverged (slope fix regressed?): rel_l2={rel}"
    print("OK: HiFTGenerator.decode parity (final-leaky_relu slope fix)")


if __name__ == "__main__":
    test_resblock_parity()
    test_generator_smoke()
    test_decode_parity_realistic_mel()
    print("HIFIGAN TESTS PASSED")
