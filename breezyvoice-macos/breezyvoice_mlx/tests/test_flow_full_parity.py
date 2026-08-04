"""End-to-end parity: MLX MaskedDiffWithXvec (token -> mel) vs the torch source.

Wires encoder + length-regulator + CFM(estimator=UNet decoder), copies all weights
(encoder direct; length_regulator + decoder.estimator with conv-layout transposes
and the decoder key remap), injects fixed CFM noise, and compares the mel output.

Requires diffusers + conformer.

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_flow_full_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)
sys.path.insert(0, os.path.join(BV, "third_party", "Matcha-TTS"))

# Stub matcha.utils.pylogger to avoid pulling training deps (lightning/hydra)
# via matcha.utils.__init__; BASECFM only needs get_pylogger.
import logging  # noqa: E402
import types  # noqa: E402
_mu = types.ModuleType("matcha.utils")
_mu.__path__ = []  # mark as package
_pl = types.ModuleType("matcha.utils.pylogger")
_pl.get_pylogger = lambda name=None: logging.getLogger(name)
sys.modules["matcha.utils"] = _mu
sys.modules["matcha.utils.pylogger"] = _pl

import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from cosyvoice.transformer.encoder import ConformerEncoder as TConformer  # noqa: E402
from cosyvoice.flow.length_regulator import InterpolateRegulator as TReg  # noqa: E402
from cosyvoice.flow.decoder import ConditionalDecoder as TDecoder  # noqa: E402
from cosyvoice.flow.flow_matching import ConditionalCFM as TCFM  # noqa: E402
from cosyvoice.flow.flow import MaskedDiffWithXvec as TFlow  # noqa: E402

from breezyvoice_mlx.transformer.encoder import ConformerEncoder  # noqa: E402
from breezyvoice_mlx.flow.length_regulator import InterpolateRegulator  # noqa: E402
from breezyvoice_mlx.flow.decoder import ConditionalDecoder, _remap_key  # noqa: E402
from breezyvoice_mlx.flow.flow_matching import ConditionalCFM  # noqa: E402
from breezyvoice_mlx.flow.flow import MaskedDiffWithXvec  # noqa: E402

CFM_PARAMS = DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                         "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7,
                         "reg_loss_type": "l1"})
IN, OUT, SPK, VOCAB = 8, 8, 8, 20


def _torch_flow():
    te = TConformer(input_size=IN, output_size=16, attention_heads=2, linear_units=32,
                    num_blocks=1, dropout_rate=0.0, positional_dropout_rate=0.0,
                    attention_dropout_rate=0.0, input_layer="linear",
                    pos_enc_layer_type="rel_pos_espnet",
                    selfattention_layer_type="rel_selfattn", use_cnn_module=False,
                    macaron_style=False)
    lr = TReg(channels=OUT, sampling_ratios=[1, 1], out_channels=OUT, groups=1)
    est = TDecoder(in_channels=4 * OUT, out_channels=OUT, channels=[16, 16], dropout=0.0,
                   attention_head_dim=8, n_blocks=1, num_mid_blocks=1, num_heads=2, act_fn="gelu")
    cfm = TCFM(in_channels=OUT, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=OUT, estimator=est)
    return TFlow(input_size=IN, output_size=OUT, spk_embed_dim=SPK, vocab_size=VOCAB,
                 encoder=te, length_regulator=lr, decoder=cfm).eval()


def _mlx_flow():
    te = ConformerEncoder(input_size=IN, output_size=16, attention_heads=2,
                          linear_units=32, num_blocks=1, static_chunk_size=0)
    lr = InterpolateRegulator(channels=OUT, sampling_ratios=[1, 1], out_channels=OUT, groups=1)
    est = ConditionalDecoder(in_channels=4 * OUT, out_channels=OUT, channels=(16, 16),
                             dropout=0.0, attention_head_dim=8, n_blocks=1,
                             num_mid_blocks=1, num_heads=2, act_fn="gelu")
    cfm = ConditionalCFM(in_channels=OUT, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=OUT,
                         estimator=est)
    return MaskedDiffWithXvec(input_size=IN, output_size=OUT, spk_embed_dim=SPK,
                              vocab_size=VOCAB, encoder=te, length_regulator=lr, decoder=cfm)


def _remap_flow(state: dict) -> list:
    out = []
    for k, v in state.items():
        a = np.asarray(v.detach().cpu().numpy(), dtype=np.float32)
        nk = k
        if k.startswith("decoder.estimator."):
            inner = _remap_key(k[len("decoder.estimator."):])
            nk = "decoder.estimator." + inner
            if k.endswith(".weight") and a.ndim == 3:
                a = np.transpose(a, (1, 2, 0)) if ".upsample.conv." in nk else np.transpose(a, (0, 2, 1))
        elif k.endswith(".weight") and a.ndim == 3:   # length_regulator convs
            a = np.transpose(a, (0, 2, 1))
        out.append((nk, mx.array(a)))
    return out


def test_flow_full_parity():
    tflow, mflow = _torch_flow(), _mlx_flow()
    mflow.load_weights(_remap_flow(dict(tflow.state_dict())), strict=False)

    rng = np.random.default_rng(0)
    Ttok = 6
    token = rng.integers(0, VOCAB, size=(1, Ttok)).astype(np.int32)
    token_len = np.array([Ttok], dtype=np.int32)
    spk = rng.standard_normal((1, SPK)).astype(np.float32)
    empty_tok = np.zeros((1, 0), dtype=np.int32)
    empty_len = np.array([0], dtype=np.int32)
    empty_feat = np.zeros((1, 0, OUT), dtype=np.float32)

    # mel length the flow will produce, to size the injected noise
    mel_len = int(token_len[0] / 50 * 22050 / 256)
    z = rng.standard_normal((1, OUT, mel_len)).astype(np.float32)

    # inject fixed z into torch CFM
    def fwd(mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        span = torch.linspace(0, 1, n_timesteps + 1)
        span = 1 - torch.cos(span * 0.5 * torch.pi)
        return tflow.decoder.solve_euler(torch.from_numpy(z), t_span=span, mu=mu,
                                         mask=mask, spks=spks, cond=cond)
    tflow.decoder.forward = fwd

    with torch.no_grad():
        t_mel = tflow.inference(
            token=torch.from_numpy(token), token_len=torch.from_numpy(token_len),
            prompt_token=torch.from_numpy(empty_tok), prompt_token_len=torch.from_numpy(empty_len),
            prompt_feat=torch.from_numpy(empty_feat), prompt_feat_len=torch.from_numpy(empty_len),
            embedding=torch.from_numpy(spk))
    m_mel = mflow.inference(
        token=mx.array(token), token_len=mx.array(token_len),
        prompt_token=mx.array(empty_tok), prompt_token_len=mx.array(empty_len),
        prompt_feat=mx.array(empty_feat), prompt_feat_len=mx.array(empty_len),
        embedding=mx.array(spk), z=mx.array(z))

    t_mel, m_mel = t_mel.numpy(), np.array(m_mel)
    print("shapes:", m_mel.shape, t_mel.shape)
    np.testing.assert_allclose(m_mel, t_mel, rtol=2e-2, atol=5e-3)
    print("OK: MaskedDiffWithXvec (token->mel) full-flow parity")


if __name__ == "__main__":
    test_flow_full_parity()
    print("FULL FLOW PARITY TEST PASSED")
