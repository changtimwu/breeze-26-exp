"""Parity: full MLX TransformerLM vs BreezyVoice's PyTorch TransformerLM.

The decode loop is stochastic (top-k multinomial), so we force GREEDY argmax on
both sides (monkeypatch torch sampling_ids; pass greedy_sampling to MLX) and
compare the generated speech-token sequences exactly. Identical logits -> identical
argmax -> identical tokens (and identical EOS stop step).

Run:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tests/test_llm_parity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

BV = os.path.join(os.path.dirname(__file__), "..", "..", "BreezyVoice")
sys.path.insert(0, BV)

import torch  # noqa: E402
from cosyvoice.transformer.encoder import ConformerEncoder as TConformer  # noqa: E402
from cosyvoice.transformer.encoder import TransformerEncoder as TTransformer  # noqa: E402
from cosyvoice.llm.llm import TransformerLM as TorchLM  # noqa: E402

from breezyvoice_mlx.transformer.encoder import ConformerEncoder, TransformerEncoder  # noqa: E402
from breezyvoice_mlx.llm.llm import TransformerLM, greedy_sampling  # noqa: E402

TEI, LIS, LOS = 16, 32, 32          # text_enc_in, llm_in, llm_out
TXT, SPK_TOK, SPK_DIM = 50, 20, 16  # text vocab, speech vocab, spk emb dim


def _torch_lm():
    te = TConformer(input_size=TEI, output_size=LIS, attention_heads=4,
                    linear_units=64, num_blocks=2, dropout_rate=0.0,
                    positional_dropout_rate=0.0, attention_dropout_rate=0.0,
                    input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                    selfattention_layer_type="rel_selfattn", use_cnn_module=False,
                    macaron_style=False, static_chunk_size=1)
    llm = TTransformer(input_size=LIS, output_size=LOS, attention_heads=4,
                       linear_units=64, num_blocks=2, dropout_rate=0.0,
                       positional_dropout_rate=0.0, attention_dropout_rate=0.0,
                       input_layer="linear_legacy", pos_enc_layer_type="rel_pos_espnet",
                       selfattention_layer_type="rel_selfattn", static_chunk_size=1)
    return TorchLM(TEI, LIS, LOS, TXT, SPK_TOK, te, llm, spk_embed_dim=SPK_DIM).eval()


def _mlx_lm():
    te = ConformerEncoder(input_size=TEI, output_size=LIS, attention_heads=4,
                          linear_units=64, num_blocks=2, static_chunk_size=1)
    llm = TransformerEncoder(input_size=LIS, output_size=LOS, attention_heads=4,
                             linear_units=64, num_blocks=2, static_chunk_size=1)
    return TransformerLM(TEI, LIS, LOS, TXT, SPK_TOK, te, llm, spk_embed_dim=SPK_DIM)


def test_transformerlm_greedy_parity():
    tmod = _torch_lm()
    mmod = _mlx_lm()
    items = {k: mx.array(v.detach().numpy()) for k, v in tmod.state_dict().items()}
    mmod.load_weights(list(items.items()), strict=False)

    # force greedy on torch
    def greedy_ids(self, weighted_scores, sampling, beam_size, ignore_eos=True):
        return weighted_scores.argmax(dim=-1, keepdim=True)
    tmod.sampling_ids = greedy_ids.__get__(tmod, TorchLM)

    rng = np.random.default_rng(0)
    Ttxt = 5
    text = rng.integers(0, TXT, size=(1, Ttxt)).astype(np.int32)
    text_len = np.array([Ttxt], dtype=np.int32)
    spk = rng.standard_normal((1, SPK_DIM)).astype(np.float32)
    empty_txt = np.zeros((1, 0), dtype=np.int32)
    empty_len = np.array([0], dtype=np.int32)

    with torch.no_grad():
        t_out = tmod.inference(
            text=torch.from_numpy(text), text_len=torch.from_numpy(text_len),
            prompt_text=torch.from_numpy(empty_txt), prompt_text_len=torch.from_numpy(empty_len),
            prompt_speech_token=torch.from_numpy(empty_txt),
            prompt_speech_token_len=torch.from_numpy(empty_len),
            embedding=torch.from_numpy(spk), sampling=25,
            max_token_text_ratio=4, min_token_text_ratio=0)
    t_tokens = t_out.numpy().reshape(-1).tolist()

    m_out = mmod.inference(
        text=mx.array(text), text_len=mx.array(text_len),
        prompt_text=mx.array(empty_txt), prompt_text_len=mx.array(empty_len),
        prompt_speech_token=mx.array(empty_txt), prompt_speech_token_len=mx.array(empty_len),
        embedding=mx.array(spk), sampling=25, max_token_text_ratio=4,
        min_token_text_ratio=0, sampling_fn=greedy_sampling)
    m_tokens = np.array(m_out).reshape(-1).tolist()

    print(f"torch tokens: {t_tokens}")
    print(f"mlx   tokens: {m_tokens}")
    assert m_tokens == t_tokens, "token sequences diverged"
    assert len(m_tokens) > 0, "no tokens generated (test would be vacuous)"
    print(f"OK: TransformerLM greedy parity ({len(m_tokens)} tokens)")


if __name__ == "__main__":
    test_transformerlm_greedy_parity()
    print("LLM PARITY TEST PASSED")
