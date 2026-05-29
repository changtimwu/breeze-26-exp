"""High-level BreezyVoice-MLX API.  [PORT STATUS: IMPLEMENTED]

Builds the three MLX stages (TransformerLM, MaskedDiffWithXvec, HiFTGenerator)
with BreezyVoice's cosyvoice.yaml config, loads converted weights
(tools/convert_breezyvoice.py), and runs inference.

Inference modes:
  * inference_sft(text, spk_id): uses a built-in speaker embedding from spk2info.pt
    — needs only the text tokenizer (no ONNX speech tokenizer / CAM++ / prompt
    audio), so it's the simplest fully-MLX path.
  * (zero-shot with arbitrary prompt audio needs the ONNX speech tokenizer + CAM++
    on CPU — see frontend.py; not required for SFT.)

Config is hard-coded from MediaTek-Research/BreezyVoice's cosyvoice.yaml to avoid a
hyperpyyaml dependency.
"""

from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from .transformer.encoder import ConformerEncoder, TransformerEncoder
from .llm.llm import TransformerLM
from .flow.flow import MaskedDiffWithXvec
from .flow.length_regulator import InterpolateRegulator
from .flow.flow_matching import ConditionalCFM
from .flow.decoder import ConditionalDecoder
from .hifigan.generator import HiFTGenerator
from .hifigan.f0_predictor import ConvRNNF0Predictor
from .model import CosyVoiceModel


class _CFMParams:
    sigma_min = 1e-6
    t_scheduler = "cosine"
    inference_cfg_rate = 0.7
    training_cfg_rate = 0.2


def build_models() -> CosyVoiceModel:
    """Instantiate the three MLX stages with BreezyVoice's config."""
    # --- LLM (TransformerLM) ---
    text_encoder = ConformerEncoder(
        input_size=512, output_size=1024, attention_heads=16, linear_units=4096,
        num_blocks=6, static_chunk_size=1)
    llm_backbone = TransformerEncoder(
        input_size=1024, output_size=1024, attention_heads=16, linear_units=4096,
        num_blocks=14, static_chunk_size=1)
    llm = TransformerLM(
        text_encoder_input_size=512, llm_input_size=1024, llm_output_size=1024,
        text_token_size=51866, speech_token_size=4096,
        text_encoder=text_encoder, llm=llm_backbone, spk_embed_dim=192)

    # --- Flow (MaskedDiffWithXvec) ---
    flow_encoder = ConformerEncoder(
        input_size=512, output_size=512, attention_heads=8, linear_units=2048,
        num_blocks=6, static_chunk_size=0)
    estimator = ConditionalDecoder(
        in_channels=320, out_channels=80, channels=(256, 256), dropout=0.0,
        attention_head_dim=64, n_blocks=4, num_mid_blocks=12, num_heads=8, act_fn="gelu")
    cfm = ConditionalCFM(in_channels=240, cfm_params=_CFMParams(), n_spks=1,
                         spk_emb_dim=80, estimator=estimator)
    flow = MaskedDiffWithXvec(
        input_size=512, output_size=80, spk_embed_dim=192, vocab_size=4096,
        input_frame_rate=50, encoder=flow_encoder,
        length_regulator=InterpolateRegulator(channels=80, sampling_ratios=[1, 1, 1, 1]),
        decoder=cfm)

    # --- HiFiGAN-NSF vocoder ---
    hift = HiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=22050,
        nsf_alpha=0.1, nsf_sigma=0.003, nsf_voiced_threshold=10,
        upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        f0_predictor=ConvRNNF0Predictor(in_channels=80, cond_channels=512))

    return CosyVoiceModel(llm, flow, hift)


class BreezyVoice:
    SAMPLE_RATE = 22050

    def __init__(self, model_dir: str, weights_dir: str = None):
        import torch
        self.model_dir = model_dir
        weights_dir = weights_dir or model_dir
        self.model = build_models()
        self.model.load(
            os.path.join(weights_dir, "llm.safetensors"),
            os.path.join(weights_dir, "flow.safetensors"),
            os.path.join(weights_dir, "hift.safetensors"))
        self.spk2info = torch.load(os.path.join(model_dir, "spk2info.pt"),
                                   map_location="cpu")
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from whisper.tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer(multilingual=True)
        return self._tokenizer

    def _encode_text(self, text: str):
        enc = self.tokenizer
        ids = (enc.encode(text, allowed_special="all") if hasattr(enc, "encode")
               else enc.encoding.encode(text, allowed_special="all"))
        return mx.array([ids], dtype=mx.int32), mx.array([len(ids)], dtype=mx.int32)

    def list_speakers(self):
        return list(self.spk2info.keys())

    def inference_sft(self, text: str, spk_id: str) -> mx.array:
        """Bare SFT: speaker embedding only, no prompt conditioning. Returns (1,T).
        NOTE: BreezyVoice is a zero-shot model; without prompt-feat conditioning the
        flow can produce a hot/over-driven mel. Prefer inference_zero_shot_builtin."""
        text_token, text_len = self._encode_text(text)
        emb = mx.array(np.asarray(self.spk2info[spk_id]["embedding"], dtype=np.float32).reshape(1, -1))
        out = self.model.inference(text=text_token, text_len=text_len,
                                   flow_embedding=emb, llm_embedding=emb)
        return out["tts_speech"]

    def inference_zero_shot_builtin(self, text: str, spk_id: str) -> mx.array:
        """Zero-shot using a built-in speaker's PRECOMPUTED prompt (embedding +
        speech_token + speech_feat from spk2info) — the intended conditioning, no
        ONNX tokenizer / prompt audio needed. Returns (1, T) waveform."""
        text_token, text_len = self._encode_text(text)
        e = self.spk2info[spk_id]
        emb = mx.array(np.asarray(e["embedding"], dtype=np.float32).reshape(1, -1))
        st = mx.array(np.asarray(e["speech_token"]).astype(np.int32))
        stl = mx.array([st.shape[1]], dtype=mx.int32)
        feat = mx.array(np.asarray(e["speech_feat"], dtype=np.float32))
        featl = mx.array([feat.shape[1]], dtype=mx.int32)
        out = self.model.inference(
            text=text_token, text_len=text_len, flow_embedding=emb, llm_embedding=emb,
            llm_prompt_speech_token=st, llm_prompt_speech_token_len=stl,
            flow_prompt_speech_token=st, flow_prompt_speech_token_len=stl,
            prompt_speech_feat=feat, prompt_speech_feat_len=featl)
        return out["tts_speech"]
