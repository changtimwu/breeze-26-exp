"""Inference orchestrator: llm -> flow -> hift.  [PORT STATUS: IMPLEMENTED]

MLX port of ../BreezyVoice/cosyvoice/cli/model.py CosyVoiceModel. MLX uses unified
memory, so there is no device juggling and no empty_cache. The three-stage call
graph mirrors the torch source:

    tts_speech_token = llm.inference(text, ..., embedding=llm_embedding)
    tts_mel          = flow.inference(token=tts_speech_token, ..., embedding=flow_embedding)
    tts_speech, _    = hift(mel=tts_mel)            # HiFTGenerator returns (wav, source)
"""

from __future__ import annotations

import mlx.core as mx

from .llm.llm import TransformerLM, top_k_sampling
from .flow.flow import MaskedDiffWithXvec
from .hifigan.generator import HiFTGenerator


def _empty(int_dtype=mx.int32):
    return mx.zeros((1, 0), dtype=int_dtype)


class CosyVoiceModel:
    def __init__(self, llm: TransformerLM, flow: MaskedDiffWithXvec, hift: HiFTGenerator):
        self.llm = llm
        self.flow = flow
        self.hift = hift

    def load(self, llm_weights: str, flow_weights: str, hift_weights: str):
        """Load converted MLX safetensors (see tools/convert_breezyvoice.py).
        strict=False so init-computed constants (EspnetRelPositionalEncoding._pe,
        HiFTGenerator.stft_window) — absent from the checkpoint — are kept."""
        self.llm.load_weights(llm_weights, strict=False)
        self.flow.load_weights(flow_weights, strict=False)
        self.hift.load_weights(hift_weights, strict=False)
        for m in (self.llm, self.flow, self.hift):
            m.eval()

    def inference(self, text, text_len, flow_embedding, llm_embedding,
                  prompt_text=None, prompt_text_len=None,
                  llm_prompt_speech_token=None, llm_prompt_speech_token_len=None,
                  flow_prompt_speech_token=None, flow_prompt_speech_token_len=None,
                  prompt_speech_feat=None, prompt_speech_feat_len=None,
                  sampling: int = 25, sampling_fn=top_k_sampling) -> dict:
        z = mx.zeros
        prompt_text = prompt_text if prompt_text is not None else _empty()
        prompt_text_len = prompt_text_len if prompt_text_len is not None else mx.zeros((1,), mx.int32)
        llm_pt = llm_prompt_speech_token if llm_prompt_speech_token is not None else _empty()
        llm_ptl = llm_prompt_speech_token_len if llm_prompt_speech_token_len is not None else mx.zeros((1,), mx.int32)
        flow_pt = flow_prompt_speech_token if flow_prompt_speech_token is not None else _empty()
        flow_ptl = flow_prompt_speech_token_len if flow_prompt_speech_token_len is not None else mx.zeros((1,), mx.int32)
        pf = prompt_speech_feat if prompt_speech_feat is not None else z((1, 0, 80))
        pfl = prompt_speech_feat_len if prompt_speech_feat_len is not None else mx.zeros((1,), mx.int32)

        tts_speech_token = self.llm.inference(
            text=text, text_len=text_len, prompt_text=prompt_text,
            prompt_text_len=prompt_text_len, prompt_speech_token=llm_pt,
            prompt_speech_token_len=llm_ptl, embedding=llm_embedding,
            sampling=sampling, sampling_fn=sampling_fn)

        tts_mel = self.flow.inference(
            token=tts_speech_token,
            token_len=mx.array([tts_speech_token.shape[1]], dtype=mx.int32),
            prompt_token=flow_pt, prompt_token_len=flow_ptl,
            prompt_feat=pf, prompt_feat_len=pfl, embedding=flow_embedding)

        tts_speech, _ = self.hift(tts_mel)
        return {"tts_speech": tts_speech}
