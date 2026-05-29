"""Text-to-speech-token LLM.  [PORT STATUS: STUB — MEDIUM]

PyTorch source: ../BreezyVoice/cosyvoice/llm/llm.py (206 lines)
Class: TransformerLM  (Conformer text_encoder + RNN/transformer `llm` + linear decoder)

This is BreezyVoice's CosyVoice2 LLM — NOT Qwen2. Do not try to reuse the
mlx-audio-plus CosyVoice3 LLM here; the backbone differs.

Port the inference() path (the AR decode loop), see source lines 147-206:
  1. concat(prompt_text, text) -> text_embedding -> encode (Conformer)
  2. spk embedding: F.normalize + affine + unsqueeze
  3. build lm_input = [sos_eos, embedding, text, task_id, prompt_speech_token_emb]
  4. step-by-step decode up to max_len, threading att_cache/cnn_cache through
     self.llm.forward_chunk(...), sampling via sampling_ids (top-k=25 multinomial),
     stop on speech_token_size (EOS) once i >= min_len.

MLX specifics:
  * Sampling uses mx.random.categorical on top-k logits. RNG: seed via
    mx.random.key for reproducibility (no global Math.random equivalent needed).
  * KV/att cache: keep as plain mx.arrays threaded through the loop, like torch.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class TransformerLM(nn.Module):  # TODO: implement
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TransformerLM MLX port pending.")

    def inference(
        self,
        text: mx.array,
        text_len: mx.array,
        prompt_text: mx.array,
        prompt_text_len: mx.array,
        prompt_speech_token: mx.array,
        prompt_speech_token_len: mx.array,
        embedding: mx.array,
        sampling: int = 25,
        max_token_text_ratio: float = 30,
        min_token_text_ratio: float = 3,
    ) -> mx.array:  # pragma: no cover
        raise NotImplementedError
