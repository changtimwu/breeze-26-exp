"""Inference orchestrator: llm -> flow -> hift.  [PORT STATUS: STUB — EASY once parts land]

PyTorch source: ../BreezyVoice/cosyvoice/cli/model.py (61 lines)
Class: CosyVoiceModel

Replaces device management (cuda/mps/cpu .to()) with MLX (unified memory — no
device juggling) and drops torch.cuda.empty_cache(). The three-stage call graph
is otherwise identical:

    tts_speech_token = llm.inference(text, ..., embedding=llm_embedding)
    tts_mel          = flow.inference(token=tts_speech_token, ..., embedding=flow_embedding)
    tts_speech       = hift.inference(mel=tts_mel)
"""

from __future__ import annotations

import mlx.core as mx

from .llm.llm import TransformerLM
from .flow.flow import MaskedDiffWithXvec
from .hifigan.generator import HiFTGenerator


class CosyVoiceModel:
    def __init__(self, llm: TransformerLM, flow: MaskedDiffWithXvec, hift: HiFTGenerator):
        self.llm = llm
        self.flow = flow
        self.hift = hift

    def load(self, llm_weights: str, flow_weights: str, hift_weights: str):
        """Load fused MLX safetensors produced by tools/convert_weights.py."""
        raise NotImplementedError("Wire up self.llm/flow/hift.load_weights once modules land.")

    def inference(self, text, text_len, flow_embedding, llm_embedding,
                  prompt_text=None, prompt_text_len=None,
                  llm_prompt_speech_token=None, llm_prompt_speech_token_len=None,
                  flow_prompt_speech_token=None, flow_prompt_speech_token_len=None,
                  prompt_speech_feat=None, prompt_speech_feat_len=None) -> dict:
        raise NotImplementedError(
            "Orchestrator pending — depends on llm/flow/hift modules. "
            "Mirror ../BreezyVoice/cosyvoice/cli/model.py:inference."
        )
