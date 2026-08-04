"""MLX quantization for the BreezyVoice LLM.

We quantize only the LLM's **Linear** layers (the ~1 GB bulk: attention q/k/v/out,
FFN, projections). We deliberately do NOT quantize embeddings:
  * `speech_embedding.weight` and `llm_embedding.weight` are indexed directly in
    the AR decode loop (`speech_embedding.weight[top_id]`); a QuantizedEmbedding's
    `.weight` is packed uint32, which would break that.
  * Flow + HiFiGAN are left fp32 — TTS quality is far more sensitive there (and
    they're small: 400 MB + 78 MB).

All LLM Linear in-features are divisible by 64, so group_size 64 / 4-bit applies
cleanly. The same predicate must be used at convert time and load time so the
module tree is swapped to QuantizedLinear identically before weights load.
"""

from __future__ import annotations

import mlx.nn as nn

DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 64


def _linear_predicate(group_size: int):
    def predicate(path: str, module: nn.Module) -> bool:
        return (isinstance(module, nn.Linear)
                and module.weight.shape[-1] % group_size == 0)
    return predicate


def quantize_llm(llm: nn.Module, bits: int = DEFAULT_BITS,
                 group_size: int = DEFAULT_GROUP_SIZE) -> None:
    """In-place: swap the LLM's eligible Linear layers to QuantizedLinear."""
    nn.quantize(llm, group_size=group_size, bits=bits,
                class_predicate=_linear_predicate(group_size))
