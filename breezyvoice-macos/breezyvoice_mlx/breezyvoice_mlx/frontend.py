"""Frontend: text normalization + feature/token/embedding extraction.
[PORT STATUS: STUB — but mostly REUSABLE on macOS, see below]

PyTorch source: ../BreezyVoice/cosyvoice/cli/frontend.py (151 lines)
Class: CosyVoiceFrontEnd

IMPORTANT — the ONNX pieces are NOT a hard blocker on Apple Silicon:
  * campplus.onnx (speaker embedding) already runs with CPUExecutionProvider.
  * speech_tokenizer_v1.onnx only used CUDAExecutionProvider *if cuda present*;
    it falls back to CPUExecutionProvider otherwise. onnxruntime ships arm64
    macOS wheels, so both run on CPU today.
  => Phase 1 strategy: REUSE the existing PyTorch/onnxruntime frontend unchanged
     and only swap the three model weights (llm/flow/hift) for MLX. Porting the
     ONNX models to MLX-native is a *later optimization*, not a prerequisite.

Pure-Python parts (text_normalize, WeTextProcessing, inflect, split_paragraph)
carry over verbatim. whisper.log_mel_spectrogram and torchaudio kaldi.fbank can
stay on CPU/torch for the prompt path, or be reimplemented with mlx later.

This module will wrap the existing frontend and return MLX arrays at the model
boundary. For now it is a placeholder describing the seam.
"""

from __future__ import annotations


class MlxFrontEnd:  # TODO: thin adapter over CosyVoiceFrontEnd, returning mx.arrays
    """Adapter that produces MLX-ready inputs from the existing frontend.

    Planned: hold a reference to the upstream CosyVoiceFrontEnd (CPU/torch),
    call its frontend_zero_shot / frontend_sft / ... methods, and convert the
    resulting tensors to mx.array at the boundary (torch -> numpy -> mx).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MlxFrontEnd adapter pending. Phase 1: wrap upstream CosyVoiceFrontEnd "
            "(runs on CPU via onnxruntime) and convert outputs to mx.array."
        )
