"""Phase 0 — stand up a working CosyVoice MLX baseline via mlx-audio-plus.

This validates that the MLX TTS stack runs end-to-end on this machine and gives
us a live reference implementation to adapt for the BreezyVoice port.

We run **CosyVoice3** here because it's the only CosyVoice with PREBUILT MLX
weights on the Hub (mlx-community/Fun-CosyVoice3-0.5B-2512-*). CosyVoice2 also
has MLX *code* in mlx-audio-plus but no published MLX weights (would need
convert_from_source on the FunAudioLLM PyTorch checkpoint).

NOTE on lineage (see PORTING_STATUS.md): BreezyVoice is CosyVoice *v1*-derived
(Conformer + RNN TransformerLM, speech_tokenizer_v1). CosyVoice3 uses a Qwen2
LLM + DiT flow; CosyVoice2 uses Qwen2 LLM + UNet1D flow (closest to BreezyVoice's
flow). So treat this baseline as a stack smoke-test + vocoder/flow reference, not
an LLM template.

Usage (from breezyvoice-macos/, venv has mlx-audio-plus):
    .venv/bin/python breezyvoice_mlx/tools/phase0_baseline.py \
        --ref-audio BreezyVoice/data/example.wav \
        --ref-text "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容..." \
        --text "歡迎使用聯發創新基地 BreezyVoice 模型。" \
        --out results/phase0_cosyvoice3_mlx.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

from mlx_audio.tts.models.cosyvoice3 import Model, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="results/phase0_cosyvoice3_mlx.wav")
    ap.add_argument("--repo", default="mlx-community/Fun-CosyVoice3-0.5B-2512-8bit")
    args = ap.parse_args()

    print(f"[phase0] resolving model {args.repo} ...")
    model_path = snapshot_download(args.repo)
    print(f"[phase0] model at {model_path}")

    model = Model(ModelConfig(model_path=model_path))

    # Reference audio: load and (the model resamples internally to its own rate).
    ref, sr = sf.read(args.ref_audio, dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    print(f"[phase0] ref audio: {len(ref)/sr:.1f}s @ {sr}Hz")

    sample_rate = model.sample_rate
    if sr != sample_rate:
        from scipy.signal import resample
        ref = resample(ref, int(len(ref) * sample_rate / sr)).astype(np.float32)

    t0 = time.time()
    results = list(model.generate(
        text=args.text,
        ref_audio=ref,
        ref_text=args.ref_text,
        verbose=True,
    ))
    gen_s = time.time() - t0

    audio = np.concatenate([np.array(r.audio).reshape(-1) for r in results])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, sample_rate)

    dur = len(audio) / sample_rate
    print(f"[phase0] wrote {args.out}  ({dur:.2f}s audio in {gen_s:.2f}s wall, "
          f"RTF={gen_s/dur:.3f}) @ {sample_rate}Hz")


if __name__ == "__main__":
    main()
