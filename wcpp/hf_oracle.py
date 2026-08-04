#!/usr/bin/env python3
"""Ground-truth reference: original Breeze-ASR-26 fp32 weights via HuggingFace transformers.

This is the oracle the ggml/MLX conversions are judged against — the unconverted,
unquantized checkpoint straight from MediaTek. Slow and memory-hungry by design;
run it on short clips only.

Usage: hf_oracle.py <audio.wav> [more.wav ...]
"""
import argparse
import json
import sys
import time

MODEL = "MediaTek-Research/Breeze-ASR-26"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--device", default=None, help="mps, cpu (default: mps if available)")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    ap.add_argument("--num-beams", type=int, default=1,
                    help="1 = greedy (default), to match the other engines under "
                         "compare.py's matched-greedy mode")
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if args.device:
        device = args.device
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # fp32 on purpose: this is the reference, not a speed run. MPS lacks kernels for
    # a few fp64/fp16 paths, so keep it simple and let it be slow.
    print(f"[oracle] loading {MODEL} (fp32) on {device} ...", file=sys.stderr)
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
    model.to(device).eval()
    print(f"[oracle] loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    import numpy as np
    import wave

    results = []
    for path in args.audio:
        with wave.open(path) as w:
            assert w.getframerate() == 16000, f"{path}: expected 16 kHz"
            assert w.getnchannels() == 1, f"{path}: expected mono"
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        dur = len(audio) / 16000.0

        # 30 s Whisper windows, matching what the engines do internally.
        window = 30 * 16000
        chunks = [audio[i:i + window] for i in range(0, len(audio), window)] or [audio]

        texts = []
        t0 = time.time()
        for ch in chunks:
            feats = processor(
                ch, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device, torch.float32)
            with torch.no_grad():
                ids = model.generate(
                    feats,
                    language=args.language,
                    task="transcribe",
                    num_beams=args.num_beams,
                    do_sample=False,
                    max_new_tokens=440,
                )
            texts.append(
                processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            )
        wall = time.time() - t0
        text = " ".join(t for t in texts if t)

        print(f"\n--- {path}  ({dur:.1f}s audio, {wall:.1f}s decode, xRT {wall / dur:.2f}) ---")
        print(text)
        results.append(
            {"file": path, "audio_s": dur, "decode_s": wall, "text": text}
        )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\n[oracle] wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
