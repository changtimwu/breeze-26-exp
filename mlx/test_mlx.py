#!/usr/bin/env python3
"""
Smoke test for the MLX 4-bit port of Breeze-ASR-26.

Model:  https://huggingface.co/doggy8088/Breeze-ASR-26-MLX-4bit
Runtime: Apple Silicon (M1/M2/M3/M4) only — MLX uses Metal.

Usage:
    python test_mlx.py                       # transcribes data/test.m4a from the
                                             # breeze-asr-taigi repo if present
    python test_mlx.py path/to/audio.wav     # transcribe a specific file
    python test_mlx.py audio.wav --fp16      # use the unquantized fp16 model
    python test_mlx.py audio.wav --no-srt    # skip SRT, only print text

Outputs:
    <audio>.mlx.txt   plain transcript
    <audio>.mlx.srt   SubRip subtitles (unless --no-srt)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path


MODEL_4BIT = "doggy8088/Breeze-ASR-26-MLX-4bit"
MODEL_FP16 = "doggy8088/Breeze-ASR-26-MLX"


def assert_apple_silicon() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit(
            f"[abort] MLX requires Apple Silicon (darwin/arm64); "
            f"detected {sys.platform}/{platform.machine()}."
        )


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def resolve_audio(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
        if not p.exists():
            sys.exit(f"[abort] audio file not found: {p}")
        return p

    # Fall back to the bundled sample from the breeze-asr-taigi repo, looked up
    # relative to this script: mlx/ lives next to a clone of breeze-asr-taigi.
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".." / "breeze-asr-taigi" / "data" / "test.m4a",
        here.parent / "breeze-asr-taigi" / "data" / "test.m4a",
        Path.cwd() / "breeze-asr-taigi" / "data" / "test.m4a",
    ]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c
    sys.exit(
        "[abort] no audio argument given and could not find "
        "breeze-asr-taigi/data/test.m4a near this script. "
        "Clone the repo next to this mlx/ dir, or pass an audio path."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="?", help="audio file path (default: bundled test.m4a)")
    parser.add_argument("--fp16", action="store_true", help=f"use {MODEL_FP16} instead of the 4-bit model")
    parser.add_argument("--language", default="zh", help="language code passed to whisper (default: zh)")
    parser.add_argument("--word-timestamps", action="store_true", help="emit per-word timestamps")
    parser.add_argument("--no-srt", action="store_true", help="skip writing the .srt file")
    args = parser.parse_args()

    assert_apple_silicon()

    try:
        import mlx_whisper
    except ImportError:
        sys.exit("[abort] mlx_whisper not installed. Run: pip install -U mlx-whisper")

    audio_path = resolve_audio(args.audio)
    model_id = MODEL_FP16 if args.fp16 else MODEL_4BIT

    print(f"[info] model:  {model_id}")
    print(f"[info] audio:  {audio_path}")
    print(f"[info] lang:   {args.language}")
    print("[info] transcribing...")

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_id,
        language=args.language,
        word_timestamps=args.word_timestamps,
    )
    elapsed = time.perf_counter() - t0

    text = result.get("text", "").strip()
    segments = result.get("segments", []) or []

    audio_dur = segments[-1]["end"] if segments else None
    xrt = (audio_dur / elapsed) if (audio_dur and elapsed > 0) else None

    print()
    print("=" * 60)
    print(text or "(empty transcript)")
    print("=" * 60)
    print(f"[done] transcribe wall: {elapsed:.2f}s"
          + (f" | audio: {audio_dur:.2f}s | xRT: {xrt:.2f}x" if xrt else ""))

    stem = audio_path.with_suffix("")
    txt_out = Path(f"{stem}.mlx.txt")
    txt_out.write_text(text + "\n", encoding="utf-8")
    print(f"[write] {txt_out}")

    if not args.no_srt and segments:
        srt_out = Path(f"{stem}.mlx.srt")
        srt_out.write_text(to_srt(segments), encoding="utf-8")
        print(f"[write] {srt_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
