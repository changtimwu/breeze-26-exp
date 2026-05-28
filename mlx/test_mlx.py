#!/usr/bin/env python3
"""
Smoke test for the MLX 4-bit port of Breeze-ASR-26.

Model:  https://huggingface.co/doggy8088/Breeze-ASR-26-MLX-4bit
Runtime: Apple Silicon (M1/M2/M3/M4) only — MLX uses Metal.

Usage:
    python test_mlx.py                       # transcribes data/test.m4a from the
                                             # breeze-asr-taigi repo if present
    python test_mlx.py path/to/audio.wav     # transcribe a specific file
    python test_mlx.py --mic                 # record from default mic until Enter
    python test_mlx.py --mic --duration 10   # record 10 seconds then transcribe
    python test_mlx.py --continuous          # fixed-chunk live transcription (Ctrl-C to stop)
    python test_mlx.py --continuous --chunk 4  # use 4s chunks instead of the 5s default
    python test_mlx.py --continuous --vad    # VAD-segmented utterances via webrtcvad
    python test_mlx.py --continuous --vad --vad-silence-ms 800
    python test_mlx.py audio.wav --fp16      # use the unquantized fp16 model
    python test_mlx.py audio.wav --no-srt    # skip SRT, only print text

Mic / continuous mode needs `pip install sounddevice` and macOS microphone
permission for your terminal (System Settings → Privacy & Security → Microphone).
The --vad mode additionally needs `pip install webrtcvad`.

Continuous modes:
  --continuous            naive fixed-chunk pseudo-streaming. Records into a
                          fixed-size buffer in the background and transcribes
                          each chunk independently when it fills. No overlap,
                          no VAD — words at chunk boundaries get cut. Tune
                          --chunk for the latency/accuracy trade-off.
  --continuous --vad      utterance-segmented streaming. webrtcvad classifies
                          30 ms frames as speech/silence; the script starts an
                          utterance on the first speech frame (with ~300 ms
                          pre-roll) and finalises it once --vad-silence-ms of
                          trailing silence is seen, then sends the whole
                          utterance to mlx_whisper. Words are no longer cut at
                          arbitrary timer boundaries.

Outputs:
    <audio>.mlx.txt           plain transcript
    <audio>.mlx.srt           SubRip subtitles (unless --no-srt)
    mic_YYYYMMDD_HHMMSS.wav   the captured audio (--mic mode)
    mic_YYYYMMDD_HHMMSS_session.wav + .mlx.txt   the captured session (--continuous)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
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


def record_mic(duration: float | None, sample_rate: int = 16000) -> Path:
    """Record from the default input device. Returns the path to a 16 kHz mono WAV.

    If duration is None, records until the user presses Enter.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        sys.exit("[abort] sounddevice not installed. Run: pip install sounddevice")

    import wave

    try:
        dev = sd.query_devices(kind="input")
        print(f"[mic] input device: {dev['name']}")
    except Exception:
        pass

    if duration is not None:
        print(f"[mic] recording {duration:.1f}s ... (speak now)")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
    else:
        print("[mic] recording ... press Enter to stop.")
        chunks: list = []

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                print(f"[mic] {status}", file=sys.stderr)
            chunks.append(indata.copy())

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=callback):
            try:
                input()
            except KeyboardInterrupt:
                print("\n[mic] interrupted; using what was captured so far.")
        if not chunks:
            sys.exit("[abort] no audio captured.")
        audio = np.concatenate(chunks, axis=0)

    seconds = len(audio) / sample_rate
    print(f"[mic] captured {seconds:.2f}s")

    out = Path(f"mic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav").resolve()
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"[mic] saved {out}")
    return out


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


def transcribe_continuous(
    model_id: str,
    language: str,
    chunk_seconds: float,
    sample_rate: int = 16000,
) -> int:
    """Record from the default mic continuously, transcribing each chunk as it fills.

    Stops on Ctrl-C. At exit, writes the full session audio and concatenated
    transcript next to the working directory.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        sys.exit("[abort] sounddevice not installed. Run: pip install sounddevice")

    import queue as queue_mod
    import wave

    try:
        import mlx_whisper
    except ImportError:
        sys.exit("[abort] mlx_whisper not installed. Run: pip install -U mlx-whisper")

    try:
        dev = sd.query_devices(kind="input")
        print(f"[mic] input device: {dev['name']}")
    except Exception:
        pass

    print(f"[continuous] warming up model ({model_id}) ...")
    mlx_whisper.transcribe(
        np.zeros(int(0.5 * sample_rate), dtype=np.float32),
        path_or_hf_repo=model_id,
        language=language,
        verbose=False,
    )
    print(f"[continuous] chunk={chunk_seconds:.1f}s, lang={language}. Press Ctrl-C to stop.\n")

    chunk_samples = int(chunk_seconds * sample_rate)
    q: queue_mod.Queue = queue_mod.Queue()
    session_chunks: list = []
    transcripts: list[str] = []

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        q.put(indata.copy())

    buffer = np.zeros((0,), dtype=np.int16)

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=callback):
            while True:
                try:
                    block = q.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                flat = block.flatten()
                buffer = np.concatenate([buffer, flat])
                session_chunks.append(flat)

                while len(buffer) >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]

                    audio_f32 = chunk.astype(np.float32) / 32768.0
                    t0 = time.perf_counter()
                    result = mlx_whisper.transcribe(
                        audio_f32,
                        path_or_hf_repo=model_id,
                        language=language,
                        verbose=False,
                    )
                    elapsed = time.perf_counter() - t0
                    text = (result.get("text") or "").strip()

                    ts = datetime.now().strftime("%H:%M:%S")
                    xrt = chunk_seconds / elapsed if elapsed > 0 else float("inf")
                    marker = text if text else "<silence>"
                    print(f"[{ts}] ({elapsed:.2f}s, {xrt:.1f}x) {marker}", flush=True)
                    if text:
                        transcripts.append(text)

                    # Warn if transcription is falling behind real-time recording.
                    backlog = q.qsize()
                    if backlog > 5:
                        print(f"[continuous] warning: backlog={backlog} blocks "
                              f"(transcription slower than real-time)", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[continuous] stopping ...")

    if not session_chunks:
        print("[continuous] no audio captured.")
        return 0

    full = np.concatenate(session_chunks)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_out = Path(f"mic_{timestamp}_session.wav").resolve()
    with wave.open(str(wav_out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full.tobytes())
    print(f"[continuous] saved {len(full) / sample_rate:.1f}s of audio to {wav_out}")

    if transcripts:
        txt_out = wav_out.with_suffix(".mlx.txt")
        txt_out.write_text("\n".join(transcripts) + "\n", encoding="utf-8")
        print(f"[continuous] saved transcript to {txt_out}")

    return 0


def transcribe_continuous_vad(
    model_id: str,
    language: str,
    aggressiveness: int,
    silence_ms: int,
    min_utterance_ms: int = 300,
    max_utterance_ms: int = 25_000,
    preroll_ms: int = 300,
    sample_rate: int = 16000,
) -> int:
    """Continuous transcription with webrtcvad-based utterance segmentation.

    Pipeline: sounddevice writes variable-size int16 blocks into a queue.
    A consumer slices 30 ms VAD frames out of an internal buffer and feeds
    them through a small state machine:
        idle -> on a speech frame, flush a ring buffer of recent audio into
                the utterance buffer (pre-roll) and switch to in-speech.
        in-speech -> append every frame; count consecutive silence frames.
                When silence >= silence_ms or utterance >= max_utterance_ms,
                hand the buffer to mlx_whisper and go back to idle.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        sys.exit("[abort] sounddevice not installed. Run: pip install sounddevice")

    try:
        import webrtcvad
    except ImportError:
        sys.exit("[abort] webrtcvad not installed. Run: pip install webrtcvad")

    import collections
    import queue as queue_mod
    import wave

    try:
        import mlx_whisper
    except ImportError:
        sys.exit("[abort] mlx_whisper not installed. Run: pip install -U mlx-whisper")

    if aggressiveness not in (0, 1, 2, 3):
        sys.exit("[abort] --vad-aggressiveness must be 0, 1, 2, or 3.")
    if sample_rate not in (8000, 16000, 32000, 48000):
        sys.exit(f"[abort] webrtcvad does not support sample rate {sample_rate}.")

    vad = webrtcvad.Vad(aggressiveness)

    frame_ms = 30
    frame_samples = sample_rate * frame_ms // 1000  # 480 at 16 kHz
    silence_frames_needed = max(1, silence_ms // frame_ms)
    min_utterance_frames = max(1, min_utterance_ms // frame_ms)
    max_utterance_frames = max_utterance_ms // frame_ms
    preroll_frames = max(1, preroll_ms // frame_ms)

    try:
        dev = sd.query_devices(kind="input")
        print(f"[mic] input device: {dev['name']}")
    except Exception:
        pass

    print(f"[vad] warming up model ({model_id}) ...")
    mlx_whisper.transcribe(
        np.zeros(int(0.5 * sample_rate), dtype=np.float32),
        path_or_hf_repo=model_id,
        language=language,
        verbose=False,
    )
    print(
        f"[vad] aggressiveness={aggressiveness}, "
        f"silence={silence_ms}ms, min={min_utterance_ms}ms, "
        f"max={max_utterance_ms}ms, preroll={preroll_ms}ms. "
        f"Press Ctrl-C to stop.\n"
    )

    q: queue_mod.Queue = queue_mod.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        q.put(indata.copy().flatten())

    buffer = np.zeros((0,), dtype=np.int16)
    session_chunks: list = []
    transcripts: list[str] = []

    preroll: collections.deque = collections.deque(maxlen=preroll_frames)
    in_speech = False
    speech_frames: list = []
    silence_count = 0

    def finalize_utterance() -> None:
        """Transcribe the accumulated speech_frames if it's long enough."""
        nonlocal in_speech, speech_frames, silence_count
        if len(speech_frames) < min_utterance_frames:
            in_speech = False
            speech_frames = []
            silence_count = 0
            return
        utterance = np.concatenate(speech_frames)
        utterance_seconds = len(utterance) / sample_rate

        audio_f32 = utterance.astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=model_id,
            language=language,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        text = (result.get("text") or "").strip()

        ts = datetime.now().strftime("%H:%M:%S")
        xrt = utterance_seconds / elapsed if elapsed > 0 else float("inf")
        if text:
            print(f"[{ts}] ({utterance_seconds:.2f}s spoken, {elapsed:.2f}s decode, {xrt:.1f}x) {text}",
                  flush=True)
            transcripts.append(text)
        else:
            print(f"[{ts}] ({utterance_seconds:.2f}s spoken, {elapsed:.2f}s decode) <no transcription>",
                  flush=True)

        in_speech = False
        speech_frames = []
        silence_count = 0

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", callback=callback):
            while True:
                try:
                    block = q.get(timeout=0.5)
                except queue_mod.Empty:
                    continue
                session_chunks.append(block)
                buffer = np.concatenate([buffer, block])

                # Pull complete 30 ms VAD frames off the buffer.
                while len(buffer) >= frame_samples:
                    frame = buffer[:frame_samples]
                    buffer = buffer[frame_samples:]
                    is_speech = vad.is_speech(frame.tobytes(), sample_rate)

                    if in_speech:
                        speech_frames.append(frame)
                        silence_count = 0 if is_speech else silence_count + 1

                        if silence_count >= silence_frames_needed:
                            # Trim trailing silence so xRT reflects spoken audio.
                            trim = silence_count - 1  # keep ~30ms of trailing silence as breathing room
                            if trim > 0:
                                speech_frames = speech_frames[:-trim]
                            finalize_utterance()
                        elif len(speech_frames) >= max_utterance_frames:
                            print(f"[vad] hit max utterance ({max_utterance_ms}ms); forcing flush",
                                  file=sys.stderr)
                            finalize_utterance()
                    else:
                        preroll.append(frame)
                        if is_speech:
                            in_speech = True
                            speech_frames = list(preroll)  # prepend recent buffered audio
                            silence_count = 0

                backlog = q.qsize()
                if backlog > 50:
                    print(f"[vad] warning: backlog={backlog} blocks", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[vad] stopping ...")
        # Flush any in-flight utterance so a long final sentence isn't lost.
        if in_speech and len(speech_frames) >= min_utterance_frames:
            finalize_utterance()

    if not session_chunks:
        print("[vad] no audio captured.")
        return 0

    full = np.concatenate(session_chunks)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_out = Path(f"mic_{timestamp}_session.wav").resolve()
    with wave.open(str(wav_out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full.tobytes())
    print(f"[vad] saved {len(full) / sample_rate:.1f}s of audio to {wav_out}")

    if transcripts:
        txt_out = wav_out.with_suffix(".mlx.txt")
        txt_out.write_text("\n".join(transcripts) + "\n", encoding="utf-8")
        print(f"[vad] saved transcript to {txt_out}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="?", help="audio file path (default: bundled test.m4a)")
    parser.add_argument("--mic", action="store_true", help="record from the default input device instead of reading a file")
    parser.add_argument("--duration", type=float, default=None,
                        help="mic recording length in seconds (default: record until Enter)")
    parser.add_argument("--continuous", action="store_true",
                        help="record from mic and transcribe chunk-by-chunk live until Ctrl-C")
    parser.add_argument("--chunk", type=float, default=5.0,
                        help="chunk size in seconds for --continuous mode (default: 5.0; ignored with --vad)")
    parser.add_argument("--vad", action="store_true",
                        help="(with --continuous) use webrtcvad to segment by utterance instead of fixed chunks")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                        help="webrtcvad aggressiveness (0=permissive .. 3=aggressive; default: 2)")
    parser.add_argument("--vad-silence-ms", type=int, default=500,
                        help="trailing silence in ms that closes a VAD utterance (default: 500)")
    parser.add_argument("--fp16", action="store_true", help=f"use {MODEL_FP16} instead of the 4-bit model")
    parser.add_argument("--language", default="zh", help="language code passed to whisper (default: zh)")
    parser.add_argument("--word-timestamps", action="store_true", help="emit per-word timestamps")
    parser.add_argument("--no-srt", action="store_true", help="skip writing the .srt file")
    args = parser.parse_args()

    if args.audio and (args.mic or args.continuous):
        sys.exit("[abort] pass either a positional audio path or --mic / --continuous, not both.")
    if args.mic and args.continuous:
        sys.exit("[abort] --continuous already records from mic; don't pass --mic too.")
    if args.duration is not None and not args.mic:
        sys.exit("[abort] --duration only makes sense with --mic. Use --chunk for --continuous.")
    if args.chunk != 5.0 and not args.continuous:
        sys.exit("[abort] --chunk only makes sense with --continuous.")
    if args.chunk <= 0:
        sys.exit("[abort] --chunk must be positive.")
    if args.vad and not args.continuous:
        sys.exit("[abort] --vad only makes sense with --continuous.")
    if args.vad_silence_ms <= 0:
        sys.exit("[abort] --vad-silence-ms must be positive.")

    assert_apple_silicon()

    model_id = MODEL_FP16 if args.fp16 else MODEL_4BIT

    if args.continuous:
        if args.vad:
            return transcribe_continuous_vad(
                model_id,
                args.language,
                aggressiveness=args.vad_aggressiveness,
                silence_ms=args.vad_silence_ms,
            )
        return transcribe_continuous(model_id, args.language, args.chunk)

    try:
        import mlx_whisper
    except ImportError:
        sys.exit("[abort] mlx_whisper not installed. Run: pip install -U mlx-whisper")

    audio_path = record_mic(args.duration) if args.mic else resolve_audio(args.audio)

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
