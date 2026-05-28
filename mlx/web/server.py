#!/usr/bin/env python3
"""
Breeze-ASR-26 live web transcription — FastAPI + WebSocket.

The browser captures mic audio at 16 kHz, sends raw int16 PCM frames over
WebSocket, and this server runs the same webrtcvad utterance-segmentation
pipeline as `mlx/test_mlx.py --continuous --vad`, then transcribes each
finalised utterance with mlx-whisper and streams the result back as JSON.

Runtime: Apple Silicon (MLX is Metal-only).

Install:
    pip install fastapi 'uvicorn[standard]' webrtcvad mlx-whisper numpy

Run:
    python server.py                           # 127.0.0.1:8000
    python server.py --host 0.0.0.0 --port 8080
    python server.py --fp16                    # full-precision MLX model
    python server.py --vad-silence-ms 800
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import platform
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np


MODEL_4BIT = "doggy8088/Breeze-ASR-26-MLX-4bit"
MODEL_FP16 = "doggy8088/Breeze-ASR-26-MLX"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480
MIN_UTTERANCE_MS = 300
MAX_UTTERANCE_MS = 25_000
PREROLL_MS = 300

# Filled in by main() before uvicorn starts. The handlers read these at
# request time, so mutating the dict from main() before uvicorn.run() works.
CONFIG: dict = {
    "model_id": MODEL_4BIT,
    "language": "zh",
    "vad_aggressiveness": 2,
    "vad_silence_ms": 500,
}

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"


def assert_apple_silicon() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit(
            f"[abort] MLX requires Apple Silicon (darwin/arm64); "
            f"detected {sys.platform}/{platform.machine()}."
        )


try:
    import mlx_whisper
except ImportError:
    sys.exit("[abort] mlx_whisper not installed. Run: pip install -U mlx-whisper")

try:
    import webrtcvad
except ImportError:
    sys.exit("[abort] webrtcvad not installed. Run: pip install webrtcvad")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
except ImportError:
    sys.exit("[abort] fastapi not installed. Run: pip install fastapi 'uvicorn[standard]'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] warming up {CONFIG['model_id']} ...")
    t0 = time.perf_counter()
    await asyncio.to_thread(
        mlx_whisper.transcribe,
        np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32),
        path_or_hf_repo=CONFIG["model_id"],
        language=CONFIG["language"],
        verbose=False,
    )
    print(f"[startup] model loaded in {time.perf_counter() - t0:.1f}s. Open http://{CONFIG['host']}:{CONFIG['port']}")
    yield
    print("[shutdown]")


app = FastAPI(lifespan=lifespan, title="Breeze-ASR-26 Live")


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": CONFIG["model_id"], "language": CONFIG["language"]}


@app.websocket("/ws")
async def transcribe_ws(ws: WebSocket):
    """One VAD pipeline + transcription worker per connection."""
    await ws.accept()

    vad = webrtcvad.Vad(CONFIG["vad_aggressiveness"])
    silence_frames_needed = max(1, CONFIG["vad_silence_ms"] // FRAME_MS)
    min_utterance_frames = max(1, MIN_UTTERANCE_MS // FRAME_MS)
    max_utterance_frames = MAX_UTTERANCE_MS // FRAME_MS
    preroll_frames = max(1, PREROLL_MS // FRAME_MS)

    buffer = np.zeros((0,), dtype=np.int16)
    preroll: collections.deque = collections.deque(maxlen=preroll_frames)
    in_speech = False
    speech_frames: list = []
    silence_count = 0

    utterance_queue: asyncio.Queue = asyncio.Queue()

    async def safe_send_json(payload: dict) -> bool:
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            return False

    async def transcriber():
        """Pull finalised utterances off the queue and run MLX transcription in a thread."""
        while True:
            utterance = await utterance_queue.get()
            if utterance is None:
                return
            try:
                audio_f32 = utterance.astype(np.float32) / 32768.0
                spoken_s = len(utterance) / SAMPLE_RATE
                t0 = time.perf_counter()
                result = await asyncio.to_thread(
                    mlx_whisper.transcribe,
                    audio_f32,
                    path_or_hf_repo=CONFIG["model_id"],
                    language=CONFIG["language"],
                    verbose=False,
                )
                elapsed = time.perf_counter() - t0
                text = (result.get("text") or "").strip()
                xrt = spoken_s / elapsed if elapsed > 0 else 0.0
                if not await safe_send_json({
                    "type": "utterance",
                    "text": text,
                    "spoken_s": round(spoken_s, 2),
                    "decode_s": round(elapsed, 2),
                    "xrt": round(xrt, 1),
                    "ts": datetime.now().strftime("%H:%M:%S"),
                }):
                    return
            except Exception as e:
                await safe_send_json({"type": "error", "message": str(e)})

    transcriber_task = asyncio.create_task(transcriber())

    await safe_send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "model": CONFIG["model_id"],
        "language": CONFIG["language"],
        "vad_aggressiveness": CONFIG["vad_aggressiveness"],
        "vad_silence_ms": CONFIG["vad_silence_ms"],
    })

    try:
        while True:
            data = await ws.receive_bytes()
            if not data:
                continue
            new_samples = np.frombuffer(data, dtype=np.int16)
            buffer = np.concatenate([buffer, new_samples])

            while len(buffer) >= FRAME_SAMPLES:
                frame = buffer[:FRAME_SAMPLES].copy()
                buffer = buffer[FRAME_SAMPLES:]
                is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)

                if in_speech:
                    speech_frames.append(frame)
                    silence_count = 0 if is_speech else silence_count + 1

                    if silence_count >= silence_frames_needed:
                        trim = silence_count - 1  # keep ~30 ms of trailing silence
                        finalize = speech_frames[:-trim] if trim > 0 else speech_frames
                        if len(finalize) >= min_utterance_frames:
                            await utterance_queue.put(np.concatenate(finalize))
                        await safe_send_json({"type": "speaking", "on": False})
                        in_speech = False
                        speech_frames = []
                        silence_count = 0
                    elif len(speech_frames) >= max_utterance_frames:
                        await utterance_queue.put(np.concatenate(speech_frames))
                        await safe_send_json({"type": "info", "message": f"hit max-utterance cap ({MAX_UTTERANCE_MS}ms); flushed"})
                        in_speech = False
                        speech_frames = []
                        silence_count = 0
                else:
                    preroll.append(frame)
                    if is_speech:
                        in_speech = True
                        speech_frames = list(preroll)
                        silence_count = 0
                        await safe_send_json({"type": "speaking", "on": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] unexpected error: {e}", file=sys.stderr)
    finally:
        # Flush any in-flight utterance so the last sentence isn't lost.
        if in_speech and len(speech_frames) >= min_utterance_frames:
            await utterance_queue.put(np.concatenate(speech_frames))
        await utterance_queue.put(None)
        try:
            await asyncio.wait_for(transcriber_task, timeout=30.0)
        except asyncio.TimeoutError:
            transcriber_task.cancel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument("--fp16", action="store_true", help=f"use {MODEL_FP16} instead of the 4-bit model")
    parser.add_argument("--language", default="zh", help="language code passed to whisper (default: zh)")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                        help="webrtcvad aggressiveness (default: 2)")
    parser.add_argument("--vad-silence-ms", type=int, default=500,
                        help="trailing silence in ms that closes a VAD utterance (default: 500)")
    args = parser.parse_args()

    if args.vad_silence_ms <= 0:
        sys.exit("[abort] --vad-silence-ms must be positive.")

    assert_apple_silicon()

    CONFIG["model_id"] = MODEL_FP16 if args.fp16 else MODEL_4BIT
    CONFIG["language"] = args.language
    CONFIG["vad_aggressiveness"] = args.vad_aggressiveness
    CONFIG["vad_silence_ms"] = args.vad_silence_ms
    CONFIG["host"] = args.host
    CONFIG["port"] = args.port

    if not INDEX_HTML.exists():
        sys.exit(f"[abort] index.html not found next to server.py: {INDEX_HTML}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
