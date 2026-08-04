#!/usr/bin/env python3
"""
Breeze-ASR-26 live web transcription — whisper.cpp backend.

The browser captures mic audio at 16 kHz and streams raw int16 PCM over a
WebSocket. This server segments it into utterances with webrtcvad, wraps each
finished utterance as a WAV, and POSTs it to a `whisper-server` process holding
the ggml model resident. Transcripts stream back as JSON.

Why a separate whisper-server instead of shelling out to whisper-cli: whisper-cli
reloads the model on every invocation (~0.9 s for f16), which is fatal for a
real-time feel. whisper-server keeps it loaded and adds ~0 startup per utterance.

This server spawns and supervises whisper-server itself, so one command is enough.

Runtime: Apple Silicon (uses the Metal build produced by wcpp/convert.sh).

Install:
    pip install fastapi 'uvicorn[standard]' webrtcvad httpx numpy

Run:
    python server.py                          # 127.0.0.1:8000, f16 model
    python server.py --model q8_0             # pick a quantization
    python server.py --host 0.0.0.0 --port 8080
    python server.py --asr-url http://127.0.0.1:8080   # attach to an existing whisper-server
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import platform
import socket
import subprocess
import sys
import time
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
MIN_UTTERANCE_MS = 300
MAX_UTTERANCE_MS = 25_000  # Whisper degrades past its 30 s window
PREROLL_MS = 300

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"
OUT_DIR = HERE.parent / "out"
WHISPER_SERVER = HERE.parent / "whisper.cpp" / "build" / "bin" / "whisper-server"

# ggml builds produced by convert.sh, in descending fidelity. Label -> filename.
MODEL_FILES = {
    "f16": "ggml-model.bin",
    "q8_0": "ggml-breeze-q8_0.bin",
    "q5_k": "ggml-breeze-q5_k.bin",
    "q4_k": "ggml-breeze-q4_k.bin",
}

CONFIG: dict = {}

try:
    import webrtcvad
except ImportError:
    sys.exit("[abort] webrtcvad not installed. Run: pip install webrtcvad")

try:
    import httpx
except ImportError:
    sys.exit("[abort] httpx not installed. Run: pip install httpx")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:
    sys.exit("[abort] fastapi not installed. Run: pip install fastapi 'uvicorn[standard]'")


# --------------------------------------------------------------------- helpers

def available_models() -> dict[str, Path]:
    """Which ggml builds actually exist on disk, in fidelity order."""
    return {
        label: OUT_DIR / fname
        for label, fname in MODEL_FILES.items()
        if (OUT_DIR / fname).is_file()
    }


def pcm_to_wav(pcm: np.ndarray) -> bytes:
    """int16 mono @16 kHz -> WAV container. whisper-server wants a real WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_whisper_server(model: Path, port: int, threads: int,
                         beam_size: int, suppress_nst: bool) -> subprocess.Popen:
    if not WHISPER_SERVER.is_file():
        sys.exit(
            f"[abort] whisper-server not found at {WHISPER_SERVER}\n"
            f"        Build it first:  ./wcpp/convert.sh"
        )
    cmd = [
        str(WHISPER_SERVER),
        "-m", str(model),
        "-l", CONFIG["language"],
        "-t", str(threads),
        # Beam search matters here. Measured in issue #8: with greedy decoding this
        # model falls into repetition loops on hard Taigi audio, and beam search
        # recovered 3/3 such clips. whisper-server's own default is -bs -1 (off),
        # so it has to be set explicitly.
        "-bs", str(beam_size),
        "-bo", str(beam_size),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-nt",
    ]
    if suppress_nst:
        # Drops "(sound of ...)"-style non-speech tags. Off by default so the app
        # shows what the model actually emits.
        cmd.append("-sns")
    print(f"[asr] spawning: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


async def wait_until_up(url: str, proc: subprocess.Popen | None, timeout: float = 180.0) -> None:
    """Poll until whisper-server answers, failing fast if it died on the way."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                sys.exit(f"[abort] whisper-server exited early (code {proc.returncode})")
            try:
                await client.get(url + "/")
                return
            except Exception:
                await asyncio.sleep(0.5)
    sys.exit(f"[abort] whisper-server did not become ready within {timeout:.0f}s")


async def transcribe(pcm: np.ndarray) -> str:
    """POST one utterance as WAV; return the transcript text."""
    files = {"file": ("utterance.wav", pcm_to_wav(pcm), "audio/wav")}
    data = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
    }
    async with httpx.AsyncClient(timeout=CONFIG["asr_timeout"]) as client:
        r = await client.post(CONFIG["asr_url"] + "/inference", files=files, data=data)
        r.raise_for_status()
        try:
            return (r.json().get("text") or "").strip()
        except Exception:
            return r.text.strip()


# ------------------------------------------------------------------ lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    proc = CONFIG.get("asr_proc")
    print(f"[startup] waiting for whisper-server at {CONFIG['asr_url']} ...")
    t0 = time.perf_counter()
    await wait_until_up(CONFIG["asr_url"], proc)
    print(f"[startup] ASR ready in {time.perf_counter() - t0:.1f}s")

    # Warm the model with 0.5 s of silence so the first real utterance isn't
    # charged for lazy Metal/graph setup.
    try:
        await transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.int16))
    except Exception as e:
        print(f"[startup] warmup failed (continuing): {e}", file=sys.stderr)

    print(f"[startup] open http://{CONFIG['host']}:{CONFIG['port']}")
    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            print("[shutdown] stopping whisper-server")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


app = FastAPI(lifespan=lifespan, title="Breeze-ASR-26 Live (whisper.cpp)")


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "engine": "whisper.cpp",
        "model": CONFIG["model_label"],
        "language": CONFIG["language"],
        "asr_url": CONFIG["asr_url"],
    }


@app.get("/models")
async def models():
    on_disk = available_models()
    return {
        "current": CONFIG["model_label"],
        "available": [
            {"label": label, "size_mb": round(p.stat().st_size / 1048576)}
            for label, p in on_disk.items()
        ],
        # /load takes a filesystem path, so hot-swapping only makes sense while the
        # ASR process shares this machine's filesystem.
        "switchable": len(on_disk) > 1,
    }


@app.post("/models/{label}")
async def switch_model(label: str):
    """Hot-swap the ggml build via whisper-server's /load endpoint."""
    models_on_disk = available_models()
    if label not in models_on_disk:
        return JSONResponse({"error": f"unknown model '{label}'"}, status_code=404)
    path = models_on_disk[label]
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=CONFIG["asr_timeout"]) as client:
            # /load insists on multipart/form-data. Passing this through `files`
            # with a None filename emits `model` as a plain multipart field —
            # `data=` would send urlencoded and whisper-server answers 400.
            r = await client.post(
                CONFIG["asr_url"] + "/load", files={"model": (None, str(path))}
            )
            r.raise_for_status()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    CONFIG["model_label"] = label
    return {"ok": True, "model": label, "load_s": round(time.perf_counter() - t0, 2)}


# ----------------------------------------------------------------- websocket

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
        while True:
            utterance = await utterance_queue.get()
            if utterance is None:
                return
            try:
                spoken_s = len(utterance) / SAMPLE_RATE
                t0 = time.perf_counter()
                text = await transcribe(utterance)
                elapsed = time.perf_counter() - t0
                if not await safe_send_json({
                    "type": "utterance",
                    "text": text,
                    "model": CONFIG["model_label"],
                    "spoken_s": round(spoken_s, 2),
                    "decode_s": round(elapsed, 2),
                    "xrt": round(spoken_s / elapsed, 1) if elapsed > 0 else 0.0,
                    "ts": datetime.now().strftime("%H:%M:%S"),
                }):
                    return
            except Exception as e:
                await safe_send_json({"type": "error", "message": str(e)})

    transcriber_task = asyncio.create_task(transcriber())

    await safe_send_json({
        "type": "ready",
        "engine": "whisper.cpp",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "model": CONFIG["model_label"],
        "language": CONFIG["language"],
        "beam_size": CONFIG["beam_size"],
        "vad_aggressiveness": CONFIG["vad_aggressiveness"],
        "vad_silence_ms": CONFIG["vad_silence_ms"],
    })

    try:
        while True:
            data = await ws.receive_bytes()
            if not data:
                continue
            buffer = np.concatenate([buffer, np.frombuffer(data, dtype=np.int16)])

            while len(buffer) >= FRAME_SAMPLES:
                frame = buffer[:FRAME_SAMPLES].copy()
                buffer = buffer[FRAME_SAMPLES:]
                is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)

                if in_speech:
                    speech_frames.append(frame)
                    silence_count = 0 if is_speech else silence_count + 1

                    if silence_count >= silence_frames_needed:
                        trim = silence_count - 1  # keep ~30 ms trailing silence
                        finalize = speech_frames[:-trim] if trim > 0 else speech_frames
                        if len(finalize) >= min_utterance_frames:
                            await utterance_queue.put(np.concatenate(finalize))
                        await safe_send_json({"type": "speaking", "on": False})
                        in_speech = False
                        speech_frames = []
                        silence_count = 0
                    elif len(speech_frames) >= max_utterance_frames:
                        await utterance_queue.put(np.concatenate(speech_frames))
                        await safe_send_json({
                            "type": "info",
                            "message": f"hit max-utterance cap ({MAX_UTTERANCE_MS} ms); flushed",
                        })
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
        if in_speech and len(speech_frames) >= min_utterance_frames:
            await utterance_queue.put(np.concatenate(speech_frames))
        await utterance_queue.put(None)
        try:
            await asyncio.wait_for(transcriber_task, timeout=60.0)
        except asyncio.TimeoutError:
            transcriber_task.cancel()


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="f16", choices=list(MODEL_FILES),
                    help="ggml build to start with (default: f16)")
    ap.add_argument("--language", default="zh",
                    help="Whisper large-v2 has no <|nan|> token, so Taigi decodes as zh")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5,
                    help="beam search width; 1 = greedy (loops more, see issue #8)")
    ap.add_argument("--suppress-nst", action="store_true",
                    help="drop non-speech tags like (music); off by default")
    ap.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    ap.add_argument("--vad-silence-ms", type=int, default=500,
                    help="trailing silence that closes an utterance (default: 500)")
    ap.add_argument("--asr-url", default=None,
                    help="attach to an already-running whisper-server instead of spawning one")
    ap.add_argument("--asr-timeout", type=float, default=180.0)
    args = ap.parse_args()

    if args.vad_silence_ms <= 0:
        sys.exit("[abort] --vad-silence-ms must be positive.")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("[warn] not Apple Silicon; the Metal build won't be available here.",
              file=sys.stderr)
    if not INDEX_HTML.exists():
        sys.exit(f"[abort] index.html not found next to server.py: {INDEX_HTML}")

    CONFIG.update({
        "host": args.host,
        "port": args.port,
        "language": args.language,
        "beam_size": args.beam_size,
        "vad_aggressiveness": args.vad_aggressiveness,
        "vad_silence_ms": args.vad_silence_ms,
        "asr_timeout": args.asr_timeout,
        "model_label": args.model,
        "asr_proc": None,
        "allow_switch": True,  # reserved
    })

    if args.asr_url:
        CONFIG["asr_url"] = args.asr_url.rstrip("/")
        CONFIG["model_label"] = args.model
        print(f"[asr] attaching to existing whisper-server at {CONFIG['asr_url']}")
    else:
        models_on_disk = available_models()
        if not models_on_disk:
            sys.exit(
                f"[abort] no ggml models found in {OUT_DIR}\n"
                f"        Build them first:  ./wcpp/convert.sh"
            )
        if args.model not in models_on_disk:
            sys.exit(
                f"[abort] model '{args.model}' not found at {OUT_DIR / MODEL_FILES[args.model]}\n"
                f"        Available: {', '.join(models_on_disk)}"
            )
        port = free_port()
        CONFIG["asr_url"] = f"http://127.0.0.1:{port}"
        CONFIG["asr_proc"] = spawn_whisper_server(
            models_on_disk[args.model], port, args.threads,
            args.beam_size, args.suppress_nst,
        )

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        proc = CONFIG.get("asr_proc")
        if proc is not None and proc.poll() is None:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
