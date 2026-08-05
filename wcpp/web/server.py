#!/usr/bin/env python3
"""
Breeze-ASR-26 live + batch web transcription — whisper.cpp backend.

Two input paths, one ASR backend:

  * realtime — browser captures mic at 16 kHz and streams int16 PCM over a
    WebSocket; this server segments it into utterances with webrtcvad, wraps each
    finished utterance as a WAV, and POSTs it to whisper-server.

  * batch — upload an audio/video file; ffmpeg transcodes it to 16 kHz mono WAV
    and the whole thing goes to whisper-server in one shot, which lets whisper.cpp
    do its own sliding window and carry decoder context across it. Runs as a job
    with progress over SSE.

Why a separate whisper-server instead of shelling out to whisper-cli: whisper-cli
reloads the model on every invocation (~0.9 s for f16), which is fatal for a
real-time feel. whisper-server keeps it loaded and adds ~0 startup per utterance.

whisper-server holds a single model context, so every inference call — mic or
batch — is serialized behind ASR_LOCK, FIFO. A long upload therefore delays live
utterances rather than corrupting them, and the UI surfaces the queue position.

This server spawns and supervises whisper-server itself, so one command is enough.

Runtime: Apple Silicon (uses the Metal build produced by wcpp/convert.sh).

Install:
    pip install fastapi 'uvicorn[standard]' webrtcvad httpx numpy python-multipart

Run:
    python server.py                          # 127.0.0.1:8000, f16 model
    python server.py --model q8_0             # pick a quantization
    python server.py --host 0.0.0.0 --port 8080
    python server.py --no-upload              # realtime only, no upload endpoint
    python server.py --asr-url http://127.0.0.1:8080   # attach to an existing whisper-server
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json as jsonlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
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

# whisper-server exposes no progress, so batch jobs get a predicted ETA instead of a
# fabricated percentage. Rough figures, measured on an M1 Max with beam 5 over 105 s,
# 228 s and 369 s of Taigi audio.
#
# Asking for segment timestamps (no_timestamps=false) costs ~5x, because whisper.cpp
# then predicts timestamp tokens while decoding. The remaining spread is driven by
# *content*, not duration: q4_k came in at xRT 0.26-0.36 on ordinary speech but 0.73
# on the clip that sends the model into a repetition loop, since a loop runs the
# decoder to its token cap. So these coefficients carry headroom, the UI labels the
# number a rough estimate, and the bar switches to indeterminate once elapsed passes
# it rather than sitting pinned at 100%.
XRT_TS_ON = {"f16": 0.57, "q8_0": 0.52, "q5_k": 0.52, "q4_k": 0.45}
XRT_TS_OFF = {"f16": 0.15, "q8_0": 0.14, "q5_k": 0.14, "q4_k": 0.12}
XRT_FALLBACK_ON = 0.65
XRT_FALLBACK_OFF = 0.18


def estimate_xrt(model: str, timestamps: bool) -> float:
    if timestamps:
        return XRT_TS_ON.get(model, XRT_FALLBACK_ON)
    return XRT_TS_OFF.get(model, XRT_FALLBACK_OFF)

JOB_TTL_S = 3600.0  # forget finished jobs after an hour so memory doesn't creep

CONFIG: dict = {}

# Every call into whisper-server goes through this. One model context upstream
# means concurrency here would just queue in the kernel with no fairness.
ASR_LOCK: asyncio.Lock | None = None
JOBS: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
JOB_QUEUE: asyncio.Queue | None = None

try:
    import webrtcvad
except ImportError:
    sys.exit("[abort] webrtcvad not installed. Run: pip install webrtcvad")

try:
    import httpx
except ImportError:
    sys.exit("[abort] httpx not installed. Run: pip install httpx")

try:
    from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                         WebSocket, WebSocketDisconnect)
    from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                                   StreamingResponse)
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


async def _infer(wav_bytes: bytes, extra: dict) -> dict | str:
    """POST one WAV to whisper-server. Serialized: one model context upstream."""
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"temperature": "0.0", "temperature_inc": "0.2", **extra}
    assert ASR_LOCK is not None
    async with ASR_LOCK:
        async with httpx.AsyncClient(timeout=CONFIG["asr_timeout"]) as client:
            r = await client.post(CONFIG["asr_url"] + "/inference", files=files, data=data)
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return r.text


async def transcribe(pcm: np.ndarray) -> str:
    """Realtime path: one utterance in, plain text out."""
    res = await _infer(pcm_to_wav(pcm), {"response_format": "json"})
    if isinstance(res, dict):
        return (res.get("text") or "").strip()
    return res.strip()


async def transcribe_wav_file(path: Path, timestamps: bool = True) -> dict:
    """Batch path: a whole WAV in, verbose_json out.

    The server is spawned with -nt for the realtime path, but no_timestamps is
    overridable per request — without flipping it off here, verbose_json comes back
    with no start/end on the segments.
    """
    params = {
        "response_format": "verbose_json",
        # language-probability estimation is a whole extra pass we never read;
        # measured at ~0.45 s even on a 6 s clip.
        "no_language_probabilities": "true",
    }
    if timestamps:
        params["no_timestamps"] = "false"
        # Must be sent explicitly: whisper-server defaults token_timestamps to
        # !no_timestamps, so asking for segment times would switch on the per-token
        # pass as well. Measured no benefit here, and it isn't free.
        params["token_timestamps"] = "false"
    res = await _infer(path.read_bytes(), params)
    if not isinstance(res, dict):
        raise RuntimeError(f"unexpected non-JSON reply: {str(res)[:300]}")
    return res


# ------------------------------------------------------------------ transcoding

def ffprobe_audio(path: Path) -> tuple[float, str]:
    """(duration_s, codec) for the first audio stream. Raises if there isn't one.

    Deliberately probes the container rather than trusting the filename or the
    client-supplied content type.
    """
    if not CONFIG.get("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH; cannot validate uploads")
    out = subprocess.run(
        [CONFIG["ffprobe"], "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise ValueError("not a media file ffmpeg can read")
    try:
        meta = jsonlib.loads(out.stdout)
    except Exception:
        raise ValueError("could not parse media metadata")
    audio = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise ValueError("file contains no audio stream")
    dur = float((meta.get("format") or {}).get("duration") or 0.0)
    if dur <= 0:
        raise ValueError("could not determine audio duration")
    return dur, audio[0].get("codec_name") or "?"


def transcode_to_wav(src: Path, dst: Path) -> None:
    """Anything ffmpeg reads -> 16 kHz mono s16 WAV, which is what Whisper wants."""
    if not CONFIG.get("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH; cannot transcode uploads")
    out = subprocess.run(
        [CONFIG["ffmpeg"], "-nostdin", "-loglevel", "error", "-y",
         "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", "-f", "wav", str(dst)],
        capture_output=True, text=True, timeout=1800,
    )
    if out.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed: {(out.stderr or '')[-400:]}")


# ------------------------------------------------------------- transcript export

def _ts(seconds: float, comma: bool = True) -> str:
    if seconds < 0:
        seconds = 0.0
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding carry
        ms = 0
        s += 1
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def segments_to_vtt(segments: list[dict]) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        out.append(f"{_ts(seg['start'], comma=False)} --> {_ts(seg['end'], comma=False)}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def segments_to_txt(segments: list[dict]) -> str:
    return "\n".join(s["text"].strip() for s in segments if s["text"].strip())


# ----------------------------------------------------------------------- jobs

def job_public(job: dict) -> dict:
    """The client-visible view of a job."""
    out = {
        k: job[k] for k in
        ("id", "filename", "status", "model", "language", "duration_s",
         "eta_s", "error", "created_at", "timestamps")
    }
    out["queue_position"] = queue_position(job)
    started = job.get("started_at")
    if started is not None:
        end = job.get("finished_at") or time.monotonic()
        out["elapsed_s"] = round(end - started, 1)
    else:
        out["elapsed_s"] = None
    if job["status"] == "done":
        out["segments"] = job["segments"]
        out["text"] = job["text"]
        out["decode_s"] = job["decode_s"]
        out["xrt"] = job["xrt"]
    return out


def queue_position(job: dict) -> int | None:
    """How many jobs are ahead of this one. 0 = next up, None = not waiting."""
    if job["status"] != "queued":
        return None
    ahead = 0
    for j in JOBS.values():
        if j is job:
            break
        if j["status"] in ("queued", "transcoding", "running"):
            ahead += 1
    return ahead


def prune_jobs() -> None:
    now = time.monotonic()
    stale = [
        jid for jid, j in JOBS.items()
        if j["status"] in ("done", "error", "cancelled")
        and now - (j.get("finished_at") or now) > JOB_TTL_S
    ]
    for jid in stale:
        JOBS.pop(jid, None)


def cleanup_job_files(job: dict) -> None:
    for key in ("src_path", "wav_path"):
        p = job.get(key)
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
            job[key] = None


async def job_worker() -> None:
    """Single consumer: keeps batch work FIFO and off the event loop."""
    assert JOB_QUEUE is not None
    while True:
        jid = await JOB_QUEUE.get()
        if jid is None:
            return
        job = JOBS.get(jid)
        if job is None or job["status"] == "cancelled":
            if job:
                cleanup_job_files(job)
            continue
        try:
            job["started_at"] = time.monotonic()
            job["status"] = "transcoding"
            wav = Path(job["src_path"]).with_suffix(".16k.wav")
            job["wav_path"] = str(wav)
            await asyncio.to_thread(transcode_to_wav, Path(job["src_path"]), wav)

            if job["status"] == "cancelled":
                cleanup_job_files(job)
                continue

            job["status"] = "running"
            t0 = time.monotonic()
            res = await transcribe_wav_file(wav, timestamps=job["timestamps"])
            decode_s = time.monotonic() - t0

            if job["status"] == "cancelled":
                cleanup_job_files(job)
                continue

            # Whisper always decodes a padded 30 s window, so the final segment's
            # end can run past the real audio (a 5.7 s file reported end=30.0).
            # Clamp, or every exported SRT/VTT cue for short audio is wrong.
            dur = job["duration_s"]
            segs = []
            for s in res.get("segments") or []:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                if not job["timestamps"]:
                    # whisper-server emits start=end=0 when timestamps are off; carry
                    # null rather than a fake 0:00–0:00 range the UI would render.
                    segs.append({"start": None, "end": None, "text": text})
                    continue
                start = min(max(0.0, float(s.get("start") or 0.0)), dur)
                end = min(max(start, float(s.get("end") or 0.0)), dur)
                if start >= dur:  # entirely inside the padding
                    continue
                segs.append({"start": start, "end": end, "text": text})
            job["segments"] = segs
            job["text"] = (res.get("text") or segments_to_txt(segs)).strip()
            job["decode_s"] = round(decode_s, 2)
            job["xrt"] = round(job["duration_s"] / decode_s, 1) if decode_s > 0 else None
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
        finally:
            job["finished_at"] = time.monotonic()
            cleanup_job_files(job)
            JOB_QUEUE.task_done() if hasattr(JOB_QUEUE, "task_done") else None


# ------------------------------------------------------------------ lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ASR_LOCK, JOB_QUEUE
    ASR_LOCK = asyncio.Lock()
    JOB_QUEUE = asyncio.Queue()

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

    worker = asyncio.create_task(job_worker()) if CONFIG["uploads"] else None
    if CONFIG["uploads"]:
        print(f"[startup] uploads enabled — max {CONFIG['max_upload_mb']} MB / "
              f"{CONFIG['max_duration_s']}s, queue {CONFIG['max_queue']}")
    else:
        print("[startup] uploads disabled (--no-upload)")

    print(f"[startup] open http://{CONFIG['host']}:{CONFIG['port']}")
    try:
        yield
    finally:
        if worker is not None:
            await JOB_QUEUE.put(None)
            try:
                await asyncio.wait_for(worker, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                worker.cancel()
        for job in JOBS.values():
            cleanup_job_files(job)
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
        "uploads": CONFIG["uploads"],
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
        assert ASR_LOCK is not None
        async with ASR_LOCK:  # don't swap the model out from under an in-flight decode
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


# ------------------------------------------------------------- batch endpoints

@app.get("/limits")
async def limits():
    return {
        "uploads": CONFIG["uploads"],
        "max_upload_mb": CONFIG["max_upload_mb"],
        "max_duration_s": CONFIG["max_duration_s"],
        "max_queue": CONFIG["max_queue"],
        "xrt_timestamps_on": XRT_TS_ON,
        "xrt_timestamps_off": XRT_TS_OFF,
    }


@app.post("/jobs")
async def create_job(file: UploadFile = File(...), model: str = Form(None),
                     language: str = Form(None), timestamps: str = Form("true")):
    if not CONFIG["uploads"]:
        raise HTTPException(status_code=403, detail="uploads are disabled (--no-upload)")

    prune_jobs()
    active = [j for j in JOBS.values()
              if j["status"] in ("queued", "transcoding", "running")]
    if len(active) >= CONFIG["max_queue"]:
        raise HTTPException(
            status_code=429,
            detail=f"queue full ({len(active)}/{CONFIG['max_queue']}); try again shortly",
        )

    if model and model not in available_models():
        raise HTTPException(status_code=400, detail=f"unknown model '{model}'")

    limit = CONFIG["max_upload_mb"] * 1024 * 1024
    tmpdir = Path(CONFIG["tmpdir"])
    tmpdir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload").suffix[:12] or ".bin"
    fd, src_name = tempfile.mkstemp(dir=tmpdir, suffix=suffix)
    src = Path(src_name)
    total = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds {CONFIG['max_upload_mb']} MB limit",
                    )
                fh.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        try:
            duration_s, codec = await asyncio.to_thread(ffprobe_audio, src)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        if duration_s > CONFIG["max_duration_s"]:
            raise HTTPException(
                status_code=413,
                detail=(f"audio is {duration_s:.0f}s; limit is "
                        f"{CONFIG['max_duration_s']}s"),
            )
    except HTTPException:
        src.unlink(missing_ok=True)
        raise
    except Exception as e:
        src.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")

    label = model or CONFIG["model_label"]
    want_ts = str(timestamps).lower() not in ("false", "0", "no", "off")
    xrt = estimate_xrt(label, want_ts)
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid,
        "filename": file.filename or "upload",
        "codec": codec,
        "status": "queued",
        "model": label,
        "language": language or CONFIG["language"],
        "duration_s": round(duration_s, 2),
        "bytes": total,
        "timestamps": want_ts,
        "eta_s": round(duration_s * xrt, 1),
        "src_path": str(src),
        "wav_path": None,
        "segments": [],
        "text": "",
        "decode_s": None,
        "xrt": None,
        "error": None,
        "created_at": datetime.now().strftime("%H:%M:%S"),
        "started_at": None,
        "finished_at": None,
    }
    JOBS[jid] = job

    # A per-job model differs from the loaded one only if the caller asked for it;
    # honour it by swapping before enqueueing so the queue stays FIFO-simple.
    if label != CONFIG["model_label"]:
        res = await switch_model(label)
        if isinstance(res, JSONResponse):
            job["status"] = "error"
            job["error"] = "could not switch model for this job"
            job["finished_at"] = time.monotonic()
            cleanup_job_files(job)
            raise HTTPException(status_code=502, detail=job["error"])

    assert JOB_QUEUE is not None
    await JOB_QUEUE.put(jid)
    return JSONResponse(job_public(job), status_code=202)


@app.get("/jobs")
async def list_jobs():
    prune_jobs()
    return {"jobs": [job_public(j) for j in JOBS.values()]}


@app.get("/jobs/{jid}")
async def get_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job_public(job)


@app.delete("/jobs/{jid}")
async def cancel_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    if job["status"] in ("done", "error", "cancelled"):
        return {"ok": False, "status": job["status"], "detail": "already finished"}
    was = job["status"]
    job["status"] = "cancelled"
    job["finished_at"] = time.monotonic()
    cleanup_job_files(job)
    # A request already in flight to whisper-server can't be interrupted; the result
    # is simply discarded when it lands.
    return {"ok": True, "was": was,
            "detail": "in-flight inference will finish but its result is discarded"
                      if was == "running" else "removed from queue"}


@app.get("/jobs/{jid}/events")
async def job_events(jid: str):
    """SSE: push job state until it reaches a terminal status."""
    if jid not in JOBS:
        raise HTTPException(status_code=404, detail="no such job")

    async def gen():
        last = None
        while True:
            job = JOBS.get(jid)
            if job is None:
                yield "event: gone\ndata: {}\n\n"
                return
            payload = jsonlib.dumps(job_public(job), ensure_ascii=False)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if job["status"] in ("done", "error", "cancelled"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs/{jid}/download")
async def download_job(jid: str, format: str = "txt"):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job is {job['status']}")

    stem = Path(job["filename"]).stem or "transcript"
    segs = job["segments"]
    if format in ("srt", "vtt") and not job.get("timestamps", True):
        raise HTTPException(
            status_code=409,
            detail="this job ran without timestamps; re-run with timestamps "
                   "enabled to export subtitles",
        )
    # Rendered from the stored segments rather than re-asking whisper-server for
    # srt/vtt, which would mean paying for inference twice.
    if format == "txt":
        body, media = segments_to_txt(segs) + "\n", "text/plain; charset=utf-8"
    elif format == "srt":
        body, media = segments_to_srt(segs), "application/x-subrip; charset=utf-8"
    elif format == "vtt":
        body, media = segments_to_vtt(segs), "text/vtt; charset=utf-8"
    elif format == "json":
        body = jsonlib.dumps(
            {
                "file": job["filename"], "model": job["model"],
                "language": job["language"], "duration_s": job["duration_s"],
                "decode_s": job["decode_s"], "xrt": job["xrt"],
                "text": job["text"], "segments": segs,
            },
            ensure_ascii=False, indent=2,
        )
        media = "application/json; charset=utf-8"
    else:
        raise HTTPException(status_code=400,
                            detail="format must be txt, srt, vtt or json")
    ext = format
    return PlainTextResponse(
        body, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'},
    )


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
    ap.add_argument("--asr-timeout", type=float, default=3600.0,
                    help="HTTP timeout for one inference call (default: 3600)")
    # batch upload
    ap.add_argument("--no-upload", action="store_true",
                    help="disable the upload/batch endpoints entirely")
    ap.add_argument("--max-upload-mb", type=int, default=100)
    ap.add_argument("--max-duration-s", type=int, default=1800,
                    help="reject audio longer than this (default: 1800 = 30 min)")
    ap.add_argument("--max-queue", type=int, default=3,
                    help="max simultaneously pending/running jobs before 429")
    ap.add_argument("--tmpdir", default=None,
                    help="scratch dir for uploads (default: system temp)")
    args = ap.parse_args()

    if args.vad_silence_ms <= 0:
        sys.exit("[abort] --vad-silence-ms must be positive.")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("[warn] not Apple Silicon; the Metal build won't be available here.",
              file=sys.stderr)
    if not INDEX_HTML.exists():
        sys.exit(f"[abort] index.html not found next to server.py: {INDEX_HTML}")

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    uploads = not args.no_upload
    if uploads and not (ffmpeg and ffprobe):
        print("[warn] ffmpeg/ffprobe not found on PATH — uploads disabled. "
              "Install with: brew install ffmpeg", file=sys.stderr)
        uploads = False

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
        "uploads": uploads,
        "max_upload_mb": args.max_upload_mb,
        "max_duration_s": args.max_duration_s,
        "max_queue": max(1, args.max_queue),
        "tmpdir": args.tmpdir or str(Path(tempfile.gettempdir()) / "breeze-uploads"),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
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
