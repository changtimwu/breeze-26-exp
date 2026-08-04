# breeze-26-exp

Evaluation scratchpad for MediaTek Research's Breeze family of Taiwanese speech models:

- **Breeze-ASR-26** — Whisper-large-v2 fine-tune for Taiwanese Hokkien ASR (via [thc1006/breeze-asr-taigi](https://github.com/thc1006/breeze-asr-taigi)).
- **BreezyVoice** — Voice-cloning TTS for Taiwanese Mandarin (via [mtkresearch/BreezyVoice](https://github.com/mtkresearch/BreezyVoice)).

| Artifact | Task | Runtime | Engine / Model |
|---|---|---|---|
| [`breeze_asr_taigi_colab.ipynb`](./breeze_asr_taigi_colab.ipynb) | ASR | Google Colab (Linux + NVIDIA T4) | Faster-Whisper (CT2, `int8_float16`) — [`paulpengtw/faster-whisper-Breeze-ASR-26`](https://huggingface.co/paulpengtw/faster-whisper-Breeze-ASR-26) |
| [`mlx/test_mlx.py`](./mlx/test_mlx.py) | ASR | Apple Silicon (Metal) | `mlx-whisper`, 4-bit — [`doggy8088/Breeze-ASR-26-MLX-4bit`](https://huggingface.co/doggy8088/Breeze-ASR-26-MLX-4bit) |
| [`mlx/web/`](./mlx/web/) | ASR (browser → backend) | Apple Silicon + any modern browser | FastAPI + WebSocket + webrtcvad, same MLX model |
| [`wcpp/`](./wcpp/) | ASR | Apple Silicon (Metal + Accelerate) | `whisper.cpp` / ggml, f16 + q8_0/q5_k/q4_k — converted here from the original weights |
| [`wcpp/web/`](./wcpp/web/) | ASR (browser → backend) | Apple Silicon + any modern browser | FastAPI + WebSocket + webrtcvad → `whisper-server`, live quantization switch |
| [`breezyvoice_colab.ipynb`](./breezyvoice_colab.ipynb) | TTS / voice cloning | Google Colab (Linux + NVIDIA T4) | BreezyVoice (CosyVoice-derived) — [`MediaTek-Research/BreezyVoice`](https://huggingface.co/MediaTek-Research/BreezyVoice) |

---

## 1. Colab notebook — Faster-Whisper on a T4

Mirrors the project's `./install.sh`, but tuned for Colab:
- skips the `cu121` torch reinstall (Colab already ships CUDA-enabled torch),
- forces `--engine fw` / `TAIGI_ASR_DEFAULT_ENGINE=fw` so the auto-router doesn't pick the HuggingFace pipeline on a 16 GB T4 — the whole point of testing this project is the low-VRAM CT2 path.

### Run it

1. Open [`breeze_asr_taigi_colab.ipynb`](./breeze_asr_taigi_colab.ipynb) in Colab (File → Upload notebook).
2. **Runtime → Change runtime type → Hardware accelerator → GPU**.
3. Run cells top-to-bottom. First run downloads the model (~2.9 GB).

The notebook covers: runtime sanity check → clone → install → engine override → model preload → smoke test on `data/test.m4a` → upload-and-transcribe your own audio → (optional) Gradio UI via `--share`.

---

## 2. MLX smoke test — 4-bit on Apple Silicon

[`mlx/test_mlx.py`](./mlx/test_mlx.py) drives the MLX 4-bit port of Breeze-ASR-26 with `mlx_whisper`. The 4-bit model is ~877 MB; the model card recommends `language="zh"` since output is biased toward Chinese characters even for Taigi input.

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4). The script aborts on any other platform — MLX is Metal-only.
- `pip install -U mlx-whisper`
- For `--mic` mode: `pip install sounddevice`, and grant your terminal Microphone access (System Settings → Privacy & Security → Microphone). The first run will trigger the macOS permission prompt.
- For `--continuous --vad` mode: additionally `pip install webrtcvad`.

### Run it

```bash
# Optional: clone the upstream repo next to mlx/ so test_mlx.py finds the bundled sample.
git clone https://github.com/thc1006/breeze-asr-taigi.git

python mlx/test_mlx.py                          # uses breeze-asr-taigi/data/test.m4a
python mlx/test_mlx.py path/to/audio.m4a       # your own file
python mlx/test_mlx.py --mic                    # record from mic until Enter, then transcribe
python mlx/test_mlx.py --mic --duration 10      # fixed 10s recording
python mlx/test_mlx.py --continuous             # live chunked transcription, Ctrl-C to stop
python mlx/test_mlx.py --continuous --chunk 4   # tune chunk size (default 5s)
python mlx/test_mlx.py --continuous --vad       # VAD-segmented streaming (webrtcvad)
python mlx/test_mlx.py --continuous --vad --vad-aggressiveness 3 --vad-silence-ms 800
python mlx/test_mlx.py audio.m4a --fp16        # compare against the unquantized MLX model
python mlx/test_mlx.py audio.m4a --word-timestamps
```

Outputs land next to the input audio (or in CWD for mic modes):
- `<audio>.mlx.txt` — plain transcript
- `<audio>.mlx.srt` — SubRip subtitles (skip with `--no-srt`)
- `mic_YYYYMMDD_HHMMSS.wav` — the captured audio when using `--mic` (16 kHz mono)
- `mic_YYYYMMDD_HHMMSS_session.wav` + `.mlx.txt` — captured audio and concatenated transcript when using `--continuous`

The script reports wall time and xRT per chunk, so 4-bit MLX numbers are directly comparable to the Faster-Whisper benchmarks in the upstream README.

#### Note on `--continuous`

mlx-whisper has no true streaming API, so `--continuous` is the simplest thing that works: record into a fixed-size buffer in a background callback, transcribe each chunk independently when it fills, print as it lands. **No VAD, no overlap, no cross-chunk decoder state** — words spoken across a chunk boundary get cut. Shorter `--chunk` = lower latency but more boundary artifacts; longer = cleaner output but laggier feedback. A watchdog warns to stderr if transcription falls behind real-time recording.

#### `--continuous --vad`

Adds `webrtcvad`-based utterance segmentation on top of the continuous pipeline. webrtcvad classifies each 30 ms frame as speech or silence; the script keeps a 300 ms rolling pre-roll, opens an utterance on the first speech frame (prepending the pre-roll so onsets aren't clipped), and finalises it when it sees `--vad-silence-ms` of trailing silence (default 500 ms). Each finalised utterance is sent to `mlx_whisper.transcribe()` as one shot, so words no longer get split at arbitrary timer boundaries.

Knobs:
- `--vad-aggressiveness {0..3}`: webrtcvad's noise robustness. 0 is permissive (more false-speech, never miss real speech); 3 is aggressive (more false-silence, may clip very quiet speech). Default 2.
- `--vad-silence-ms`: how long a pause must be before the script decides the utterance is over. Lower = snappier but cuts pauses-mid-thought; higher = waits for full sentences but more lag. Default 500.

There's also a hardcoded 25 s safety cap on utterance length (Whisper degrades past 30 s of context), and a 300 ms minimum so single coughs don't trigger transcription. Tweak the constants in `transcribe_continuous_vad()` if you need to.

Pattern is borrowed from [Progressing-Llama/Whisper-Live-STT](https://github.com/Progressing-Llama/Whisper-Live-STT) but switched from window-gating to utterance-segmentation.

---

---

## 3. Web app — `mlx/web/` (browser mic → MLX backend)

A tiny FastAPI + WebSocket app that gives the CLI's `--continuous --vad` pipeline a browser frontend. The browser captures mic at 16 kHz, sends raw int16 PCM frames over a WebSocket, and the server runs the same webrtcvad utterance segmentation as `test_mlx.py`, then transcribes each finalised utterance with `mlx-whisper` and streams results back as JSON.

### Run it

```bash
pip install fastapi 'uvicorn[standard]' webrtcvad mlx-whisper numpy
cd mlx/web
python server.py                          # → http://127.0.0.1:8000
python server.py --port 8080 --fp16       # full-precision MLX model
python server.py --vad-silence-ms 800     # wait longer before closing an utterance
python server.py --host 0.0.0.0           # expose to LAN
```

Open the URL in any modern browser, click **Start**, allow microphone access, and start talking. Utterances appear as the backend finalises them, with per-utterance timings (spoken duration, decode wall time, xRT).

### Architecture

- **Browser** (`index.html`): `AudioContext({sampleRate: 16000})` → `AudioWorklet` that batches Float32 into Int16 PCM in 120 ms chunks (4 × 30 ms VAD frames) and posts them through a WebSocket as binary messages. UI follows the cream/charcoal design system in `mlx/DESIGN.md`.
- **Server** (`server.py`): one `webrtcvad` state machine per connection — pre-roll deque, speech accumulator, silence-triggered finalize — same logic as `transcribe_continuous_vad()` in `test_mlx.py`. Transcription runs in `asyncio.to_thread()` so the WebSocket event loop stays responsive. The model is warmed up at startup with 0.5 s of silence.
- **Protocol**: client → server sends raw `int16` PCM; server → client sends JSON: `{"type": "ready", ...}` on connect, `{"type": "speaking", "on": bool}` on VAD transitions, `{"type": "utterance", "text", "spoken_s", "decode_s", "xrt", "ts"}` per finalised utterance.

### Caveats

- Apple Silicon only (the server aborts on anything else — MLX is Metal-only).
- Browser must honour `AudioContext({sampleRate: 16000})`. Modern Safari and Chrome on macOS do; the page warns in the UI if a different rate comes back.
- Single user per server. No auth, no TLS. Bind to `127.0.0.1` unless you trust the LAN.
- The Camera Plain Variable typeface called out in the design spec isn't free; the page uses the `ui-sans-serif, system-ui` fallback chain — on macOS that resolves to San Francisco.

---

## 4. whisper.cpp — ggml on Apple Silicon (`wcpp/`)

Breeze-ASR-26 is a plain `whisper-large-v2` fine-tune, so whisper.cpp's HF→ggml
converter takes it **unmodified**. This directory converts the original MediaTek
weights ourselves rather than trusting a third-party repo, and measures the result
against the unconverted fp32 checkpoint.

```bash
./wcpp/convert.sh                 # build whisper.cpp, convert, quantize
wcpp/whisper.cpp/build/bin/whisper-cli -m wcpp/out/ggml-model.bin -l zh -f audio.wav -t 8
```

`-l zh` is mandatory: Whisper large-v2's `lang_to_id` has 99 entries and **`<|nan|>`
does not exist**, so Taigi has to be decoded as Chinese.

### Conversion is faithful

The ggml header came out correct on every field — including `n_text_ctx = 448`,
which the converter has to *infer*, because Breeze's `config.json` sets
`"max_length": null` and the script falls back to `max_target_positions`. All
**1259 of 1259** tensors were written; the only skipped entry is `proj_out.weight`,
which is tied to the decoder token embedding and correctly dropped.

Accuracy vs. the original fp32 checkpoint run through HuggingFace transformers
(12 clips / 297 s, all engines pinned to *identical* greedy decoding so the numbers
reflect weights rather than search strategy):

| build | size | CER vs fp32 | median CER | exact match | xRT |
|---|---|---|---|---|---|
| ggml f16 | 2951 MB | **4.46 %** | **1.60 %** | 5/12 | 0.082 |
| ggml q8_0 | 1579 MB | 4.88 % | 1.60 % | 5/12 | 0.076 |
| ggml q5_k | 1030 MB | 7.67 % | 6.63 % | 1/12 | 0.076 |
| ggml q4_k | 847 MB | 8.65 % | 7.72 % | 2/12 | **0.065** |
| `mlx-whisper` 4-bit | 877 MB | 14.92 % | 14.41 % | 1/12 | — |

f16 and q8_0 are indistinguishable from each other; q5_k/q4_k cost a few points of
CER for a 3x size cut. The third-party MLX 4-bit model is roughly **3x the error**
of our ggml f16 — a conversion-lineage difference as much as a quantization one.

Residual f16-vs-fp32 disagreement is expected, not a defect: whisper.cpp computes
its own mel spectrogram and runs f16 kernels, so on genuinely ambiguous audio the
two implementations can land on different (equally plausible) transcripts. The four
ggml variants agree closely with *each other* on exactly those clips.

Also settled: whisper.cpp ignores `suppress_tokens` entirely (Breeze dropped it from
`config.json`; only `generation_config.json` still carries the 88-token list) and it
made **no observable difference** — 5 of 12 clips matched HuggingFace character for
character anyway.

### Acceleration on this box (M1 Max, 32 GB)

Metal, Accelerate, and the ARM CPU extensions all engage — `whisper-cli` reports
`MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | ACCELERATE = 1`.
Encoder time for one 30 s window, f16:

| encoder backend | encode time | vs Metal |
|---|---|---|
| **Metal (GPU)** | **428 ms** | — |
| Core ML (`-DWHISPER_COREML=1`) | 565 ms | 1.3x slower |
| CPU only (`-ng`) | 2749 ms | 6.4x slower |

So **Metal is the right backend here** — Core ML/ANE is a net loss for this model.
Two caveats before concluding ANE is useless: the `.mlpackage` we built is fp32
(`--quantize` untried), and upstream marks `--optimize-ane` as "currently broken".
The first Core ML run also costs a one-time ~2.8 s ANE compile.

Quantization is a **footprint win, not a throughput win** — q4_k cuts the model 3.5x
and load time from 915 ms to 321 ms, but encode time barely moves, because
dequantization eats what the smaller weights save.

### Word timestamps work — with three non-obvious flags

Breeze inherited large-v2's DTW alignment heads *unchanged*: its 23 pairs in
`generation_config.json` are byte-identical to whisper.cpp's `g_aheads_large_v2`.
So `--dtw large.v2` is valid. But:

```bash
whisper-cli -m wcpp/out/ggml-model.bin -l zh --dtw large.v2 \
  -nfa -bs 1 -ml 1 -oj -of out audio.wav
```

- **`-nfa` is required.** Flash attention is on by default and silently disables DTW
  (`dtw_token_timestamps is not supported with flash_attn - disabling`).
- **Beam search must be off.** With the default `-bs 5`, DTW emits *zero* segments.
- Read the **JSON** (`-oj`), not stdout — the terminal rendering duplicates and
  reverses tokens even though the underlying data is fine.

Verified: 118 monotonically increasing token spans covering 0.000–30.000 s.

### The repetition loops are the model, not the port

`mlx/mic_20260528_231037_session.mlx.txt` opens with a degenerate `奶奶奶奶…` loop, and
it was an open question whether 4-bit quantization caused it. It did not — **the
original fp32 checkpoint loops on the same audio** (`妳... 妳...` x140, `師兄` x90) once
temperature fallback is disabled. It is ordinary Whisper degeneracy on hard Taigi
audio, and Whisper's `compression_ratio > 2.4` retry exists precisely for it.

What differs is how well each engine's *default* decoder recovers. On the three
degenerate clips, with each engine left at its own defaults:

- **whisper.cpp** (beam 5 + entropy/temperature fallback) recovered **all three** into
  coherent dialogue.
- **mlx-whisper** (greedy + temperature ladder) still looped on **two of three**.

That is a practical argument for the ggml path in the streaming app, independent of
raw accuracy.

### Live web app — `wcpp/web/`

A browser front end for evaluating Breeze-ASR-26 on your own speech. Mic → WebSocket →
webrtcvad utterance segmentation → WAV → `whisper-server` → transcript on the page.

```bash
pip install fastapi 'uvicorn[standard]' webrtcvad httpx numpy
python wcpp/web/server.py                 # → http://127.0.0.1:8000
python wcpp/web/server.py --model q8_0    # start on a different build
python wcpp/web/server.py --host 0.0.0.0  # expose to LAN
```

One command is enough — `server.py` spawns and supervises `whisper-server` itself, then
shuts it down on exit.

**Why `whisper-server` and not `whisper-cli`:** `whisper-cli` reloads the model on every
invocation (~0.9 s for f16), which kills the real-time feel. `whisper-server` keeps the
ggml model resident, so per-utterance cost is decode only.

**Eval affordances:**
- **Live quantization switch.** The dropdown hot-swaps f16/q8_0/q5_k/q4_k through
  `whisper-server`'s `/load` — measured at **0.5–0.65 s** per switch. Each utterance is
  tagged with the model that produced it, so you can A/B builds on the same voice in one
  session.
- **Copy / Download** the transcript, with per-utterance timings, for offline scoring.
- **Repetition-loop flag.** Utterances that collapse into a repeated token get a visible
  ⚠ marker, approximating Whisper's own `compression_ratio > 2.4` heuristic.

**Beam search is pinned on** (`--beam-size 5`). `whisper-server`'s own default is `-bs -1`
(off), and greedy decoding makes this model fall into repetition loops on hard Taigi audio —
so the app sets it explicitly. `--beam-size 1` reproduces the degenerate behaviour if you
want to see it.

Measured on this box, streaming committed recordings through the WebSocket as if they were
mic input: **0.84 s decode for a 5.3 s utterance (6.4x realtime)**, and **1.09 s for a
20.3 s utterance (18.6x)**. Very short utterances run *below* realtime (a 0.39 s clip took
0.51 s) because Whisper always processes a padded 30 s window — that floor is inherent, not
a backend problem.

Caveats:
- `whisper-server` holds one model context, so concurrent users **serialize**. Fine for one
  or two evaluators; not a multi-tenant service.
- No auth, no TLS. Keep it on `127.0.0.1` unless you trust the network.
- The browser must honour `AudioContext({sampleRate: 16000})`; the page warns if it doesn't.
- `--suppress-nst` is available to drop `(music)`-style non-speech tags, but is **off** by
  default so the app shows what the model actually emits.

### Files

- [`wcpp/convert.sh`](./wcpp/convert.sh) — build + convert + quantize, end to end
- [`wcpp/compare.py`](./wcpp/compare.py) — cross-engine CER harness with decoding pinned
- [`wcpp/hf_oracle.py`](./wcpp/hf_oracle.py) — fp32 reference via transformers
- [`wcpp/web/server.py`](./wcpp/web/server.py) — FastAPI + VAD + `whisper-server` supervisor
- [`wcpp/web/index.html`](./wcpp/web/index.html) — mic capture (AudioWorklet) + transcript UI

### Toolchain notes (both unrelated to the model)

- CMake picks Homebrew's clang, which cannot find the macOS C++ stdlib. Force
  `/usr/bin/clang++`.
- On this box `/Library/Developer/CommandLineTools/usr/include/c++/v1` was a **stale
  2022 leftover** with 11 entries and no `<array>`. Clang searches it first, fails, and
  never falls back to the SDK's complete copy. `convert.sh` detects this and adds
  `-isystem` on the SDK's libc++; the real fix is reinstalling the Command Line Tools.
- Full Xcode is **not** required. `GGML_METAL_EMBED_LIBRARY=ON` compiles Metal shaders
  at runtime instead of needing `xcrun metal`, and `coremltools.models.utils.compile_model()`
  produces the `.mlmodelc` that `xcrun coremlc` would otherwise be needed for.

---

## 5. BreezyVoice notebook — voice-cloning TTS on a T4

[`breezyvoice_colab.ipynb`](./breezyvoice_colab.ipynb) drives `single_inference.py` from the BreezyVoice repo to clone a speaker's voice and synthesize Taiwanese Mandarin text.

### Why it's not just `pip install -r requirements.txt`

BreezyVoice depends on `ttsfrd-0.3.9-cp310-cp310-linux_x86_64.whl` (the Mandarin text frontend), which is built for **Python 3.10 only**. Modern Colab ships Python 3.11, so the notebook bootstraps a Python 3.10 venv with `uv` and installs everything into it. Cells then invoke the inference scripts through `.venv/bin/python`.

Other tweaks:
- `DS_BUILD_OPS=0` is set so DeepSpeed installs without ahead-of-time op compilation (JIT at runtime instead). Without this the install can stall for ten minutes.
- `PYTHONUTF8=1` is exported globally — required by the upstream README for any non-ASCII text on the CLI.

### Run it

1. Open [`breezyvoice_colab.ipynb`](./breezyvoice_colab.ipynb) in Colab and switch to a GPU runtime.
2. Run cells top-to-bottom. The requirements install is slow (~10–20 min) and the first inference downloads the model (~a few GB).
3. The bundled inference example uses `data/example.wav` as the speaker prompt and writes `results/out.wav`.
4. The "clone your own voice" cell uploads a short reference recording and synthesizes whatever target text you give it.

### Tips

- Always pass `--speaker_prompt_text_transcription` for your reference audio if you have it — the script otherwise falls back to Whisper, which adds latency and can mis-hear domain terms.
- Use manual 注音 hints inline for tricky polyphones: `"今天天氣真好[:ㄏㄠ3]"`. The upstream auto-annotator handles most cases on its own.

---

## Reference: upstream VRAM decision table

From the project README, for context on what the auto-router would have picked:

| VRAM | Auto engine | compute_type | Notes |
|---|---|---|---|
| ≥ 22 GB | HuggingFace | float16 | A100/L4 |
| ≥ 14 GB | HuggingFace | float16 | **T4 lands here — overridden in the Colab notebook** |
| ≥ 3.5 GB | Faster-Whisper | int8_float16 | RTX 3050 4 GB — project's primary tuning target |
| no CUDA | Faster-Whisper | int8 (CPU) | — |
