# breeze-26-exp

Evaluation scratchpad for MediaTek Research's Breeze family of Taiwanese speech models:

- **Breeze-ASR-26** — Whisper-large-v2 fine-tune for Taiwanese Hokkien ASR (via [thc1006/breeze-asr-taigi](https://github.com/thc1006/breeze-asr-taigi)).
- **BreezyVoice** — Voice-cloning TTS for Taiwanese Mandarin (via [mtkresearch/BreezyVoice](https://github.com/mtkresearch/BreezyVoice)).

| Artifact | Task | Runtime | Engine / Model |
|---|---|---|---|
| [`breeze_asr_taigi_colab.ipynb`](./breeze_asr_taigi_colab.ipynb) | ASR | Google Colab (Linux + NVIDIA T4) | Faster-Whisper (CT2, `int8_float16`) — [`paulpengtw/faster-whisper-Breeze-ASR-26`](https://huggingface.co/paulpengtw/faster-whisper-Breeze-ASR-26) |
| [`mlx/test_mlx.py`](./mlx/test_mlx.py) | ASR | Apple Silicon (Metal) | `mlx-whisper`, 4-bit — [`doggy8088/Breeze-ASR-26-MLX-4bit`](https://huggingface.co/doggy8088/Breeze-ASR-26-MLX-4bit) |
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

## 3. BreezyVoice notebook — voice-cloning TTS on a T4

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
