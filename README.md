# breeze-26-exp

Evaluation scratchpad for [thc1006/breeze-asr-taigi](https://github.com/thc1006/breeze-asr-taigi) — a wrapper around MediaTek's **Breeze-ASR-26** (a Whisper-large-v2 fine-tune for Taiwanese Hokkien).

Two runtimes are tested here:

| Artifact | Runtime | Engine | Model |
|---|---|---|---|
| [`breeze_asr_taigi_colab.ipynb`](./breeze_asr_taigi_colab.ipynb) | Google Colab (Linux + NVIDIA T4) | Faster-Whisper (CT2, `int8_float16`) | [`paulpengtw/faster-whisper-Breeze-ASR-26`](https://huggingface.co/paulpengtw/faster-whisper-Breeze-ASR-26) |
| [`mlx/test_mlx.py`](./mlx/test_mlx.py) | Apple Silicon (Metal) | `mlx-whisper`, 4-bit quantized | [`doggy8088/Breeze-ASR-26-MLX-4bit`](https://huggingface.co/doggy8088/Breeze-ASR-26-MLX-4bit) |

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

### Run it

```bash
# Optional: clone the upstream repo next to mlx/ so test_mlx.py finds the bundled sample.
git clone https://github.com/thc1006/breeze-asr-taigi.git

python mlx/test_mlx.py                       # uses breeze-asr-taigi/data/test.m4a
python mlx/test_mlx.py path/to/audio.m4a    # your own file
python mlx/test_mlx.py audio.m4a --fp16     # compare against the unquantized MLX model
python mlx/test_mlx.py audio.m4a --word-timestamps
```

Outputs land next to the input audio:
- `<audio>.mlx.txt` — plain transcript
- `<audio>.mlx.srt` — SubRip subtitles (skip with `--no-srt`)

The script reports wall time and xRT, so 4-bit MLX numbers are directly comparable to the Faster-Whisper benchmarks in the upstream README.

---

## Reference: upstream VRAM decision table

From the project README, for context on what the auto-router would have picked:

| VRAM | Auto engine | compute_type | Notes |
|---|---|---|---|
| ≥ 22 GB | HuggingFace | float16 | A100/L4 |
| ≥ 14 GB | HuggingFace | float16 | **T4 lands here — overridden in the Colab notebook** |
| ≥ 3.5 GB | Faster-Whisper | int8_float16 | RTX 3050 4 GB — project's primary tuning target |
| no CUDA | Faster-Whisper | int8 (CPU) | — |
