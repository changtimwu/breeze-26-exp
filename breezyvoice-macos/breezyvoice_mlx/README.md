# breezyvoice-mlx

Apple **MLX** port of [BreezyVoice](https://huggingface.co/MediaTek-Research/BreezyVoice)
(a CosyVoice-**v1**-derived zero-shot TTS model), targeting Apple Silicon / macOS.
Replaces the CUDA/PyTorch inference path with MLX.

> ⚠️ **Language — read this:** BreezyVoice synthesizes **Taiwanese *Mandarin***
> (台灣腔的普通話 / 台灣人講的中文). It does **NOT** speak **Taiwanese Hokkien / Taigi**
> (台語 / 閩南語). The Taigi TTS is a *different, newer, unreleased* model —
> **BreezyVoice 26** (CosyVoice 2-based, Breeze 3 family); only Breeze ASR 26 and
> Breeze Guard 26 were open-sourced. Feeding this model Hokkien-orthography text
> yields a *Mandarin reading* of those characters, not Taigi pronunciation.
> See [issue #5](https://github.com/changtimwu/breeze-26-exp/issues/5).

> **Status:** the full pipeline (LLM → flow → HiFiGAN) is ported, parity-tested vs
> PyTorch (10 suites), runs end-to-end on real weights, and supports 4/8-bit LLM
> quantization + packaging. See **[PORTING_STATUS.md](PORTING_STATUS.md)**.
> Research & divergence analysis: https://github.com/changtimwu/breeze-26-exp/issues/3

## Layout

```
breezyvoice_mlx/
  nn/weight_norm.py      ✅ weight_norm fuse + runtime layer (MLX has no built-in)
  transformer/           ✅ attention + Conformer/Transformer encoder (parity-tested)
  llm/llm.py             ✅ text -> speech tokens (AR loop, parity-tested)
  flow/                  ✅ flow matching (CFM solver + UNet1D decoder, parity-tested)
  hifigan/               ✅ HiFiGAN-NSF vocoder + F0 predictor (parity-tested)
  cosyvoice.py           ✅ high-level API (build + SFT / zero-shot-builtin)
  model.py               ✅ llm -> flow -> hift orchestrator
  quantize.py            ✅ 4/8-bit LLM quantization
tools/convert_breezyvoice.py ✅ real llm/flow/hift.pt -> MLX (+ --quantize)
tools/{run_sft,package_model,ab_quant}.py ✅ synthesis / packaging / fidelity A/B
tests/*.py               ✅ 10 PyTorch-parity suites
```

## Quick start

```bash
# from breezyvoice-macos/, using the existing venv (has torch + now mlx)
.venv/bin/python -m pytest breezyvoice_mlx/tests/ -q

# convert vocoder weights once you have the BreezyVoice model dir:
.venv/bin/python breezyvoice_mlx/tools/convert_weights.py \
    --pt /path/to/BreezyVoice-model/hift.pt --out hift.safetensors
```

## End-to-end (Phase 6 — runs on real weights)

```bash
# 1. convert MediaTek-Research/BreezyVoice .pt checkpoints -> MLX safetensors
.venv/bin/python breezyvoice_mlx/tools/convert_breezyvoice.py --out-dir converted_mlx

# 2. synthesize (built-in speaker, zero-shot conditioning from spk2info)
PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tools/run_sft.py \
    --weights converted_mlx --spk 中文女 \
    --text "歡迎使用聯發創新基地 BreezyVoice 模型。" --out out.wav
```

### Quantization (LLM Linear layers; flow/vocoder stay fp32)

```bash
python breezyvoice_mlx/tools/convert_breezyvoice.py --out-dir converted_q8 --quantize --bits 8
# run_sft / BreezyVoice auto-detect quant_config.json and load quantized
```

A/B vs fp32 (LLM logit fidelity, teacher-forced; total = llm+flow+hift on disk):

| precision | total disk | top-1 | top-5 | softmax cosine |
|---|---|---|---|---|
| fp32 | 1744 MB | — | — | — |
| **8-bit (recommended)** | **940 MB** | **99.6%** | 99.3% | **100.0%** |
| 4-bit | 801 MB | 84.3% | 86.1% | 97.0% |

**8-bit is near-lossless** and 1.85× smaller. 4-bit is noticeably lossy and only
saves ~140 MB more (flow+vocoder fp32 dominate the rest), so 8-bit is the default
recommendation. NOTE: greedy decode degenerates this model (it's sampling-trained),
so quality is measured by logit fidelity, not token argmax.

### Packaging

```bash
python breezyvoice_mlx/tools/package_model.py --weights-dir converted_q8 \
    --out dist/BreezyVoice-300M-MLX-8bit
```
Assembles a self-contained model dir (weights + `quant_config.json` + `spk2info.pt`,
hard-linked) and writes a generated `MODEL_CARD.md` (lineage, precision, speakers,
usage, fidelity, license). Load it directly: `BreezyVoice(model_dir=DIR, weights_dir=DIR)`.

### Status

The full MLX pipeline (LLM → flow → HiFiGAN) runs end-to-end on real
weights and produces speech. Known issue: ~9% sample clipping in the vocoder
(strided-conv/ConvTranspose fp32 drift amplified by the exp() magnitude path) —
a quality refinement, see PORTING_STATUS.md. Arbitrary-prompt zero-shot (own
reference audio) additionally needs the ONNX speech tokenizer + CAM++ on CPU
(`frontend.py`, optional).

## Strategy in one line
Reference the already-working **CosyVoice3-MLX** (`mlx-audio-plus`) for patterns,
port BreezyVoice's *own* CosyVoice2 modules (different LLM/flow lineage), reuse the
ONNX frontend on CPU for Phase 1, and fuse `weight_norm` at conversion time.
