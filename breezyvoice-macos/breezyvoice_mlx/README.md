# breezyvoice-mlx

Apple **MLX** port of [BreezyVoice](../BreezyVoice) (a CosyVoice2-derived zero-shot
TTS model), targeting Apple Silicon / macOS. Replaces the CUDA/PyTorch inference
path with MLX.

> **Status: scaffold.** Structure, the `weight_norm` blocker (solved + tested), and
> the PyTorch→MLX weight converter are in place. Model modules are faithful stubs.
> See **[PORTING_STATUS.md](PORTING_STATUS.md)** for the module map and build order.
> Research & divergence analysis: https://github.com/changtimwu/breeze-26-exp/issues/3

## Layout

```
breezyvoice_mlx/
  nn/weight_norm.py      ✅ weight_norm fuse + runtime layer (MLX has no built-in)
  transformer/           ⬜ attention, Conformer encoder
  llm/llm.py             ⬜ text -> speech tokens (AR loop)
  flow/                  ⬜ flow matching (CFM solver + UNet1D decoder)
  hifigan/               ⬜ HiFiGAN-NSF vocoder + F0 predictor
  frontend.py            ⬜ adapter over the existing CPU/ONNX frontend
  model.py               ⬜ llm -> flow -> hift orchestrator
tools/convert_weights.py ✅ .pt -> MLX safetensors (fuses weight_norm, fixes conv layout)
tests/test_weight_norm.py ✅ parity vs PyTorch
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
