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

## Strategy in one line
Reference the already-working **CosyVoice3-MLX** (`mlx-audio-plus`) for patterns,
port BreezyVoice's *own* CosyVoice2 modules (different LLM/flow lineage), reuse the
ONNX frontend on CPU for Phase 1, and fuse `weight_norm` at conversion time.
