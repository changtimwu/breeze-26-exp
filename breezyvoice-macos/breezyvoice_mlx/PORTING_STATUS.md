# BreezyVoice → MLX port: status & plan

Apple MLX port of **BreezyVoice** (CosyVoice2-derived zero-shot TTS).
Research + full divergence map: https://github.com/changtimwu/breeze-26-exp/issues/3

Reference PyTorch source lives next door: `../BreezyVoice/cosyvoice/`.

## Pipeline

```
text ──[LLM]──▶ speech tokens ──[Flow]──▶ mel ──[HiFiGAN-NSF]──▶ waveform
```

## Module status

| Module | MLX file | Source | Difficulty | Status |
|---|---|---|---|---|
| weight_norm fuse/layer | `nn/weight_norm.py` | (MLX gap) | — | ✅ **done + parity test** |
| weight converter | `tools/convert_weights.py` | — | — | ✅ scaffold (verify on hift.pt) |
| Attention | `transformer/attention.py` | `transformer/attention.py` | EASY | ⬜ stub |
| Conformer encoder | `transformer/encoder.py` | `transformer/encoder.py` | MEDIUM | ⬜ stub |
| LLM (AR loop) | `llm/llm.py` | `llm/llm.py` | MEDIUM | ⬜ stub |
| Flow wrapper | `flow/flow.py` | `flow/flow.py` | MEDIUM | ⬜ stub |
| CFM ODE solver | `flow/flow_matching.py` | `flow/flow_matching.py` | EASY-MED | ⬜ stub |
| UNet1D decoder | `flow/decoder.py` | `flow/decoder.py` | MEDIUM | ⬜ stub |
| HiFiGAN-NSF | `hifigan/generator.py` | `hifigan/generator.py` | HARD | ⬜ stub |
| F0 predictor | `hifigan/f0_predictor.py` | `hifigan/f0_predictor.py` | MEDIUM | ⬜ stub |
| Frontend adapter | `frontend.py` | `cli/frontend.py` | EASY* | ⬜ stub |
| Orchestrator | `model.py` | `cli/model.py` | EASY | ⬜ stub |

\* Frontend is easy because the ONNX models (campplus, speech_tokenizer) already
run on CPU via onnxruntime on macOS — **reuse them as-is in Phase 1**; MLX-native
ports are a later optimization, not a blocker.

## Two solved gotchas

1. **`weight_norm`** (MLX has none): `nn/weight_norm.py`. Since BreezyVoice calls
   `remove_weight_norm()` at inference, we **fuse `weight_g`/`weight_v` at
   conversion time** — no runtime layer needed, bit-exact. Verified by
   `tests/test_weight_norm.py` against PyTorch.
2. **Conv layout**: PyTorch `(out,in,k)` → MLX `(out,k,in)` (Conv1d) / `(out,kH,kW,in)`
   (Conv2d). Handled in `tools/convert_weights.py`.

## Recommended build order

1. **Phase 0 — baseline** *(do first, validates the whole thesis)*: `pip install
   mlx-audio-plus`, run **CosyVoice3 0.5B** end-to-end as a working MLX TTS
   reference. Borrow patterns (NOT weights — different lineage).
2. **Phase 1 — converter**: run `convert_weights.py` on `hift.pt`/`flow.pt`/`llm.pt`,
   confirm fused-weight-norm + conv-layout against a tiny PyTorch forward.
3. **Phase 2 — transformer core**: `attention.py` → `encoder.py` (skip
   chunking/grad-ckpt; inference uses full context).
4. **Phase 3 — LLM**: `llm/llm.py` AR decode loop + KV cache + top-k sampling.
5. **Phase 4 — flow**: `flow_matching.py` (Euler + CFG) → `decoder.py` (UNet1D,
   Matcha blocks) → `flow.py`.
6. **Phase 5 — HiFiGAN** (hardest): ResBlock+Snake → NSF source → STFT/iSTFT
   (manual framing + Hann window + rfft/overlap-add) → assemble.
7. **Phase 6 — integration**: `frontend.py` adapter (reuse ONNX on CPU) + `model.py`
   + `cosyvoice.py` API; validate numerically vs PyTorch on a fixed seed.

## What stays on torch/CPU (Phase 1)
- `campplus.onnx`, `speech_tokenizer_v1.onnx` (onnxruntime CPU)
- text normalization (WeTextProcessing / ttsfrd / inflect)
- prompt feature extraction (whisper log-mel, kaldi fbank)

## Not ported (training-only)
- `bin/train.py`, `utils/train_utils.py` (DDP, `torch.cuda.amp.autocast`), dataset
  pipeline, gradient checkpointing, label-smoothing loss.
