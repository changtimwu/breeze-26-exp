# BreezyVoice → MLX port: status & plan

Apple MLX port of **BreezyVoice** (CosyVoice **v1**-derived zero-shot TTS).
Research + full divergence map: https://github.com/changtimwu/breeze-26-exp/issues/3

Reference PyTorch source lives next door: `../BreezyVoice/cosyvoice/`.

> **Lineage (confirmed in Phase 0):** BreezyVoice is **CosyVoice v1**-derived
> (README cites `du2024cosyvoice`; LLM = Conformer text encoder + RNN/transformer
> `TransformerLM`; `speech_tokenizer_v1.onnx`; HF `MediaTek-Research/BreezyVoice`).
> NOT the Qwen2-based CosyVoice2/3. This matters for the LLM stage (see below).

## Phase 0 — DONE ✅ (baseline stands up)

`mlx-audio-plus` (0.1.8) ships full MLX ports of **both `cosyvoice2` and
`cosyvoice3`** — a far stronger reference than expected. Verified the MLX TTS
stack runs end-to-end on this machine:

    .venv/bin/python breezyvoice_mlx/tools/phase0_baseline.py \
        --repo mlx-community/Fun-CosyVoice3-0.5B-2512-8bit \
        --ref-audio BreezyVoice/data/example.wav --ref-text "…" --text "…"
    # -> results/phase0_cosyvoice3_mlx.wav  (9.12s @ 24kHz, real audio, RMS 0.07)

Notes: only **CosyVoice3** has prebuilt MLX weights on the Hub
(`mlx-community/Fun-CosyVoice3-0.5B-2512-{8bit,...}`); `mlx-community/CosyVoice2-0.5B`
does **not** exist (CosyVoice2 MLX needs `convert_from_source` on the FunAudioLLM
PyTorch checkpoint). First-run RTF (~11) is dominated by model + S3-tokenizer
download and MLX warmup — not representative.

### Reusability map (mlx-audio-plus → BreezyVoice)

| Stage | Verdict | Notes |
|---|---|---|
| **LLM** | ❌ REWRITE | both MLX ports use **Qwen2**; BreezyVoice v1 = Conformer + RNN `TransformerLM`. Port BreezyVoice's own. |
| **Flow (CFM + UNet1D)** | ✅ ADAPT | `cosyvoice2` uses **UNet1D** (same family as BreezyVoice, *not* v3's DiT). Euler+CFG reusable; port Conformer encoder + length regulator. |
| **HiFiGAN-NSF vocoder** | ✅ ADAPT-CONFIG | `cosyvoice2.hifigan.CosyHiFTGenerator` near-identical; weight_norm baked at load, STFT/iSTFT provided. Adapt 24k→22.05k + upsample rates. |
| **Speaker enc (CAM++)** | ✅ REUSE | native MLX CAM++ in `mlx_audio.codec.models.s3gen.xvector` replaces `campplus.onnx`. |
| **Speech tokenizer** | ⚠️ PARTIAL | native S3 tokenizer exists (`mlx_audio.codec.models.s3tokenizer`) but it's v2/v3's; BreezyVoice's is `speech_tokenizer_v1.onnx` → keep ONNX on CPU for Phase 1. |

**Net:** the genuinely novel work narrows to (1) BreezyVoice's **v1 Conformer+RNN
LLM** and (2) the **v1 speech tokenizer**. Vocoder/flow/speaker-encoder lean on
existing MLX code. The HiFiGAN "HARD" item in the table below is downgraded.

Reference MLX source to read while porting:
`.venv/lib/python3.10/site-packages/mlx_audio/tts/models/cosyvoice2/`
(`hifigan.py`, `flow_matching.py`, `flow*`, `speaker_encoder.py`, `llm/llm.py`,
`scripts/convert.py`).

## Pipeline

```
text ──[LLM]──▶ speech tokens ──[Flow]──▶ mel ──[HiFiGAN-NSF]──▶ waveform
```

## Module status

| Module | MLX file | Source | Difficulty | Status |
|---|---|---|---|---|
| weight_norm fuse/layer | `nn/weight_norm.py` | (MLX gap) | — | ✅ **done + parity test** |
| weight converter | `tools/convert_weights.py` | — | — | ✅ scaffold (verify on hift.pt) |
| Attention | `transformer/attention.py` | `transformer/attention.py` | EASY | ✅ **done + parity test** |
| Conformer encoder | `transformer/encoder.py` (+embedding/subsampling/ffn/encoder_layer) | `transformer/encoder.py` | MEDIUM | ✅ **done + e2e parity test** |
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

1. **Phase 0 — baseline** ✅ DONE: `mlx-audio-plus` installed; CosyVoice3 0.5B
   runs end-to-end (`tools/phase0_baseline.py`). cosyvoice2 MLX source is the
   primary reference to adapt (closer to BreezyVoice's UNet1D flow + HiFT vocoder).
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
