#!/usr/bin/env bash
# =============================================================================
# Run BreezyVoice single-inference on macOS / MPS.
#
# Defaults reproduce the upstream run_single_inference.sh example, but reading
# data/example.wav and writing results/out.wav inside the cloned BreezyVoice/.
# Override by passing args through:
#   ./run.sh --content_to_synthesize "..." --speaker_prompt_audio_path "..."
#
# Required env vars:
#   PYTHONUTF8=1                  — upstream requires UTF-8 mode for CLI args.
#   PYTORCH_ENABLE_MPS_FALLBACK=1 — silently fall back to CPU for ops that
#                                   don't have an MPS kernel yet (some
#                                   Matcha-TTS / CosyVoice ops). Without this
#                                   you'll hit NotImplementedError mid-run.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
UPSTREAM_DIR="$PWD/BreezyVoice"
VENV_PY="$PWD/.venv/bin/python"

[[ -x "$VENV_PY" ]]      || { echo "[abort] .venv missing; run ./setup.sh first." >&2; exit 1; }
[[ -d "$UPSTREAM_DIR" ]] || { echo "[abort] BreezyVoice not cloned; run ./setup.sh first." >&2; exit 1; }

mkdir -p "$UPSTREAM_DIR/results"

export PYTHONUTF8=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$UPSTREAM_DIR"

if [[ $# -eq 0 ]]; then
    exec "$VENV_PY" single_inference.py \
        --speaker_prompt_audio_path "data/example.wav" \
        --speaker_prompt_text_transcription "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。" \
        --content_to_synthesize "歡迎使用聯發創新基地 BreezyVoice 模型。" \
        --output_path "results/out.wav"
else
    exec "$VENV_PY" single_inference.py "$@"
fi
