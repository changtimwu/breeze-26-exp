#!/usr/bin/env bash
# Reproduce the whisper.cpp (ggml) build of Breeze-ASR-26 on Apple Silicon.
#
# Produces:
#   out/ggml-model.bin          f16, ~2.9 GiB
#   out/ggml-breeze-q8_0.bin    ~1.5 GiB
#   out/ggml-breeze-q5_k.bin    ~1.0 GiB
#   out/ggml-breeze-q4_k.bin    ~0.8 GiB
#
# Prereqs: cmake, git, a Python with torch + transformers + numpy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-../mlx/venv/bin/python}"
MODEL_REPO="MediaTek-Research/Breeze-ASR-26"

# ---------------------------------------------------------------------------
# 0. macOS toolchain notes
# ---------------------------------------------------------------------------
# Two things bit us on a clean-ish machine, both unrelated to the model:
#
#  * Homebrew's clang cannot find the macOS C++ stdlib. Force Apple's.
#  * If /Library/Developer/CommandLineTools/usr/include/c++/v1 exists but is a
#    stale partial copy (ours was a 2022 leftover with 11 entries and no
#    <array>), clang searches it FIRST, fails, and never falls back to the
#    complete copy inside the SDK. -isystem on the SDK's libc++ works around it
#    without touching system files. The real fix is reinstalling the Command
#    Line Tools.
SDK="$(xcrun --show-sdk-path)"
SDKCXX="$SDK/usr/include/c++/v1"
EXTRA_CXX_FLAGS=""
if ! echo '#include <array>
int main(){return 0;}' | /usr/bin/clang++ -x c++ - -o /dev/null 2>/dev/null; then
  echo "[warn] default libc++ include path is broken; forcing SDK libc++"
  EXTRA_CXX_FLAGS="-isystem $SDKCXX"
fi

# ---------------------------------------------------------------------------
# 1. whisper.cpp, built with every accelerator this Mac offers
# ---------------------------------------------------------------------------
# GGML_METAL          -> GPU (M1 Max: MTLGPUFamilyApple7, simdgroup matmul)
# GGML_METAL_EMBED_LIBRARY -> embed shader source, compiled at runtime.
#                       Avoids needing full Xcode for `xcrun metal`.
# GGML_BLAS/ACCELERATE-> Accelerate.framework sgemm on CPU
# CPU backend picks up NEON + ARM_FMA + FP16_VA + DOTPROD automatically.
if [ ! -d whisper.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp
fi

cmake -S whisper.cpp -B whisper.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_CXX_FLAGS="$EXTRA_CXX_FLAGS" \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON \
  -DGGML_BLAS=ON \
  -DWHISPER_BUILD_EXAMPLES=ON
cmake --build whisper.cpp/build --config Release -j"$(sysctl -n hw.ncpu)"

# ---------------------------------------------------------------------------
# 2. Mel filter bank
# ---------------------------------------------------------------------------
# convert-h5-to-ggml.py wants <dir>/whisper/assets/mel_filters.npz from a clone
# of openai/whisper. That one 4 KB file is all it needs, so skip the clone.
# Breeze is num_mel_bins=80 (large-v2 geometry), so the mel_80 key is the one
# that gets used.
mkdir -p whisper-assets/whisper/assets
if [ ! -f whisper-assets/whisper/assets/mel_filters.npz ]; then
  curl -fsSL -o whisper-assets/whisper/assets/mel_filters.npz \
    https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
fi

# ---------------------------------------------------------------------------
# 3. Weights
# ---------------------------------------------------------------------------
# Sharded safetensors are fine here: the converter goes through
# WhisperForConditionalGeneration.from_pretrained, which resolves shards itself.
# (mlx-whisper's converter does NOT — it mx.loads a single model.safetensors.)
# training_args.bin is a pickle we don't need.
SNAP="$("$PYTHON" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$MODEL_REPO", ignore_patterns=["training_args.bin"]))
PY
)"
echo "[info] weights: $SNAP"

# ---------------------------------------------------------------------------
# 4. HF -> ggml f16
# ---------------------------------------------------------------------------
mkdir -p out
"$PYTHON" whisper.cpp/models/convert-h5-to-ggml.py \
  "$SNAP" whisper-assets out | tee out/convert.log | tail -3

# ---------------------------------------------------------------------------
# 5. Quantize
# ---------------------------------------------------------------------------
Q=whisper.cpp/build/bin/whisper-quantize
for t in q8_0 q5_k q4_k; do
  [ -f "out/ggml-breeze-$t.bin" ] || "$Q" out/ggml-model.bin "out/ggml-breeze-$t.bin" "$t"
done

ls -la out/*.bin

cat <<'EOF'

Done. Transcribe with (language MUST be zh — Whisper large-v2 has no <|nan|> token):

  whisper.cpp/build/bin/whisper-cli -m out/ggml-model.bin -l zh -f audio.wav -t 8

Word timestamps work with the large-v2 DTW preset, because Breeze inherited
large-v2's alignment heads unchanged:

  whisper.cpp/build/bin/whisper-cli -m out/ggml-model.bin -l zh --dtw large.v2 \
    -ml 1 -f audio.wav
EOF
