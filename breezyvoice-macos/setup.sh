#!/usr/bin/env bash
# =============================================================================
# BreezyVoice macOS bootstrap (Apple Silicon, PyTorch MPS).
#
# Clones BreezyVoice next to this script, creates a Python 3.10 venv via uv,
# installs the macOS-compatible requirements, and patches three lines of
# hardcoded CUDA device selection so the model runs on Apple's MPS backend.
#
# Usage:
#   ./setup.sh
#
# Re-runnable: skips steps that already completed.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$PWD"
UPSTREAM_DIR="$SCRIPT_DIR/BreezyVoice"
VENV_DIR="$SCRIPT_DIR/.venv"

say()  { printf "\033[1m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[warn ]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[31m[abort]\033[0m %s\n" "$*" >&2; exit 1; }

# ---- 0. Sanity checks --------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    die "this setup targets macOS on Apple Silicon (darwin/arm64); detected $(uname -s)/$(uname -m)."
fi

if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew not found. Install from https://brew.sh first."
fi

if ! command -v uv >/dev/null 2>&1; then
    warn "uv not found; installing via brew."
    brew install uv
fi

# WeTextProcessing depends on pynini, which on macOS needs OpenFst headers.
if ! brew list openfst >/dev/null 2>&1; then
    say "installing openfst (required by pynini -> WeTextProcessing)"
    brew install openfst
fi

# ---- 1. Clone BreezyVoice ----------------------------------------------------

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    say "cloning mtkresearch/BreezyVoice"
    git clone https://github.com/mtkresearch/BreezyVoice.git "$UPSTREAM_DIR"
else
    say "BreezyVoice already cloned at $UPSTREAM_DIR (skipping git clone)"
fi

# ---- 2. Apply MPS patches ----------------------------------------------------
#
# BreezyVoice (via vendored CosyVoice) hardcodes the device as
#   torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# in three places. On Mac this silently falls back to CPU, ignoring MPS.
# Patch all three to a tri-state cuda > mps > cpu selection.
#
# A marker comment marks the patched lines so the sed is idempotent.

PATCH_OLD="torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
PATCH_NEW="torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))  # patched-mps"

for f in \
    "$UPSTREAM_DIR/cosyvoice/cli/model.py" \
    "$UPSTREAM_DIR/cosyvoice/cli/frontend.py" \
    "$UPSTREAM_DIR/single_inference.py"
do
    [[ -f "$f" ]] || { warn "expected file missing: $f"; continue; }
    if grep -q "patched-mps" "$f"; then
        say "already patched: ${f#$UPSTREAM_DIR/}"
        continue
    fi
    if ! grep -qF "$PATCH_OLD" "$f"; then
        warn "no CUDA device line found in ${f#$UPSTREAM_DIR/} — upstream may have changed; review manually."
        continue
    fi
    say "patching ${f#$UPSTREAM_DIR/} for MPS"
    # Use python for the in-place edit to avoid sed quoting hell on the
    # nested parens/quotes in PATCH_NEW.
    python3 - "$f" "$PATCH_OLD" "$PATCH_NEW" <<'PY'
import sys, pathlib
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
p.write_text(p.read_text().replace(old, new))
PY
done

# ---- 3. Create venv ----------------------------------------------------------

if [[ ! -d "$VENV_DIR" ]]; then
    say "creating Python 3.10 venv at .venv"
    uv venv --python 3.10 "$VENV_DIR"
else
    say ".venv already exists (skipping venv create)"
fi

# ---- 4. Install requirements -------------------------------------------------

say "installing requirements (this takes a few minutes)"
uv pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements-macos.txt"

# Install BreezyVoice's Matcha-TTS third-party as importable (single_inference
# already prepends its path to sys.path, so no -e install is required).

# ---- 5. Smoke check ----------------------------------------------------------

say "verifying torch + MPS"
"$VENV_DIR/bin/python" - <<'PY'
import torch
print(f"  torch:               {torch.__version__}")
print(f"  cuda available:      {torch.cuda.is_available()}")
print(f"  mps available:       {torch.backends.mps.is_available()}")
print(f"  mps built:           {torch.backends.mps.is_built()}")
PY

cat <<'EOF'

================================================================
  Setup finished.
  Run ./run.sh to synthesize the bundled example with MPS.
================================================================
EOF
