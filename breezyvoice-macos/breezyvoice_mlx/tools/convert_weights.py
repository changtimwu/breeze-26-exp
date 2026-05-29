"""Convert BreezyVoice PyTorch checkpoints (.pt) -> MLX safetensors.

Handles the two PyTorch->MLX gotchas this model hits:
  1. weight_norm: collapse (`*.weight_g`, `*.weight_v`) pairs into a single fused
     `*.weight` (w = g * v / ||v||), matching PyTorch's remove_weight_norm.
  2. Conv layout: PyTorch Conv1d weight is (out, in, kernel); MLX Conv1d expects
     (out, kernel, in). PyTorch Conv2d (out,in,kH,kW) -> MLX (out,kH,kW,in).

This is a *scaffold*: the conv-detection heuristic (by key suffix + ndim) is
deliberately conservative and logs every decision. Verify against a known module
(e.g. hift.pt) before trusting it wholesale, and extend the skip/keep lists as
the real modules land.

Usage:
    python tools/convert_weights.py --pt /path/to/hift.pt --out hift.safetensors
    python tools/convert_weights.py --pt /path/to/llm.pt  --out llm.safetensors --no-fuse-wn
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("torch required for conversion (read the source .pt). Use the BreezyVoice venv.")

import mlx.core as mx

from breezyvoice_mlx.nn.weight_norm import fuse_weight_norm  # noqa: E402


def _is_conv_weight(key: str, arr) -> bool:
    """Heuristic: a `.weight` on a conv layer (ndim 3 = Conv1d, 4 = Conv2d).

    Linear weights are ndim 2 and must NOT be transposed (MLX Linear matches
    PyTorch's (out, in) layout).
    """
    return key.endswith(".weight") and arr.ndim in (3, 4)


def _transpose_conv(arr: mx.array) -> mx.array:
    if arr.ndim == 3:        # Conv1d (out, in, k) -> (out, k, in)
        return mx.transpose(arr, (0, 2, 1))
    if arr.ndim == 4:        # Conv2d (out, in, kH, kW) -> (out, kH, kW, in)
        return mx.transpose(arr, (0, 2, 3, 1))
    return arr


def convert(pt_path: str, fuse_wn: bool = True, fix_conv_layout: bool = True) -> dict:
    state = torch.load(pt_path, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    # numpy bridge: torch -> numpy -> mx (no torch/mlx dtype coupling).
    raw = {k: mx.array(v.detach().to(torch.float32).numpy()) for k, v in state.items()}

    out: dict[str, mx.array] = {}
    fused, skipped_wn, transposed = 0, set(), 0

    # 1. fuse weight_norm pairs
    if fuse_wn:
        g_keys = [k for k in raw if k.endswith(".weight_g")]
        for gk in g_keys:
            base = gk[: -len(".weight_g")]
            vk = base + ".weight_v"
            if vk not in raw:
                print(f"  [warn] {gk} has no matching {vk}; leaving as-is")
                continue
            out[base + ".weight"] = fuse_weight_norm(raw[gk], raw[vk], dim=0)
            skipped_wn.update({gk, vk})
            fused += 1
        consumed = skipped_wn
    else:
        consumed = set()

    # 2. copy the rest, fixing conv layout
    for k, v in raw.items():
        if k in consumed:
            continue
        out[k] = v

    if fix_conv_layout:
        for k in list(out):
            if _is_conv_weight(k, out[k]):
                out[k] = _transpose_conv(out[k])
                transposed += 1

    print(f"  fused weight_norm pairs: {fused}")
    print(f"  conv weights transposed: {transposed}")
    print(f"  total tensors out:       {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True, help="source PyTorch .pt checkpoint")
    ap.add_argument("--out", required=True, help="destination .safetensors path")
    ap.add_argument("--no-fuse-wn", dest="fuse_wn", action="store_false",
                    help="don't fuse weight_g/weight_v (use for llm.pt which has none)")
    ap.add_argument("--no-conv-fix", dest="fix_conv", action="store_false",
                    help="don't transpose conv layouts")
    args = ap.parse_args()

    print(f"Converting {args.pt} ...")
    weights = convert(args.pt, fuse_wn=args.fuse_wn, fix_conv_layout=args.fix_conv)
    mx.save_safetensors(args.out, weights)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
