"""Convert a BreezyVoice PyTorch model dir (MediaTek-Research/BreezyVoice) to MLX.

Produces llm.safetensors, flow.safetensors, hift.safetensors next to the MLX port.
Each stage uses the appropriate remap:
  * LLM  — direct (all Linear/LayerNorm/Embedding/pos_bias; no convs, no transpose)
  * Flow — encoder/proj/spk/input_embedding direct; length_regulator convs
           transposed; decoder.estimator.* via the UNet decoder remap
  * HiFi — weight_norm fused + conv layouts + F0 condnet index collapse

Usage:
    python tools/convert_breezyvoice.py --model-dir <BreezyVoice dir> --out-dir <dst>
    # or resolve the HF repo automatically:
    python tools/convert_breezyvoice.py --out-dir converted_mlx
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import mlx.core as mx

try:
    import torch
except ImportError:
    raise SystemExit("torch is required to read the .pt checkpoints")

from breezyvoice_mlx.flow.decoder import _remap_key as _decoder_remap_key
from breezyvoice_mlx.hifigan.generator import remap_hift_weights


def _to_mx(t) -> mx.array:
    return mx.array(np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                               dtype=np.float32))


def convert_llm(state: dict) -> list:
    # No convolutions in the LLM (Conformer use_cnn_module=False, linear input
    # layers) -> every weight is a Linear/LayerNorm/Embedding/pos_bias, loads
    # directly with no transpose.
    return [(k, _to_mx(v)) for k, v in state.items()]


def convert_flow(state: dict) -> list:
    out = []
    for k, v in state.items():
        a = _to_mx(v)
        if k.startswith("decoder.estimator."):
            inner = _decoder_remap_key(k[len("decoder.estimator."):])
            nk = "decoder.estimator." + inner
            if k.endswith(".weight") and a.ndim == 3:
                a = mx.transpose(a, (1, 2, 0)) if ".upsample.conv." in nk \
                    else mx.transpose(a, (0, 2, 1))
            out.append((nk, a))
        elif k.startswith("decoder."):
            # ConditionalCFM has no other learnable params; skip stray buffers.
            continue
        else:
            # encoder.* / encoder_proj.* / spk_embed_affine_layer.* /
            # input_embedding.* are Linear/LayerNorm/Embedding (no transpose);
            # length_regulator.model.* convs need (out,in,k)->(out,k,in).
            if k.endswith(".weight") and a.ndim == 3:
                a = mx.transpose(a, (0, 2, 1))
            out.append((k, a))
    return out


def convert_hift(state: dict) -> list:
    return remap_hift_weights(state)


def _load(path):
    sd = torch.load(path, map_location="cpu")
    return sd.state_dict() if hasattr(sd, "state_dict") else sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=None,
                    help="BreezyVoice model dir (default: resolve the HF repo)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quantize", action="store_true",
                    help="quantize the LLM's Linear layers (flow/hift stay fp32)")
    ap.add_argument("--bits", type=int, default=4, help="quantization bits (4 or 8)")
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("MediaTek-Research/BreezyVoice")
    os.makedirs(args.out_dir, exist_ok=True)

    # flow + hift: plain fp32 conversion
    for name, conv in (("flow", convert_flow), ("hift", convert_hift)):
        src = os.path.join(model_dir, f"{name}.pt")
        print(f"[convert] {name}.pt ...")
        weights = dict(conv(_load(src)))
        dst = os.path.join(args.out_dir, f"{name}.safetensors")
        mx.save_safetensors(dst, weights)
        print(f"          -> {dst}  ({len(weights)} tensors)")

    # llm: fp32, or 4-bit quantized
    print("[convert] llm.pt ...")
    llm_weights = dict(convert_llm(_load(os.path.join(model_dir, "llm.pt"))))
    if args.quantize:
        import json
        from mlx.utils import tree_flatten
        from breezyvoice_mlx.cosyvoice import build_models
        from breezyvoice_mlx.quantize import quantize_llm
        m = build_models()
        m.llm.load_weights(list(llm_weights.items()), strict=False)
        quantize_llm(m.llm, args.bits, args.group_size)
        qweights = dict(tree_flatten(m.llm.parameters()))
        mx.save_safetensors(os.path.join(args.out_dir, "llm.safetensors"), qweights)
        with open(os.path.join(args.out_dir, "quant_config.json"), "w") as f:
            json.dump({"bits": args.bits, "group_size": args.group_size,
                       "scope": "llm.Linear"}, f)
        print(f"          -> llm.safetensors  ({len(qweights)} tensors, {args.bits}-bit Linear)")
    else:
        mx.save_safetensors(os.path.join(args.out_dir, "llm.safetensors"), llm_weights)
        print(f"          -> llm.safetensors  ({len(llm_weights)} tensors, fp32)")

    print("done. Also copy spk2info.pt (+ cosyvoice.yaml / ONNX models for the "
          "arbitrary-prompt frontend).")


if __name__ == "__main__":
    main()
