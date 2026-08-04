"""End-to-end BreezyVoice-MLX SFT synthesis (text + built-in speaker -> wav).

Usage:
    PYTHONPATH=breezyvoice_mlx .venv/bin/python breezyvoice_mlx/tools/run_sft.py \
        --weights converted_mlx --text "歡迎使用聯發創新基地 BreezyVoice 模型。" \
        --spk 中文女 --out results/breezyvoice_mlx_sft.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from breezyvoice_mlx.cosyvoice import BreezyVoice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="dir with {llm,flow,hift}.safetensors")
    ap.add_argument("--model-dir", default=None, help="dir with spk2info.pt (default: HF cache)")
    ap.add_argument("--text", default="歡迎使用聯發創新基地 BreezyVoice 模型。")
    ap.add_argument("--spk", default="中文女")
    ap.add_argument("--out", default="results/breezyvoice_mlx_sft.wav")
    ap.add_argument("--mode", default="zero_shot", choices=["zero_shot", "sft"],
                    help="zero_shot uses spk2info prompt token+feat (recommended)")
    args = ap.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("MediaTek-Research/BreezyVoice")

    print("[run] building + loading MLX models ...")
    bv = BreezyVoice(model_dir, weights_dir=args.weights)
    print("[run] speakers:", bv.list_speakers())

    t0 = time.time()
    fn = bv.inference_zero_shot_builtin if args.mode == "zero_shot" else bv.inference_sft
    wav = np.array(fn(args.text, args.spk)).reshape(-1)
    dt = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, bv.SAMPLE_RATE)
    dur = len(wav) / bv.SAMPLE_RATE
    rms = float(np.sqrt(np.mean(wav ** 2)))
    print(f"[run] wrote {args.out}: {dur:.2f}s @ {bv.SAMPLE_RATE}Hz, "
          f"gen {dt:.1f}s (RTF {dt/max(dur,1e-6):.2f}), RMS {rms:.4f}")


if __name__ == "__main__":
    main()
