"""A/B: fp32 LLM vs 4-bit-quantized LLM (flow/vocoder identical fp32 in both).

Quantization only affects the LLM. NOTE: greedy decode degenerates this model
(CosyVoice-style LLMs need sampling), so token-argmax is not a valid metric.
We measure quality two ways:
  1. LLM logit fidelity (deterministic teacher-forced prompt forward): top-1 and
     top-5 agreement + softmax cosine between fp32 and 4-bit. This is the direct
     quantization-error metric.
  2. Real sampled audio from each (top-k) — confirms both produce non-degenerate
     speech; listen to results/ab_{fp32,q4}.wav.
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import soundfile as sf

from breezyvoice_mlx.cosyvoice import BreezyVoice

QDIR = sys.argv[1] if len(sys.argv) > 1 else "converted_mlx_q4"
QTAG = sys.argv[2] if len(sys.argv) > 2 else "q4"
TEXT = sys.argv[3] if len(sys.argv) > 3 else "歡迎使用聯發創新基地 BreezyVoice 模型。"
SPK = sys.argv[4] if len(sys.argv) > 4 else "中文女"
from huggingface_hub import snapshot_download
MD = snapshot_download("MediaTek-Research/BreezyVoice")
SR = 22050


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def run(weights_dir, tag):
    bv = BreezyVoice(MD, weights_dir=weights_dir)
    logits = np.array(bv.llm_prompt_logits(TEXT, SPK))[0]           # (T, V) deterministic
    wav = np.array(bv.inference_zero_shot_builtin(TEXT, SPK)).reshape(-1)  # sampled
    sz = sum(os.path.getsize(os.path.join(weights_dir, f)) for f in os.listdir(weights_dir)
             if f.endswith(".safetensors")) / 1e6
    os.makedirs("results", exist_ok=True)
    sf.write(f"results/ab_{tag}.wav", wav, SR)
    print(f"[{tag}] quant={bv.quantized} disk={sz:.0f}MB | wav {len(wav)/SR:.2f}s "
          f"rms={np.sqrt((wav**2).mean()):.3f} clip={np.mean(np.abs(wav)>=.985)*100:.1f}%")
    del bv; gc.collect()
    return logits


lp = run("converted_mlx", "fp32")
lq = run(QDIR, QTAG)

n = min(len(lp), len(lq))
lp, lq = lp[:n], lq[:n]
top1 = float(np.mean(lp.argmax(-1) == lq.argmax(-1))) * 100
t5p = np.argsort(-lp, -1)[:, :5]
t5q = np.argsort(-lq, -1)[:, :5]
top5 = float(np.mean([len(set(a) & set(b)) for a, b in zip(t5p, t5q)]) / 5) * 100
pp, qq = softmax(lp), softmax(lq)
cos = float(np.mean(np.sum(pp * qq, -1) / (np.linalg.norm(pp, axis=-1) * np.linalg.norm(qq, axis=-1) + 1e-9))) * 100
print("\n=== LLM logit fidelity (4-bit vs fp32, teacher-forced, deterministic) ===")
print(f"positions={n}  top-1 agreement={top1:.1f}%  top-5 overlap={top5:.1f}%  "
      f"softmax cosine={cos:.2f}%")
