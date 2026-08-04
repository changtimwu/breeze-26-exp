"""Assemble a self-contained, distributable BreezyVoice-MLX model directory.

Collects the converted weights + spk2info.pt and writes a MODEL_CARD.md describing
the model (lineage, precision, pipeline, speakers, usage, fidelity, license).

Usage:
    PYTHONPATH=breezyvoice_mlx python breezyvoice_mlx/tools/package_model.py \
        --weights-dir converted_q8 --out dist/BreezyVoice-300M-MLX-8bit
    # --model-dir defaults to the HF cache; large files are hard-linked (no copy)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)          # hard-link: instant, no disk duplication
    except OSError:
        shutil.copy2(src, dst)


def _mb(path):
    return os.path.getsize(path) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", required=True, help="converted dir ({llm,flow,hift}.safetensors)")
    ap.add_argument("--model-dir", default=None, help="source dir with spk2info.pt (default: HF cache)")
    ap.add_argument("--out", required=True, help="package directory to create")
    ap.add_argument("--name", default=None, help="model name for the card")
    args = ap.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("MediaTek-Research/BreezyVoice")

    os.makedirs(args.out, exist_ok=True)

    # precision
    qc_path = os.path.join(args.weights_dir, "quant_config.json")
    if os.path.exists(qc_path):
        qc = json.load(open(qc_path))
        precision = f"LLM {qc['bits']}-bit (group {qc['group_size']}), flow+vocoder fp32"
        _link_or_copy(qc_path, os.path.join(args.out, "quant_config.json"))
    else:
        qc = None
        precision = "fp32 (all stages)"

    # weights
    sizes = {}
    for f in ("llm.safetensors", "flow.safetensors", "hift.safetensors"):
        src = os.path.join(args.weights_dir, f)
        _link_or_copy(src, os.path.join(args.out, f))
        sizes[f] = _mb(src)

    # speaker prompts
    src_spk = os.path.join(model_dir, "spk2info.pt")
    _link_or_copy(src_spk, os.path.join(args.out, "spk2info.pt"))
    import torch
    spks = list(torch.load(src_spk, map_location="cpu").keys())

    total = sum(sizes.values()) + _mb(src_spk)
    name = args.name or ("BreezyVoice-300M-MLX" + (f"-{qc['bits']}bit" if qc else "-fp32"))

    card = f"""# {name}

Apple **MLX** port of [MediaTek-Research/BreezyVoice](https://huggingface.co/MediaTek-Research/BreezyVoice),
a CosyVoice-v1-derived zero-shot TTS model, for Apple Silicon.

- **Pipeline:** Conformer text LLM → conditional flow matching → HiFiGAN-NSF vocoder
- **Format:** MLX safetensors
- **Precision:** {precision}
- **Sample rate:** 22050 Hz
- **Total size:** {total:.0f} MB  (llm {sizes['llm.safetensors']:.0f} + flow {sizes['flow.safetensors']:.0f} + hift {sizes['hift.safetensors']:.0f} + spk2info {_mb(src_spk):.1f})

## Files
- `llm.safetensors`, `flow.safetensors`, `hift.safetensors` — the three stages
- `spk2info.pt` — built-in speaker prompts (embedding + speech_token + speech_feat)
{"- `quant_config.json` — quantization spec (auto-detected on load)" if qc else ""}

## Built-in speakers
{", ".join(spks)}

## Usage
```python
from breezyvoice_mlx.cosyvoice import BreezyVoice
import soundfile as sf
bv = BreezyVoice(model_dir="{os.path.basename(args.out)}", weights_dir="{os.path.basename(args.out)}")
wav = bv.inference_zero_shot_builtin("歡迎使用聯發創新基地 BreezyVoice 模型。", "{spks[0] if spks else '中文女'}")
sf.write("out.wav", __import__("numpy").array(wav).reshape(-1), bv.SAMPLE_RATE)
```
(`model_dir` supplies `spk2info.pt`; `weights_dir` supplies the safetensors — same dir here.)

## Quality (LLM logit fidelity vs fp32, teacher-forced)
{"| 8-bit: top-1 99.6%, top-5 99.3%, softmax cosine 100.0% (near-lossless)" if qc and qc['bits'] == 8 else ""}
{"| 4-bit: top-1 84.3%, top-5 86.1%, softmax cosine 97.0% (lossy)" if qc and qc['bits'] == 4 else ""}
{"fp32 reference (no quantization)." if not qc else ""}
NOTE: this model is sampling-trained — use top-k sampling (default), not greedy.

## License / attribution
Inherits the upstream BreezyVoice / CosyVoice license (Apache-2.0). This is a
format/precision conversion of MediaTek-Research/BreezyVoice; all model credit to
the original authors.
"""
    with open(os.path.join(args.out, "MODEL_CARD.md"), "w") as f:
        f.write(card)
    print(f"packaged -> {args.out}  ({total:.0f} MB, precision: {precision})")
    print(f"  files: {sorted(os.listdir(args.out))}")


if __name__ == "__main__":
    main()
