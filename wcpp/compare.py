#!/usr/bin/env python3
"""Cross-engine comparison for Breeze-ASR-26 on Apple Silicon.

Runs the same audio through whisper.cpp (any ggml quantization) and mlx-whisper
with *decoding parameters forced to match*, so output differences reflect weights
and numerics rather than search strategy.

Why the forcing matters: whisper-cli defaults to beam search (beam=5, best_of=5)
while mlx_whisper defaults to greedy. Both default to temperature fallback, which
retries a failed decode at temperature>0 and makes output nondeterministic.
Comparing engines under their own defaults measures the schedulers, not the models.

Usage:
  compare.py --clips a.wav b.wav --ggml out/ggml-model.bin out/ggml-breeze-q5_k.bin
  compare.py --clips a.wav --ggml out/ggml-model.bin --mlx --reference out/oracle.json
"""
import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHISPER_CLI = HERE / "whisper.cpp" / "build" / "bin" / "whisper-cli"

MLX_MODELS = {
    "mlx-4bit": "doggy8088/Breeze-ASR-26-MLX-4bit",
    "mlx-fp16": "doggy8088/Breeze-ASR-26-MLX",
}


# ---------------------------------------------------------------- text metrics

def normalize(text: str) -> str:
    """Strip what shouldn't count toward CER.

    Chinese output has no word spaces, and the engines disagree about punctuation
    and about inserting spaces at segment joins. None of that is a recognition
    error, so remove it before scoring.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return "".join(c for c in text if not unicodedata.category(c).startswith("P"))


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return float("nan")
    return edit_distance(h, r) / len(r)


def duration_s(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


# ------------------------------------------------------------------- engines

def run_ggml(model: Path, audio: str, language: str, threads: int,
             greedy: bool) -> dict:
    """whisper.cpp via whisper-cli."""
    cmd = [
        str(WHISPER_CLI), "-m", str(model), "-f", audio,
        "-l", language, "-t", str(threads),
        "-nt",  # no timestamps -> plain text on stdout
    ]
    if greedy:
        cmd += [
            "-bs", "1",    # no beam search
            "-bo", "1",    # single candidate
            "-tp", "0.0",  # temperature 0
            "-nf",         # no temperature fallback -> deterministic
            "-mc", "0",    # don't condition on previous text
        ]
    t0 = time.time()
    # Capture bytes, not text: whisper.cpp emits tokens incrementally and a BPE
    # token can be a fragment of a multi-byte CJK character, so the stream can end
    # mid-sequence. text=True raises UnicodeDecodeError on exactly that.
    proc = subprocess.run(cmd, capture_output=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr.decode("utf-8", "replace")[-2000:], "wall_s": wall}
    out = proc.stdout.decode("utf-8", "replace")
    text = " ".join(ln.strip() for ln in out.splitlines() if ln.strip())
    return {"text": text.strip(), "wall_s": wall}


def run_mlx(repo: str, audio: str, language: str, greedy: bool) -> dict:
    """mlx-whisper in-process."""
    import mlx_whisper

    kwargs = dict(path_or_hf_repo=repo, language=language)
    if greedy:
        kwargs.update(
            temperature=0.0,               # scalar -> no fallback ladder
            condition_on_previous_text=False,
        )
    t0 = time.time()
    res = mlx_whisper.transcribe(audio, **kwargs)
    wall = time.time() - t0
    return {"text": res["text"].strip(), "wall_s": wall}


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--ggml", nargs="*", default=[], help="ggml model paths")
    ap.add_argument("--mlx", nargs="*", default=[],
                    help=f"mlx variants to run, from {list(MLX_MODELS)}")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reference", help="oracle JSON from hf_oracle.py")
    ap.add_argument("--no-greedy", action="store_true",
                    help="use each engine's own defaults instead of matched greedy")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    greedy = not args.no_greedy

    refs = {}
    if args.reference:
        for row in json.load(open(args.reference)):
            refs[Path(row["file"]).name] = row["text"]

    engines = [(Path(p).name.replace("ggml-", "").replace(".bin", ""), "ggml", p)
               for p in args.ggml]
    engines += [(name, "mlx", MLX_MODELS[name]) for name in args.mlx]

    mode = "matched greedy (deterministic)" if greedy else "engine defaults"
    print(f"decoding: {mode} | language={args.language} | threads={args.threads}\n")

    results = []
    for clip in args.clips:
        dur = duration_s(clip)
        ref = refs.get(Path(clip).name)
        print("=" * 78)
        print(f"{Path(clip).name}  ({dur:.1f}s)")
        print("=" * 78)
        if ref:
            print(f"  [oracle  ] {ref}")
        for label, kind, target in engines:
            if kind == "ggml":
                r = run_ggml(Path(target), clip, args.language, args.threads, greedy)
            else:
                r = run_mlx(target, clip, args.language, greedy)
            if "error" in r:
                print(f"  [{label:9s}] FAILED: {r['error'][:300]}")
                continue
            xrt = r["wall_s"] / dur
            line = f"  [{label:9s}] {r['text']}"
            print(line)
            score = f"    wall {r['wall_s']:6.1f}s  xRT {xrt:5.2f}"
            if ref:
                score += f"  CER vs oracle {cer(r['text'], ref) * 100:6.2f}%"
            print(score)
            results.append({
                "clip": Path(clip).name, "audio_s": dur, "engine": label,
                "text": r["text"], "wall_s": r["wall_s"], "xrt": xrt,
                "cer_vs_oracle": cer(r["text"], ref) if ref else None,
            })
        print()

    # pairwise agreement between engines on each clip
    if len(engines) > 1:
        print("=" * 78)
        print("pairwise character disagreement (lower = engines agree)")
        print("=" * 78)
        for clip in {r["clip"] for r in results}:
            rows = [r for r in results if r["clip"] == clip]
            print(f"\n{clip}")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    d = cer(rows[i]["text"], rows[j]["text"]) * 100
                    print(f"  {rows[i]['engine']:9s} vs {rows[j]['engine']:9s}  {d:6.2f}%")

    if args.json:
        json.dump(results, open(args.json, "w"), ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
