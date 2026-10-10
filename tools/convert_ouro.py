#!/usr/bin/env python3
"""Convert Ouro-2.6B-Thinking to GGUF.

Thin wrapper around the llama.cpp PR #29823 converter, since arch "ouro" is not upstream.
Pass --hf a local dir that holds config.json + model.safetensors (see fetch_assets.py).
"""
import argparse
import subprocess
import sys
from pathlib import Path

OURO_PR = Path(r"C:\Users\tnmh\projects\llama.cpp-ouro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="models/ouro", help="HF repo id or local dir")
    ap.add_argument("--out", default="models/ouro/ouro.f16.gguf")
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32"])
    args = ap.parse_args()

    conv = OURO_PR / "convert_hf_to_gguf.py"
    if not conv.exists():
        sys.exit(f"converter not found: {conv} (is the PR #29823 worktree checked out?)")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(conv), args.hf, "--outfile", str(out), "--outtype", args.outtype]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(OURO_PR))


if __name__ == "__main__":
    main()
