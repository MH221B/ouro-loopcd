#!/usr/bin/env python3
"""Fetch Ouro config, tokenizer and custom code from HF into models/ouro/.

Weights are already local (model.safetensors); do not re-download them.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "ByteDance/Ouro-2.6B-Thinking"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dst", default="models/ouro")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dst),
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"],
    )
    print("assets in", dst.resolve())
    for p in sorted(dst.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
