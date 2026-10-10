#!/usr/bin/env python3
"""Diff the engine tokenizer/renderer against the HuggingFace reference.

The engine side is produced by replays of `loopcd tokenize` and `loopcd render` (see the task steps);
this script only compares the collected JSON.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

EXE = "build-vk/loopcd.exe"
MODEL = "models/ouro/ouro.f16.gguf"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_tokenize(text):
    r = subprocess.run([EXE, "tokenize", "-m", MODEL, "--text", text], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"tokenize failed: {r.stderr}")
    line = [l for l in r.stdout.splitlines() if l.startswith("ids:")][0]
    body = line[len("ids:"):].strip()
    return [int(x) for x in body.split(",") if x != ""]


def run_render(messages, enable_thinking):
    cmd = [EXE, "render"]
    for m in messages:
        cmd += ["--msg", f"{m['role']}={m['content']}"]
    if not enable_thinking:
        cmd += ["--no-think"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"render failed: {r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="tests/reference/out/tok_reference.json")
    args = ap.parse_args()

    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    fail = False

    for case in ref["strings"]:
        got = run_tokenize(case["text"])
        if got != case["ids"]:
            fail = True
            print(f"FAIL tokenize {case['text']!r}: engine={got} ref={case['ids']}")
        else:
            print(f"ok   tokenize {case['text']!r} ({len(got)} ids)")

    for case in ref["conversations"]:
        got = run_render(case["messages"], case["enable_thinking"])
        if got != case["rendered"]:
            fail = True
            print(f"FAIL render {case['messages']}:")
            print(f"  engine: {got!r}")
            print(f"  ref   : {case['rendered']!r}")
        else:
            print(f"ok   render {case['messages']} (thinking={case['enable_thinking']})")

    print("TOKENIZER/TEMPLATE: FAIL" if fail else "TOKENIZER/TEMPLATE: PASS")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
