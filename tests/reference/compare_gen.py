#!/usr/bin/env python3
"""Diff the engine's greedy continuation against the HuggingFace reference (spec 7.2)."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="file of comma-separated engine token ids")
    ap.add_argument("--ref", required=True, help="JSON written by gen_reference.py")
    args = ap.parse_args()

    raw = Path(args.engine).read_text().replace("\n", "")
    engine = [int(t) for t in raw.split(",") if t.strip() != ""]
    ref = json.loads(Path(args.ref).read_text())["gen_ids"]

    # The engine (M4 generator) stops at EOS WITHOUT emitting it; HF `generate` includes the
    # trailing EOS token in the sequence. Normalize: if the engine equals the reference prefix
    # exactly and is one token shorter, the reference's last token is the stop token.
    if len(engine) == len(ref) - 1 and engine == ref[:-1]:
        print(f"note: engine stopped at the stop token; reference trailing id {ref[-1]} dropped")
        ref = ref[:-1]

    for i in range(min(len(engine), len(ref))):
        if engine[i] != ref[i]:
            print(f"MISMATCH at step {i}: engine={engine[i]} ref={ref[i]}")
            print(f"  engine: {engine}")
            print(f"  ref   : {ref}")
            print("GEN: FAIL")
            raise SystemExit(1)
    if len(engine) != len(ref):
        print(f"LENGTH mismatch: engine={len(engine)} ref={len(ref)}")
        print(f"  engine: {engine}")
        print(f"  ref   : {ref}")
        print("GEN: FAIL")
        raise SystemExit(1)

    print(f"greedy continuation matches HF: {len(engine)} tokens")
    print("GEN: PASS")


if __name__ == "__main__":
    main()
