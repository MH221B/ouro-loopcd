#!/usr/bin/env python3
"""Diff a loopcd dump against a HuggingFace reference npz.

Positional check: also verifies the argmax of every position matches, to catch sign/permutation bugs
that a loose max-abs tolerance can hide.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_bin(meta, key):
    # engine tensors are [dim, n] in ne0-fastest order; return them as [n, dim]
    t = meta["tensors"][key]
    shape = tuple(t["shape"])          # [dim, n]
    a = np.fromfile(t["file"], dtype="<f4")
    return a.reshape(shape[::-1])      # [n, dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="prefix passed to --dump")
    ap.add_argument("--ref", required=True, help="reference .npz")
    ap.add_argument("--tol", default="tests/gate_tol.json")
    args = ap.parse_args()

    meta = json.loads(Path(args.dump + ".meta.json").read_text())
    ref = np.load(args.ref)
    tol = json.loads(Path(args.tol).read_text())

    # engine stores [dim, n]; reference stores [n, dim] -> transpose when comparing
    pairs = [
        ("zR", ref["zR"], True),
        ("z1", ref["z1"], True),
        ("h1", ref["h1"], False),
        ("hR", ref["hR"], False),
    ]

    fail = False
    for key, r, is_logits in pairs:
        e = load_bin(meta, key)            # [n, dim]
        if e.shape != r.shape:
            print(f"FAIL {key}: shape {e.shape} vs {r.shape}")
            fail = True
            continue
        d = float(np.abs(e - r).max())
        ok = d <= tol[key]
        note = ""
        if is_logits:
            same = bool((e.argmax(axis=-1) == r.argmax(axis=-1)).all())
            note = f" argmax_match={same}"
            if not same:
                ok = False
        print(f"{'ok  ' if ok else 'FAIL'} {key}: max|d|={d:.4g} tol={tol[key]}{note}")
        fail = fail or not ok

    print("GATE: FAIL" if fail else "GATE: PASS")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
