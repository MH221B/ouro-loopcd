#!/usr/bin/env python3
"""Diff a loopcd dump against a HuggingFace reference npz.

Gate metric (M2):
  * magnitude - keys in `mag` are gated against their bound; keys in `info` are reported
    against an informational ceiling and do NOT fail the gate.
  * argmax    - enforced ONLY for keys in `enforce_argmax` (default ["zR"]). The model's
    output key is zR; z1 is an intermediate (the LoopCD reference) whose near-tie argmax
    can flip under weight/ KV quantization, so it is reported, not enforced.
A flat {key: bound} tolerance file (the M1/G1 `tests/gate_tol.json`) is accepted and treated
as `mag` with argmax enforced on zR.
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
    raw = json.loads(Path(args.tol).read_text())

    if any(k in raw for k in ("mag", "info", "enforce_argmax")):
        mag            = raw.get("mag", {})
        info           = raw.get("info", {})
        enforce_argmax = set(raw.get("enforce_argmax", ["zR"]))
        note           = raw.get("_note", "")
    else:
        mag, info      = raw, {}
        enforce_argmax = {"zR"}
        note           = ""

    # engine stores [dim, n]; reference stores [n, dim] -> transpose when comparing
    pairs = [
        ("zR", ref["zR"], True),
        ("z1", ref["z1"], True),
        ("h1", ref["h1"], False),
        ("hR", ref["hR"], False),
    ]

    fail = False
    for key, r, is_logits in pairs:
        if key not in mag and key not in info:
            print(f"skip {key}: not gated")
            continue
        e = load_bin(meta, key)            # [n, dim]
        if e.shape != r.shape:
            print(f"FAIL {key}: shape {e.shape} vs {r.shape}")
            fail = True
            continue

        d = float(np.abs(e - r).max())

        am = ""
        same = True
        if is_logits:
            am_e = e.argmax(axis=-1)
            am_r = r.argmax(axis=-1)
            same = bool((am_e == am_r).all())
            nflip = int((am_e != am_r).sum())
            enforced = key in enforce_argmax
            am = f"argmax_match={same} flips={nflip}" + (" (enforced)" if enforced else " (info)")

        if key in mag:
            ok = d <= mag[key]
            if is_logits and key in enforce_argmax and not same:
                ok = False
            print(f"{'ok  ' if ok else 'FAIL'} {key}: max|d|={d:.4g} tol={mag[key]}" + (f" {am}" if am else ""))
            fail = fail or not ok
        else:
            # informational magnitude: reported against a ceiling, never fails on magnitude
            within = d <= info[key]
            print(f"info {key}: max|d|={d:.4g} ceiling={info[key]} within={within}" + (f" {am}" if am else ""))
            if is_logits and key in enforce_argmax and not same:
                fail = True   # argmax of an enforced key still gates

    if note:
        print(f"note: {note}")
    print("GATE: FAIL" if fail else "GATE: PASS")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
