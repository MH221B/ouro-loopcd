#!/usr/bin/env python3
"""Seeded LoopCD on-vs-off check (spec 7.4).

Runs `loopcd run` on a pinned prompt with greedy decoding and `--dump-guided`, then checks the
guided-logit relationships fixed by the math:

  - off    : z' == zR (guidance disabled)
  - logits : z' == zR + omega*(zR - z1)   (Eq. 3) and z' != zR
  - hidden : z' != logits z' in general (pre-norm combine + output_norm), and both forms == zR at omega=0
  - deterministic: a second logits run is byte-identical

Usage:
    python tests/reference/loopcd_onoff.py --exe build-vk/loopcd.exe \
        --model models/ouro/ouro.f32.gguf --tokens-file tests/reference/out/p0.r4.tokens.txt
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(exe, model, tokens_file, mode, omega, out):
    cmd = [str(exe), "run", "-m", str(model), "--tokens-file", str(tokens_file),
           "--device", "cpu", "--kv-type", "f32", "--ctx", "256",
           "--max-tokens", "8", "--loopcd", mode, "--loopcd-omega", str(omega),
           "--dump-guided", str(out)]
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"engine exited {r.returncode} for mode={mode}")


def load(p):
    return np.fromfile(str(p), dtype="<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build-vk/loopcd.exe")
    ap.add_argument("--model", default="models/ouro/ouro.f32.gguf")
    ap.add_argument("--tokens-file", default="tests/reference/out/p0.r4.tokens.txt")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    omega = 0.5
    run(args.exe, args.model, args.tokens_file, "off",    0.0,   outdir / "lc_off")
    run(args.exe, args.model, args.tokens_file, "logits", omega, outdir / "lc_logits")
    run(args.exe, args.model, args.tokens_file, "hidden", omega, outdir / "lc_hidden")
    run(args.exe, args.model, args.tokens_file, "logits", 0.0,   outdir / "lc_logits0")
    run(args.exe, args.model, args.tokens_file, "hidden", 0.0,   outdir / "lc_hidden0")
    run(args.exe, args.model, args.tokens_file, "logits", omega, outdir / "lc_logits2")

    off    = load(outdir / "lc_off.guided.bin")
    off_u  = load(outdir / "lc_off.unguided.bin")
    lg     = load(outdir / "lc_logits.guided.bin")
    lg_u   = load(outdir / "lc_logits.unguided.bin")
    lg_1   = load(outdir / "lc_logits.z1.bin")
    hd     = load(outdir / "lc_hidden.guided.bin")
    lg0    = load(outdir / "lc_logits0.guided.bin")
    lg0_u  = load(outdir / "lc_logits0.unguided.bin")
    hd0    = load(outdir / "lc_hidden0.guided.bin")
    hd0_u  = load(outdir / "lc_hidden0.unguided.bin")
    lg2    = load(outdir / "lc_logits2.guided.bin")

    fail = False

    def report(name, ok, detail):
        nonlocal fail
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        fail = fail or not ok

    report("off guided == unguided", bool(np.array_equal(off, off_u)),
           f"max|d|={float(np.abs(off - off_u).max()):.3g}")

    want = lg_u + omega * (lg_u - lg_1)
    report("logits z' == zR + omega*(zR - z1)",
           float(np.abs(lg - want).max()) < 1e-3, f"max|d|={float(np.abs(lg - want).max()):.3g}")
    report("logits z' != zR", float(np.abs(lg - lg_u).max()) > 1e-6,
           f"max|d|={float(np.abs(lg - lg_u).max()):.3g}")
    report("hidden z' != logits z' (pre-norm combine)",
           float(np.abs(hd - lg).max()) > 1e-3, f"max|d|={float(np.abs(hd - lg).max()):.3g}")
    # omega = 0: both forms coincide with the unguided logits
    report("logits omega=0 == zR", float(np.abs(lg0 - lg0_u).max()) < 1e-4,
           f"max|d|={float(np.abs(lg0 - lg0_u).max()):.3g}")
    report("hidden omega=0 == zR", float(np.abs(hd0 - hd0_u).max()) < 1e-3,
           f"max|d|={float(np.abs(hd0 - hd0_u).max()):.3g}")
    report("deterministic logits run", bool(np.array_equal(lg, lg2)),
           f"max|d|={float(np.abs(lg - lg2).max()):.3g}")

    print("LOOPCD ON/OFF: FAIL" if fail else "LOOPCD ON/OFF: PASS")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
