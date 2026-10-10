#!/usr/bin/env python3
"""Assert the converted GGUF has the Ouro shape we expect.

Expects: arch "ouro", 531 tensors (533 weights minus 2 early_exit_gate),
ouro.num_loops=4, ouro.skip_loop_final_norm=false, and no early_exit_gate tensor.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\tnmh\projects\llama.cpp-ouro\gguf-py")
from gguf import GGUFReader  # noqa: E402

GLOBAL = ["token_embd.weight", "output_norm.weight", "output.weight"]
PER_LAYER = [
    "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
    "post_attention_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm",
]


def field(reader, key):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    reader = GGUFReader(args.path)
    names = {t.name for t in reader.tensors}

    arch = field(reader, "general.architecture")
    assert arch == "ouro", f"arch is {arch!r}, expected 'ouro'"

    expect = set(GLOBAL)
    for i in range(48):
        for p in PER_LAYER:
            expect.add(f"blk.{i}.{p}.weight")

    missing = expect - names
    extra = names - expect
    assert not missing, f"missing tensors: {sorted(missing)[:8]}"
    assert not extra, f"unexpected tensors: {sorted(extra)[:8]}"
    assert len(names) == 531, f"got {len(names)} tensors, expected 531"

    loops = field(reader, "ouro.num_loops")
    assert int(loops) == 4, f"ouro.num_loops = {loops}, expected 4"

    skip = field(reader, "ouro.skip_loop_final_norm")
    assert skip is False or skip == 0, f"ouro.skip_loop_final_norm = {skip}, expected false"

    assert not any("early_exit_gate" in n for n in names), "early_exit_gate was not dropped"

    print(f"OK: {Path(args.path).name}: arch=ouro, {len(names)} tensors, loops={int(loops)}")


if __name__ == "__main__":
    main()
