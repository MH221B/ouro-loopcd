#!/usr/bin/env python3
"""Reference greedy continuation for the loopcd decode gate (spec 7.2).

Reads a comma-separated prompt token file, greedily continues `max_new` tokens with
transformers `generate(do_sample=False)`, and cross-checks the result against a
teacher-forced greedy loop (full forward each step, no KV cache). Writes a JSON with
the prompt ids, the generated ids, and the full sequence.

    python tests/reference/gen_reference.py \
        --tokens-file tests/reference/out/p0.r4.tokens.txt --max-new 32
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def force_loops(model, n):
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        cfg = getattr(root, "config", None)
        if cfg is not None and hasattr(cfg, "total_ut_steps"):
            cfg.total_ut_steps = n
        if hasattr(root, "total_ut_steps"):
            root.total_ut_steps = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ouro")
    ap.add_argument("--tokens-file", required=True)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--out", default="tests/reference/out/p0.gen.json")
    args = ap.parse_args()

    raw = Path(args.tokens_file).read_text().replace("\n", "")
    prompt = [int(t) for t in raw.split(",") if t.strip() != ""]
    assert prompt, f"no token ids in {args.tokens_file}"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    force_loops(model, args.loops)
    eos = model.config.eos_token_id

    inp = torch.tensor([prompt], dtype=torch.long)

    # spec: greedy generate
    with torch.no_grad():
        gen_out = model.generate(inp, do_sample=False, max_new_tokens=args.max_new)
    seq = gen_out.sequences[0] if hasattr(gen_out, "sequences") else gen_out[0]
    full = seq.tolist()
    gen = full[len(prompt):]

    # cross-check: teacher-forced greedy (independent of HF's cache wiring)
    tf = []
    cur = prompt[:]
    with torch.no_grad():
        for _ in range(args.max_new):
            logits = model(torch.tensor([cur], dtype=torch.long)).logits[0, -1]
            nxt = int(torch.argmax(logits).item())
            tf.append(nxt)
            cur.append(nxt)
            if nxt == eos:
                break
    if gen != tf:
        raise SystemExit(
            f"generate() != teacher-forced greedy:\n  generate: {gen}\n  teacher : {tf}\n"
            "The HF cache path for this model is unreliable; fix the reference before gating."
        )

    out = {"prompt_ids": prompt, "gen_ids": gen, "full_ids": full}
    Path(args.out).write_text(json.dumps(out))
    print(f"prompt={len(prompt)} gen={len(gen)} -> {args.out}")
    print("gen_ids:", ",".join(str(t) for t in gen))


if __name__ == "__main__":
    main()
