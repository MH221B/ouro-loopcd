#!/usr/bin/env python3
"""Dump reference tensors from HuggingFace transformers for the loopcd gate.

Writes tests/reference/out/<id>.npz with: h1, hR ([n, H]), zR, z1 ([n, V]), input_ids ([n]).
Overrides the loop count so we can gate at R=4 and R=2.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ap.add_argument("--prompts", default="tests/reference/prompts.json")
    ap.add_argument("--out", default="tests/reference/out")
    ap.add_argument("--loops", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    force_loops(model, args.loops)

    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise SystemExit("could not find model.norm; inspect the custom modeling file and fix this path")

    head = model.get_output_embeddings()
    assert head is not None, "no output embedding (lm_head) found"

    prompts = json.loads(Path(args.prompts).read_text())
    for pr in prompts:
        pid = pr["id"]
        enc = tok(pr["text"], return_tensors="pt")
        input_ids = enc["input_ids"]

        caps = []
        handle = norm.register_forward_hook(lambda mod, i, o: caps.append(o.detach().float().cpu().numpy()))
        # The custom Ouro forward does not implement output_hidden_states: passing it runs the
        # whole forward then raises TypeError, which would double-run the model. The loop-norm
        # states we need are captured by the hook, so call it plainly.
        with torch.no_grad():
            outputs = model(input_ids)
        handle.remove()

        assert len(caps) == args.loops, (
            f"{pid}: model.norm fired {len(caps)} times, expected {args.loops}. "
            f"Loop count override did not take effect."
        )

        logits = outputs.logits.float().cpu().numpy()[0]           # [n, V]
        h1 = caps[0][0]                                            # [n, H]
        hR = caps[-1][0]                                           # [n, H]

        with torch.no_grad():
            z1 = head(torch.from_numpy(h1)).float().cpu().numpy()
            zR = head(torch.from_numpy(hR)).float().cpu().numpy()

        # self-check: our hR must reproduce the model's own final logits
        d = float(np.abs(zR - logits).max())
        if d > 2e-2:
            raise SystemExit(f"{pid}: head(hR) != logits (max|d|={d:.4g}); tap assumption is wrong")

        np.savez_compressed(out / f"{pid}.r{args.loops}.npz",
                            h1=h1, hR=hR, zR=zR, z1=z1, input_ids=input_ids.numpy()[0])
        (out / f"{pid}.r{args.loops}.tokens.txt").write_text(
            ",".join(str(int(t)) for t in input_ids.numpy()[0])
        )
        print(f"{pid} r{args.loops}: n={input_ids.shape[1]} head-match max|d|={d:.3g} -> {out}")


if __name__ == "__main__":
    main()
