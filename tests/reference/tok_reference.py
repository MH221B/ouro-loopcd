#!/usr/bin/env python3
"""Dump the HuggingFace tokenizer ids and chat-template renders for the pinned cases (spec 7.3)."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ouro")
    ap.add_argument("--cases", default="tests/reference/conversations.json")
    ap.add_argument("--out", default="tests/reference/out/tok_reference.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    out = {"strings": [], "conversations": []}
    for s in cases["strings"]:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        out["strings"].append({"text": s, "ids": ids})

    for c in cases["conversations"]:
        rendered = tok.apply_chat_template(
            c["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=c.get("enable_thinking", False),
        )
        out["conversations"].append({"messages": c["messages"],
                                     "enable_thinking": c.get("enable_thinking", False),
                                     "rendered": rendered})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {len(out['strings'])} strings and {len(out['conversations'])} conversations -> {args.out}")


if __name__ == "__main__":
    main()
