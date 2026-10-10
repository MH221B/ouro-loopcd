#!/usr/bin/env python3
"""Run the pinned task set through the loopcd server under each LoopCD condition.

For one config (from tests/eval/configs.json) it starts `loopcd serve`, waits for /health, then issues
one deterministic (/v1/chat/completions, temperature 0) request per (item, condition) and records the
generated text, the wall time, the completion tokens, and whether the item scored.

Usage:
    python tests/eval/run_eval.py --config f32_clean [--limit N] [--conditions off,logits,...]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from score import score_math, score_mc, score_gen

# task -> (items file, scorer, preset (omega, omega_max))
TASKS = {
    "math": ("tests/eval/items/math.json", lambda t, it: score_math(t, it["gold"]), (0.5, 1.5)),
    "mc":   ("tests/eval/items/mc.json",   lambda t, it: score_mc(t, it["gold"]),   (0.5, 1.0)),
    "gen":  ("tests/eval/items/gen.json",  lambda t, it: score_gen(t, it["checks"]), (0.3, 1.0)),
}

ALL_CONDITIONS = ["off", "logits", "logits_adaptive", "hidden"]


def loopcd_for(condition, task):
    omega, cap = TASKS[task][2]
    if condition == "off":
        return {"mode": "off"}
    if condition == "logits":
        return {"mode": "logits", "omega": omega, "omega_max": cap}
    if condition == "logits_adaptive":
        return {"mode": "logits", "omega_max": cap, "adaptive": True}
    if condition == "hidden":
        return {"mode": "hidden", "omega": 0.5}
    raise ValueError(f"unknown condition: {condition}")


def start_server(exe, cfg, port, loops=0):
    cmd = [exe, "serve", "-m", cfg["model"], "--device", cfg["device"],
           "--kv-type", cfg["kv_type"], "--ctx", str(cfg.get("ctx", 1024)), "--port", str(port)]
    if loops > 0:
        cmd += ["--loops", str(loops)]
    print("+", " ".join(cmd))
    log = open(f"tests/eval/out/{cfg['name']}.server.log", "w", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    base = f"http://127.0.0.1:{port}"
    # generous: the clean config loads a 10 GB f32 GGUF before /health answers
    for _ in range(600):
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return proc, base
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise SystemExit("server did not become ready")


def chat(base, messages, enable_thinking, max_tokens, loopcd, seed=1234):
    body = {"messages": messages, "enable_thinking": enable_thinking, "max_tokens": max_tokens,
            "temperature": 0, "seed": seed, "extra_body": {"loopcd": loopcd}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        j = json.loads(r.read().decode())
    dt = time.time() - t0
    return j["choices"][0]["message"]["content"], dt, j.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exe", default="build-vk/loopcd.exe")
    ap.add_argument("--configs", default="tests/eval/configs.json")
    ap.add_argument("--port", type=int, default=18082)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N items per task")
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--tasks", default=",".join(TASKS.keys()))
    ap.add_argument("--loops", type=int, default=0, help="override the recurrent loop count (R)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfgs = {c["name"]: c for c in json.loads(Path(args.configs).read_text())["configs"]}
    if args.config not in cfgs:
        raise SystemExit(f"unknown config {args.config!r}; have {sorted(cfgs)}")
    cfg = cfgs[args.config]
    conditions = [c for c in args.conditions.split(",") if c]
    tasks = [t for t in args.tasks.split(",") if t]

    loops = args.loops or cfg.get("loops", 0)
    proc, base = start_server(args.exe, cfg, args.port, loops)
    results = {"config": args.config, "model": cfg["model"], "device": cfg["device"],
               "kv_type": cfg["kv_type"], "conditions": conditions, "loops": loops, "items": []}
    try:
        for task in tasks:
            items_path, scorer, _ = TASKS[task]
            items = json.loads(Path(items_path).read_text())
            if args.limit:
                items = items[:args.limit]
            for item in items:
                for cond in conditions:
                    text, dt, usage = chat(base, item["messages"], item.get("enable_thinking", False),
                                           item["max_tokens"], loopcd_for(cond, task))
                    ok = bool(scorer(text, item))
                    results["items"].append({
                        "task": task, "id": item["id"], "condition": cond, "ok": ok,
                        "elapsed_s": dt, "completion_tokens": usage.get("completion_tokens"),
                        "text": text,
                    })
                    print(f"{cfg['name']} {task}/{item['id']} {cond}: {'PASS' if ok else 'FAIL'} "
                          f"({dt:.1f}s, {usage.get('completion_tokens', '?')} tok)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    out = Path(args.out or f"tests/eval/out/{args.config}.results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
