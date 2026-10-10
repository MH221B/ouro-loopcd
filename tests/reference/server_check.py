#!/usr/bin/env python3
"""End-to-end server check: /health, /v1/models, /v1/chat/completions (stream + non-stream), /v1/completions."""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = None


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return r.status, r.read().decode()


def post(path, body, stream=False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return r.status, r.read().decode()
        lines = []
        for raw in r:
            lines.append(raw.decode(errors="replace"))
        return r.status, "".join(lines)


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build-vk/loopcd.exe")
    ap.add_argument("--model", default="models/ouro/ouro.f16.gguf")
    ap.add_argument("--port", type=int, default=18081)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    proc = subprocess.Popen([args.exe, "serve", "-m", args.model, "--device", "cpu", "--kv-type", "f32",
                             "--ctx", "512", "--port", str(args.port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fail = False
    try:
        for _ in range(120):
            try:
                if get("/health")[0] == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server did not become ready")

        st, body = get("/health")
        ok = st == 200 and json.loads(body)["status"] == "ok"
        print(("ok  " if ok else "FAIL") + " /health")
        fail = fail or not ok

        st, body = get("/v1/models")
        ok = st == 200 and json.loads(body)["data"][0]["id"] == "ouro-2.6b-thinking"
        print(("ok  " if ok else "FAIL") + " /v1/models")
        fail = fail or not ok

        chat = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "temperature": 0}
        st, body = post("/v1/chat/completions", chat)
        j = json.loads(body)
        ok = st == 200 and "choices" in j and j["choices"][0]["message"]["role"] == "assistant"
        print(("ok  " if ok else "FAIL") + " /v1/chat/completions (non-stream)")
        fail = fail or not ok

        st, sbody = post("/v1/chat/completions", {**chat, "stream": True}, stream=True)
        ok = st == 200 and "data: [DONE]" in sbody and any(l.startswith("data: {") for l in sbody.splitlines())
        print(("ok  " if ok else "FAIL") + " /v1/chat/completions (stream)")
        fail = fail or not ok

        st, body = post("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 4, "temperature": 0})
        ok = st == 200 and "choices" in json.loads(body)
        print(("ok  " if ok else "FAIL") + " /v1/completions")
        fail = fail or not ok

        # context overflow -> 400
        big = "word " * 2000
        req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps({"prompt": big, "max_tokens": 4}).encode(), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=60)
            ok = False
        except urllib.error.HTTPError as e:
            ok = e.code == 400
        print(("ok  " if ok else "FAIL") + " context overflow -> 400")
        fail = fail or not ok
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    print("SERVER: FAIL" if fail else "SERVER: PASS")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
