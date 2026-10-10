# loopcd

A minimal, single-model inference engine for **Ouro-2.6B-Thinking** with **LoopCD**
contrastive decoding — written from scratch in C++ on top of [ggml](https://github.com/ggml-org/ggml),
running on a local **AMD Radeon 890M iGPU** through **Vulkan**, and serving an
OpenAI-compatible chat endpoint.

It is deliberately **not** a llama.cpp fork. It links ggml and *copies* a few specific
pieces from llama.cpp PR #29823 (which adds Ouro support): the Ouro graph, the Ouro GGUF
converter, and the BPE tokenizer. The engine is Ouro-only — one architecture, one model.

The point of the project is to **verify a paper's claim on my own hardware** rather than
trust it, and to measure what the technique actually costs to run.

---

## Background & inspiration

**The model — Ouro.** [*Scaling Latent Reasoning via Looped Language Models*](https://arxiv.org/abs/2510.25741)
(ByteDance Seed) introduces Ouro, a *looped* transformer: rather than 192 distinct layers it
has **48 physical layers applied `R` times** (R=4, i.e. 192 logical layers), reusing the same
weights each loop. Latent reasoning happens *inside* the loop rather than in emitted chain-of-thought
tokens. The model used here is [`ByteDance/Ouro-2.6B-Thinking`](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking).

The loop is also what makes the next idea possible: at any decode step the model has an
**intermediate** answer (after loop 1) and a **final** answer (after loop R), computed from the
same weights.

**The technique — LoopCD.** [*Decoding Looped Transformers Better for (Almost) Free*](https://arxiv.org/abs/2610.02185)
(Apple) exploits that pair with contrastive decoding. Instead of sampling from the final-loop
logits `z_R`, it pushes the distribution **away from the shallow answer** and **toward the deep one**:

```
z' = z_R + ω · (z_R − z_1)        # LoopCD-Logits
```

where `z_k = lm_head(h_k)` and `ω` is the guidance strength (fixed, or adaptive/annealed up to a
cap). Ouro has no coda, so the hidden-state form applies `output_norm` *after* the contrast:

```
z' = lm_head(output_norm(g_R + ω · (g_R − g_1)))   # LoopCD-Hidden
```

The paper's hook is the **"almost free"** in the title: at a fixed loop count, LoopCD adds only a
second `lm_head` pass (logits form) or nothing at all (hidden form) — no extra transformer depth.
The larger, genuinely computational win the paper reports is the **reduced-depth regime**: run the
stack for *fewer* loops (e.g. halving R), where the answer would normally degrade, and let LoopCD
recover the accuracy — cutting FLOPs by ~22–48%.

This repo implements both forms (fixed and adaptive `ω`) and, in the M5 measurement below, checks
both halves of that story.

---

## What's implemented

| Milestone | What | State |
|---|---|---|
| **M0** | Scaffold, CMake + pinned ggml, Ouro GGUF converter (`arch="ouro"`), Q4_K_M quantize | done |
| **M1** | Loader + hand-written ggml forward pass (sandwich RMSNorms, RoPE θ=1e6, SwiGLU FFN, R×48 unrolled); `logits` command | done |
| **M2** | Vulkan backend, 192-slot **q8_0** KV cache (`ggml_flash_attn_ext`), autoregressive decode | done |
| **M3** | **LoopCD** — logits + hidden forms, fixed + adaptive ω, enabled by default | done |
| **M4** | OpenAI-compatible server (`/v1/chat/completions` streaming + non-streaming, `/v1/completions`, `/health`, `/v1/models`) | done |
| **M5** | Measurement harness (`tests/eval/`) + results (`tests/eval/RESULTS.md`) | done |

---

## Measured results (M5)

The eval is a **tiny pinned task set** — 4 math, 5 multiple-choice, 2 code-generation items —
greedy decoding, one sample per item. The shipping stack is `ouro.Q4_K_M.gguf` + `q8_0` KV on
Vulkan at ctx 4096.

**Headline:** at fixed depth, LoopCD's accuracy effect on this set is **within noise** (one item
swings 20 points). The reproducible result is the other half of the paper — **halving the recurrent
depth (R=4 → R=2) gives ~1.8× throughput at no measured accuracy cost.**

### R=4 (default) — Q4_K_M + q8_0 KV + Vulkan

| task | condition | accuracy | tok/s |
|---|---|---|---|
| math | off / logits / logits_adaptive / hidden | 100% | ~10.4–11.7 |
| mc | off | 80% | 11.8 |
| mc | logits / logits_adaptive / hidden | 100% | 8.6–11.2 |
| gen | all | 100% | ~10.2–11.8 |

### R=2 (halved depth)

| task | condition | accuracy | tok/s |
|---|---|---|---|
| math | all | 100% | ~18.8–21.6 |
| mc | off / logits_adaptive | 100% | ~19.6–21.2 |
| mc | logits / hidden | 80% | ~19.0–21.8 |
| gen | all | 100% | ~18.6–22.0 |

| task | R=4 tok/s | R=2 tok/s | accuracy change |
|---|---|---|---|
| math | 11.5 | 20.8 | none (100% → 100%) |
| mc | ~11.8 | ~21.2 | ±1 item, sign flips → noise |
| gen | 11.5 | 21.5 | none |

### Honest caveats

- **One item = 20–25 accuracy points.** The MCQ "LoopCD helps" / "LoopCD hurts" signals flip
  direction between R=4 and R=2; that is a one-item swing, not a finding.
- **`LoopCD-Hidden` is only meaningful on f32** weights + f32 KV. On the quantized shipping stack it
  is known to degenerate to a repeating token, so its shipping rows are reported for completeness,
  not as a reproduction. (The `f32_clean` config exists in `tests/eval/configs.json` but has not been
  run yet — it is slow, ~1 tok/s on CPU.)
- **These are not the paper's numbers.** The paper's AIME pass@1 uses 30 problems × 16 samples; the
  reference values in `RESULTS.md` are a direction/scale check, not a pass/fail bar.

Full per-task tables, deltas and the R=4-vs-R=2 comparison are generated into
[`tests/eval/RESULTS.md`](tests/eval/RESULTS.md) by `tests/eval/report.py`.

---

## Layout

```
src/            engine: model loader, graph, KV cache, loopcd, tokenizer, chat template,
                sampling, decode loop, OpenAI-compatible server
tools/          asset fetch + Ouro GGUF conversion + GGUF metadata check
tests/
  reference/    HuggingFace reference harness + numeric gate comparison
  unit/         focused unit tests (ops, kv, loopcd math, sampling, tokenizer)
  eval/         M5 measurement harness (run_eval.py, score.py, report.py, items/, RESULTS.md)
third_party/    vendored cpp-httplib + nlohmann/json
```

## Build

Requires CMake ≥ 3.20, a C++17 compiler, Ninja, and (for the shipping backend) the Vulkan SDK.
ggml is fetched and pinned by CMake.

```sh
# Vulkan (shipping)
cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk

# CPU only (no Vulkan SDK needed)
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLOOPCD_VULKAN=OFF
cmake --build build-cpu
```

Fetch model config/tokenizer, convert to GGUF, and quantize:

```sh
python tools/fetch_assets.py                 # config/tokenizer/custom code into models/ouro/
python tools/convert_ouro.py                 # safetensors -> ouro.f16.gguf (arch="ouro")
# quantize with the Ouro-aware llama.cpp-ouro build:
llama-quantize.exe models/ouro/ouro.f16.gguf models/ouro/ouro.Q4_K_M.gguf Q4_K_M
```

## Run

```sh
# one-shot generation
build-vk/loopcd run -m models/ouro/ouro.Q4_K_M.gguf --device vulkan --kv-type q8_0 \
    -p "Explain why the sky is blue." --loopcd logits --loopcd-omega 0.3

# server
build-vk/loopcd serve -m models/ouro/ouro.Q4_K_M.gguf --device vulkan --kv-type q8_0 \
    --ctx 4096 --loopcd logits --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"Say hi in five words."}],
  "enable_thinking": false, "max_tokens": 64, "temperature": 0,
  "extra_body": {"loopcd": {"mode": "logits", "omega": 0.3}}
}'
```

LoopCD modes: `off`, `logits`, `hidden`; `--loopcd-adaptive` anneals ω up to `--loopcd-omega-max`;
`--loops N` sets the recurrent depth (default 4, the model's trained value).

## Verification

Correctness is a numeric gate against a pinned HuggingFace `transformers` f32 forward pass,
not eyeballing output:

- **Graph fidelity** — f32 weights + f32 KV, R=4 and R=2, all four tensors (`z_R`, `z_1`, `h_1`, `h_R`)
  compared tightly. f32 (not f16) because ggml re-quantizes activations inside f16/bf16 matmuls and
  the 192-layer loop amplifies that past the tolerance.
- **Serving stack** — Q4_K_M + q8_0 KV + Vulkan vs the same f32 reference; enforced criterion is
  **`z_R` argmax at every position**, with logit magnitudes recorded against a ceiling.
- **Decode loop** — greedy continuation matches HF token-for-token.
- Unit tests: q8_0 KV round-trip vs ggml, LoopCD combine math, RoPE/RMSNorm/sandwich vs numpy,
  tokenizer and chat-template parity with HF.

## Limitations

- Ouro only; no other architectures.
- Single request at a time (no batching) — it's daily-driver-single-stream, not a server fleet.
- Windows + Vulkan/CPU; no CUDA/Metal.
- The adaptive-depth / `early_exit_gate` path is out of scope.
- Prefill is dense and runs the stack R times, so long prompts are slow; travel light on `--ctx`.

## References

- Ouro: *Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed — arXiv:2510.25741
- LoopCD: *Decoding Looped Transformers Better for (Almost) Free*, Apple — arXiv:2610.02185
- llama.cpp PR #29823 (Ouro architecture + converter) and [ggml](https://github.com/ggml-org/ggml)
