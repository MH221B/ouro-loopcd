# loopcd + kitty

loopcd is an inference engine for one model, Ouro-2.6B-Thinking. It is C++ on ggml, with a Vulkan
backend on one AMD Radeon 890M iGPU, serving an OpenAI-compatible chat endpoint.

It is not a llama.cpp fork. It links ggml and takes three pieces from
[llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823): the Ouro graph, the GGUF
converter, and the BPE tokenizer.

I built it to test a paper's claim on my own hardware: that LoopCD improves a looped model at
almost no cost. I wanted to measure the cost.

| Item | Value |
|---|---|
| Model | Ouro-2.6B-Thinking, Q4_K_M GGUF |
| Layers | 48, applied 4 times (R=4) |
| Backend | ggml + Vulkan |
| Device | AMD Radeon 890M iGPU |
| Server | OpenAI-compatible, `/v1/chat/completions` |

## The model and the method

Ouro has 48 layers and applies them R times, reusing the same weights. The reasoning happens
inside the loop, not in output tokens. The model was trained at R=4.

![The Ouro loop. The 48 layers run R times, and LoopCD taps the answers at loop 1 and loop R.](images/loop.png)

LoopCD uses the two answers the model already holds: one after loop 1, one after loop R. It pushes
probability away from the shallow answer and toward the deep one.

```text
Logits form:  z' = z_R + omega * (z_R - z_1)
Hidden form:  z' = lm_head(output_norm(g_R + omega * (g_R - g_1)))
```

Here `z_k = lm_head(h_k)`. `omega` is the strength, fixed or annealed up to a maximum. Ouro has no
coda, so the hidden form applies `output_norm` after the contrast.

The method is from *Decoding Looped Transformers Better for (Almost) Free* (Apple). The paper's
compute claim is the reduced-depth regime: run fewer loops, accept some accuracy loss, and let
LoopCD recover it. It reports 22% to 48% fewer FLOPs.

LoopCD needs at least two loops. At R=1 there is nothing to contrast.

## The pipeline

The engine handles one request at a time.

```text
loopcd serve                      src/server.cpp      OpenAI-compatible HTTP
  ouro_backend_init / load        src/model.cpp       weights to Vulkan
  ouro_kv_cache_init              src/kv_cache.cpp    one slot per (loop, layer)
  ouro_generator::begin           src/generate.cpp    one sequence
    next()
      prefill: ouro_forward_kv    src/graph.cpp       R*48 layers, flash attention
      pack / evict                src/kitty_kv.cpp    only for the packed cache
      ouro_loopcd_guided_logits   src/loopcd.cpp      z' = z_R + omega*(z_R - z_1)
      sample                      src/sampling.cpp
      detokenize                  src/tokenizer.cpp
```

![The decode pipeline. Prefill runs the whole prompt. Each step returns the taps, LoopCD combines them, and the engine samples one token.](images/decode-flow.png)

The graph returns six taps at once: `zR`, `z1`, `h1`, `hR`, `g1`, `gR`. LoopCD is a short step
after the graph, not a second forward pass.

## What is built

| Milestone | Work | State |
|---|---|---|
| M0 | Scaffold. CMake with pinned ggml. Ouro GGUF converter. Q4_K_M quantize. | done |
| M1 | Loader and hand-written ggml forward pass. Sandwich RMSNorms, RoPE theta=1e6, SwiGLU FFN, R*48 unrolled. | done |
| M2 | Vulkan backend. 192-slot q8_0 KV cache with `ggml_flash_attn_ext`. Autoregressive decode. | done |
| M3 | LoopCD, logits and hidden forms, fixed and adaptive omega. On by default. | done |
| M4 | OpenAI-compatible server. Streaming and non-streaming chat, completions, health, models. | done |
| M5 | Measurement harness in `tests/eval/`. Results in `tests/eval/RESULTS.md`. | done |
| M6 | Kitty-KV 2-bit cache as an f16 fake-quant, with a HuggingFace oracle go/no-go. | done |
| M7a | Per-slot KV allocation. Lifts the ggml 2 GiB buffer and 4 GiB SSBO-range walls. | done |
| M7b/c | Packed 2-bit KV cache plus the `KITTY_QK` and `KITTY_SV` ops in a patched ggml. | done |

The design for the work M7 left unfinished is in [optimization.md](optimization.md).

## Measured results

The eval is small and pinned: 4 math items, 5 multiple-choice, 2 code generation, greedy decoding,
one sample each. The shipping stack is Q4_K_M with a q8_0 KV cache on Vulkan at ctx 4096.

The honest summary: on this set, LoopCD's accuracy effect is inside the noise floor. One item is
worth 20 accuracy points, so the multiple-choice columns flip sign between runs. They mean nothing.
The reproducible result is the other half of the paper. Halving the loop count runs about 1.8x
faster with no accuracy change I can measure.

| Task | R=4 | R=2 |
|---|---|---|
| math | 100%, 10.4 to 11.7 tok/s | 100%, 18.8 to 21.6 tok/s |
| mc | 80% off, else 100%, 8.6 to 11.8 tok/s | 80% logits and hidden, else 100%, 19.0 to 21.8 tok/s |
| gen | 100%, 10.2 to 11.8 tok/s | 100%, 18.6 to 22.0 tok/s |

Three notes, because the table flatters the result:

- One item is worth 20 to 25 points. The "LoopCD helps MCQ" reading at R=4 and the "LoopCD hurts
  MCQ" reading at R=2 are the same item moving in opposite directions. Neither is a finding.
- LoopCD-Hidden only means anything on f32 weights with an f32 KV cache. On the quantized stack it
  collapses into a repeating token, so its rows above are there for completeness.
- These are not the paper's numbers. The paper's AIME pass@1 uses 30 problems and 16 samples. The
  values in `RESULTS.md` are a direction check, not a bar to pass.

Full per-task tables and the R=4 against R=2 comparison are in
[`tests/eval/RESULTS.md`](tests/eval/RESULTS.md), generated by `tests/eval/report.py`.

## The packed 2-bit KV cache

M7 replaces the KV cache with a packed 2-bit one. Every channel is 2 bits. The most important key
channels in each page are promoted to 4 bits. The sink and the pending page stay in f16.

![One slot of the K cache, and the four parts of one packed page.](images/kitty-layout.png)

The layout follows Kitty ([arXiv 2511.18643](https://arxiv.org/abs/2511.18643)), not Kitty's
Triton implementation. The two ops mirror Kitty's structure: QK, softmax, SV. The kernels read the
packed data in per-channel tiles, and they never stage a full page.

![The attention path over the packed cache. The two ops read the packed planes and the f16 sink directly, so the engine never builds an f16 copy of the whole cache.](images/kitty-attention.png)

![Committed memory at ctx 4096](images/memory.png)
![Decode speed on long generations, ctx 4096](images/throughput.png)

Memory. The packed cache uses 2.03 GiB less, 39%. Absolute figures include the model and the
driver, so the difference is the number that matters.

Throughput is a regression, and it is reported rather than hidden. Packed decode is about 2x slower
on long generations. The cause is structural: decode uses one query, so the first-cut kernel gives
only 16 workgroups on a 16-CU iGPU. Split-K is the known fix, and it is not written.

Capacity. M6 capped the cache at ctx 2048 and I first blamed the driver. It was ggml: it rejects
any buffer over 2 GiB, and no tensor may exceed the 4 GiB SSBO range. One tensor per slot removes
both walls, so f16, q8_0, and the packed cache all run at ctx 8192 with no override. The packed
cache reaches furthest, because its planes are small tensors.

What is not claimed. Packed-against-f16 token agreement is 0.625, not the 0.98 I specified. That is
not a defect. The two modes are different quantizations: f16 stores `fp16(q*scale + min)`, and the
packed path dequantizes `q*fp16(scale) + fp16(min)` in f32, so their inputs differ by half an fp16
ulp per element. I recorded the number, left the threshold alone, and enforced the three checks in
Verification instead. The accuracy table is also a reduced set, one item per cell, so M7 has no
task-accuracy claim yet.

A correction to M6. `ouro_kv_cache_clear` was never called, so the kitty cursors survived across
requests and every request after the first skipped quantization. M6's `kitty_shipping` table was
measured with that bug, so it is an upper bound. Re-running it is outstanding. M6's oracle results
never used the server, so they stand, and the M5 tables above are q8_0.

## The optimization I didn't do

Packed decode runs about 2x slower than q8_0, and I never fixed it. It is the one piece of M7 I
left unfinished, and I would rather hand it over than pretend it is fine.

The short version: decode attention (`KITTY_SV`) launches 128 threads on a 16-CU GPU, and the host
packer does a blocking device round-trip per slot, per token. Either could be the real cost, and
nobody has profiled it. The fix is probably splitting the key axis on `KITTY_SV` and batching the
packer, run at R=2 where the numbers get interesting.

There is a roofline, so the expectations are set. The weights are re-read once per loop, which puts
60 tok/s out of reach at R>=2 no matter how good the kernels get. The target is 27 tok/s at R=2,
about 5x today's packed number.

The full design is in [optimization.md](optimization.md), including the gate and the rules for
deciding which fixes are worth doing. If you want to take it on, that file is the spec.

## Layout

```text
src/            engine: model loader, graph, KV cache (including the packed 2-bit path),
                kitty packer, loopcd, tokenizer, chat template, sampling, decode loop, server
tools/          asset fetch, Ouro GGUF conversion, GGUF metadata check,
                ggml-kitty-ops.patch (the M7 delta applied to ggml)
tests/
  reference/    HuggingFace reference harness, numeric gate comparison, the kitty pack
                fixture, and the packed-against-f16 gate script
  unit/         focused unit tests: ops, kv, loopcd math, sampling, tokenizer, cache
                allocation, and the kitty pack, ops, and equivalence checks
  eval/         measurement harness: run_eval.py, score.py, report.py, items/, RESULTS.md
third_party/    vendored cpp-httplib and nlohmann/json
images/         diagram sources (.mmd) and the rendered PNGs used in this README
```

## Build

You need CMake 3.20 or newer, a C++17 compiler, Ninja, git, and for the shipping backend the
Vulkan SDK. CMake fetches and pins ggml, then applies one patch to it.

```sh
# Vulkan, the shipping backend
cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk

# CPU only, no Vulkan SDK needed
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLOOPCD_VULKAN=OFF
cmake --build build-cpu
```

M7's packed KV cache needs two ops, `KITTY_QK` and `KITTY_SV`, which ggml does not have at the
pinned commit. Rather than maintain a fork, the delta lives in
[`tools/ggml-kitty-ops.patch`](tools/ggml-kitty-ops.patch), 723 lines across 12 files, and CMake
applies it to the fetched tree with `PATCH_COMMAND`. The upstream commit is pinned, so the patch is
deterministic. It is a clean, general addition, so it is a reasonable candidate to send upstream.
At that point the patch goes away.

For an offline or faster rebuild, pass `-DFETCHCONTENT_SOURCE_DIR_GGML=<dir>` to use a local ggml
checkout instead of fetching. That directory must be a clean checkout of the pinned commit, because
the patch is applied to whatever source directory you give it.

Then fetch the model config and tokenizer, convert to GGUF, and quantize:

```sh
python tools/fetch_assets.py                 # config, tokenizer, custom code into models/ouro/
python tools/convert_ouro.py                 # safetensors -> ouro.f16.gguf, arch="ouro"
llama-quantize.exe models/ouro/ouro.f16.gguf models/ouro/ouro.Q4_K_M.gguf Q4_K_M
```

The quantize step needs the Ouro-aware build, which is
[llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823). Upstream `llama-quantize`
rejects `arch="ouro"`.

## Run

The engine has six commands: `inspect`, `logits`, `run`, `tokenize`, `render`, and `serve`.

```sh
# one-shot generation. `run` takes token ids, not text, so tokenize first:
build-vk/loopcd tokenize -m models/ouro/ouro.Q4_K_M.gguf --text "Explain why the sky is blue."
# then pass those ids with --tokens a,b,c (or --tokens-file f), for example:
build-vk/loopcd run -m models/ouro/ouro.Q4_K_M.gguf --device vulkan --kv-type q8_0 \
    --tokens-file prompt.ids --max-tokens 256 --loopcd logits --loopcd-omega 0.3

# server: text in, text out. This is the usual entry point.
build-vk/loopcd serve -m models/ouro/ouro.Q4_K_M.gguf --device vulkan --kv-type q8_0 \
    --ctx 4096 --loopcd logits --port 8080
```

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/v1/models` | GET | Model list |
| `/v1/chat/completions` | POST | Chat, streaming and non-streaming |
| `/v1/completions` | POST | Text completion |

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"Say hi in five words."}],
  "enable_thinking": false, "max_tokens": 64, "temperature": 0,
  "extra_body": {"loopcd": {"mode": "logits", "omega": 0.3}}
}'
```

LoopCD modes are `off`, `logits`, and `hidden`. `--loopcd-adaptive` increases omega up to
`--loopcd-omega-max`. `--loops N` sets the recurrent depth. The default is 4, which is what the
model was trained at.

`--kv-type` is `q8_0` (the default), `f16`, `f32`, or `kitty`. `kitty` selects the packed 2-bit
cache from M7. `--kitty-storage f16` selects M6's fake-quant instead. `--kitty-boost N` sets how
many channels per page are promoted to 4 bits (32 is 25%, 25 is 20%). `--kitty-sink N` sets the
f16 sink length, default 32.

## Verification

Correctness is a numeric gate against a pinned HuggingFace `transformers` f32 forward pass, not a
look at the output text.

- Graph fidelity. f32 weights and f32 KV, at R=4 and R=2, comparing all four tensors `z_R`, `z_1`,
  `h_1`, and `h_R` tightly. The test uses f32 rather than f16 because ggml re-quantizes activations
  inside f16 and bf16 matmuls, and the 192-layer loop amplifies that error past the tolerance.
- Serving stack. Q4_K_M plus q8_0 KV on Vulkan, against the same f32 reference. The enforced
  criterion is the `z_R` argmax at every position, with the logit magnitudes recorded against a
  ceiling. A `z_R` argmax flip is a finding to localize, not something to wave away.
- Decode loop. A greedy continuation must match HuggingFace token for token.
- Unit tests. These cover the q8_0 KV round-trip against ggml, the LoopCD combine math, RoPE,
  RMSNorm, and the sandwich block against numpy, plus tokenizer and chat-template parity.
- Packed KV, against an independent reference rather than itself. The packed attention reproduces a
  from-scratch dequantize-and-attend reference to 7.6e-10 on CPU and 5.5e-10 on Vulkan. The C++
  emitter reproduces a Python oracle's packed bytes exactly. A cache-level check confirms that the
  packed page's dequantized values differ from what the f16 mode stores by no more than fp16
  rounding, 2.44e-3 against a 9.96e-3 bound. That last check fails when I break it on purpose,
  because a test that cannot fail is not evidence.

## Limitations

- Ouro only. No other architectures, by design.
- One request at a time. There is no batching, so one slow generation blocks the server.
- Windows with Vulkan or CPU. No CUDA, no Metal.
- The adaptive-depth and `early_exit_gate` path is out of scope.
- The packed 2-bit cache is about 2x slower than q8_0 on long generations. It buys memory and
  context, not speed.
- Packed-KV accuracy is smoke-tested only, one item per cell. It has a memory win and six correct
  items, not a task-accuracy measurement.
- Prefill runs the stack R times. For `kitty` it is chunked at the page size, so long prompts cost
  proportionally. The other KV types use one dense pass.

## Glossary

| Term | Meaning |
|---|---|
| loop | One pass of the 48 layers. |
| R | The number of loops. The model was trained at R=4. |
| slot | One KV cache tensor. There is one slot per (loop, layer) pair, so 48*R slots. |
| page | A group of 128 tokens. It is the unit for key quantization. |
| sink | The first tokens of the sequence, kept in f16. |
| boost | The key channels in a page that are promoted from 2 bits to 4 bits. |
| packed cache | The 2-bit KV storage from M7. |
| tap | One of the six outputs the graph returns for LoopCD: `zR`, `z1`, `h1`, `hR`, `g1`, `gR`. |
| prefill | The first forward pass, over the whole prompt. |
| decode | One token at a time, after prefill. |

## References

- *Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741)
- *Decoding Looped Transformers Better for (Almost) Free*, Apple, [arXiv 2610.02185](https://arxiv.org/abs/2610.02185)
- *Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost*, [arXiv 2511.18643](https://arxiv.org/abs/2511.18643), the layout the packed 2-bit cache follows
- [llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823), the Ouro architecture and converter
- [ggml](https://github.com/ggml-org/ggml)
