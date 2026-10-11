# loopcd + kitty

loopcd is an inference engine for one model, Ouro-2.6B-Thinking. It is written in C++ and runs on
ggml. The shipping backend is Vulkan, on one AMD Radeon 890M iGPU. The engine serves an
OpenAI-compatible chat endpoint.

It is not a fork of llama.cpp. It links ggml and copies three pieces from
[llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823): the Ouro graph, the Ouro
GGUF converter, and the BPE tokenizer. The engine handles one architecture and one model.

I built it to test a paper's claim on my own hardware. The paper says a method called LoopCD
improves a looped model at almost no cost. I wanted to measure the real cost.

| Item | Value |
|---|---|
| Model | Ouro-2.6B-Thinking, Q4_K_M GGUF |
| Layers | 48, applied 4 times (R=4) |
| Backend | ggml + Vulkan |
| Device | AMD Radeon 890M iGPU |
| Server | OpenAI-compatible, `/v1/chat/completions` |
| Platform | Windows, Vulkan or CPU |

## The model and the method

Ouro is a looped transformer. It has 48 layers and applies them R times. Each pass reuses the same
weights. The model does its reasoning inside the loop. It does not write that reasoning into the
output tokens. The model was trained at R=4.

![The Ouro loop. The 48 layers run R times, and LoopCD taps the answers at loop 1 and loop R.](images/loop.png)

LoopCD is the method this project tests. At each decode step the model already holds two answers:
one after loop 1, and one after loop R. LoopCD samples from neither. It pushes probability away
from the loop-1 answer and toward the loop-R answer.

```text
Logits form:  z' = z_R + omega * (z_R - z_1)
Hidden form:  z' = lm_head(output_norm(g_R + omega * (g_R - g_1)))
```

Here `z_k = lm_head(h_k)`, and `omega` is the strength. `omega` can be fixed, or it can increase
up to a maximum. Ouro has no coda, so the hidden form applies `output_norm` after the contrast.

The method comes from *Decoding Looped Transformers Better for (Almost) Free* (Apple). The paper's
main compute claim is the reduced-depth regime. Run fewer loops, accept some accuracy loss, and
let LoopCD recover it. The paper reports 22% to 48% fewer FLOPs.

LoopCD needs at least two loops. At R=1 there is nothing to contrast.

## The pipeline

The engine handles one request at a time. The server calls a generator. The generator calls the
graph. The graph runs the layers.

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

The graph returns six taps at once: `zR`, `z1`, `h1`, `hR`, `g1`, `gR`. LoopCD is a short step
after the graph. It is not a second forward pass.

## What is built

| Milestone | Work | State |
|---|---|---|
| M0 | Scaffold. CMake with pinned ggml. Ouro GGUF converter for `arch="ouro"`. Q4_K_M quantize. | done |
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

The eval is a small pinned set: 4 math items, 5 multiple-choice items, and 2 code-generation
items. Decoding is greedy, and each item runs once. The shipping stack is `ouro.Q4_K_M.gguf`
with a q8_0 KV cache on Vulkan, at context 4096.

The honest summary: on this set, LoopCD's accuracy effect is inside the noise floor. One item is
worth 20 accuracy points, so the multiple-choice columns change sign between runs. They do not
mean anything. The reproducible result is the other half of the paper. Halving the loop count
from R=4 to R=2 runs about 1.8x faster with no accuracy change I can measure.

At R=4, Q4_K_M plus q8_0 KV on Vulkan:

| Task | Condition | Accuracy | tok/s |
|---|---|---|---|
| math | off, logits, logits_adaptive, hidden | 100% | 10.4 to 11.7 |
| mc | off | 80% | 11.8 |
| mc | logits, logits_adaptive, hidden | 100% | 8.6 to 11.2 |
| gen | all | 100% | 10.2 to 11.8 |

At R=2:

| Task | Condition | Accuracy | tok/s |
|---|---|---|---|
| math | all | 100% | 18.8 to 21.6 |
| mc | off, logits_adaptive | 100% | 19.6 to 21.2 |
| mc | logits, hidden | 80% | 19.0 to 21.8 |
| gen | all | 100% | 18.6 to 22.0 |

The two configurations side by side:

| Task | R=4 tok/s | R=2 tok/s | Accuracy change |
|---|---|---|---|
| math | 11.5 | 20.8 | none, 100% both |
| mc | ~11.8 | ~21.2 | one item, sign flips, noise |
| gen | 11.5 | 21.5 | none |

Four notes, because the tables flatter the result:

- One item is worth 20 to 25 accuracy points. The "LoopCD helps MCQ" reading at R=4 and the
  "LoopCD hurts MCQ" reading at R=2 are the same single item moving in opposite directions.
  Neither reading is a finding.
- LoopCD-Hidden is only meaningful on f32 weights with an f32 KV cache. On the quantized stack it
  collapses into a repeating token, so its shipping rows are there for completeness only. The
  `f32_clean` config exists in `tests/eval/configs.json` but has not been run. On CPU it runs at
  about 1 token per second.
- These are not the paper's numbers. The paper's AIME pass@1 uses 30 problems and 16 samples. The
  reference values in `RESULTS.md` are a direction check, not a bar to pass.
- M6's `kitty_shipping` numbers were measured with a bug. See the note under the packed cache
  below.

The full per-task tables, the deltas, and the R=4 against R=2 comparison are generated into
[`tests/eval/RESULTS.md`](tests/eval/RESULTS.md) by `tests/eval/report.py`.

## The packed 2-bit KV cache

The KV cache is the largest memory term at long context. M7 replaces it with a packed 2-bit
cache. Every channel is stored at 2 bits. The most important key channels in each page are
promoted to 4 bits. The sink and the pending page stay in f16.

The layout follows Kitty ([arXiv 2511.18643](https://arxiv.org/abs/2511.18643)), but not Kitty's
Triton implementation. The two attention ops mirror Kitty's structure: QK, then softmax, then SV.
The kernels read the packed data in per-channel tiles. They do not stage a full page.

Memory, measured as committed memory of the server at context 4096:

| Cache | Committed memory |
|---|---|
| q8_0 | 5.19 GiB |
| packed 2-bit | 3.16 GiB |

The packed cache uses 2.03 GiB less, which is 39%. The absolute numbers include the model and the
driver, so the difference is the number that matters. The arithmetic predicts about 2.53 bits per
element against q8_0's 8.5, and the measured increments agree.

Capacity. M6 capped the cache at context 2048, and I first assumed the driver caused that cap. It
did not. ggml rejects any buffer larger than 2 GiB (`ggml-vulkan-buffers.cpp`), using a
`maxBufferSize` that the AMD driver advertises but does not enforce. Also, no tensor may exceed
the 4 GiB SSBO range. When the cache allocates one tensor per slot, both walls disappear. Now f16,
q8_0, and the packed cache all run at context 8192 with no override. The packed cache reaches
furthest, because its planes are small tensors.

Throughput is a regression, and the README reports it rather than hiding it:

| Cache | Long generations | Short items |
|---|---|---|
| q8_0 | 11.5 tok/s | 11.8 tok/s |
| packed 2-bit | 5.6 tok/s | 9.2 tok/s |

Packed decode is about 2x slower on long generations. The cause is structural. Decode uses one
query, so the first-cut kernel gives only 16 workgroups on a 16-CU iGPU. Split-K is the known fix,
and it is not written. The packed cache buys memory and context, not speed.

How it is verified. Each check compares against an independent reference, not against itself.

- The packed attention (`KITTY_QK`, then `KITTY_SV`) reproduces a from-scratch
  dequantize-and-attend reference to 7.6e-10 on CPU and 5.5e-10 on Vulkan.
- The C++ emitter reproduces a Python oracle's packed bytes exactly.
- A cache-level check confirms that the packed page's dequantized values differ from what the f16
  mode stores by no more than fp16 rounding: 2.44e-3 measured against a 9.96e-3 bound. The check
  fails when I break it on purpose.

What is not claimed. End-to-end packed-against-f16 token agreement is 0.625, not the 0.98 I first
specified. That is not a storage defect or a kernel defect. The two modes are different
quantizations. The f16 mode stores `fp16(q*scale + min)`. The packed path dequantizes
`q*fp16(scale) + fp16(min)` in f32. Their attention inputs differ by about half an fp16 ulp per
element, which is smaller than the quantization error itself. I recorded the number, left the
threshold untouched, and moved enforcement to the three checks above. The accuracy table is also
a reduced set, one item per cell, so M7 has no task-accuracy measurement yet. It has six correct
items and a memory win.

A correction to M6. A crash in the packed path turned up the fact that `ouro_kv_cache_clear` was
never called. The kitty cursors survived across requests, so every request after the first
skipped quantization entirely. M6's `kitty_shipping` table was measured with that bug, so its
numbers are an upper bound, not Kitty-on-Ouro. Re-running it is outstanding. M6's oracle results,
which were the go/no-go, never used the server, so they are unaffected. The M5 tables above use
q8_0, so they are unaffected too.

## The optimization I didn't do

Packed decode runs about 2x slower than q8_0, and I never fixed it. It is the one piece of M7 I
left unfinished, and I would rather hand it over than pretend it is fine.

The short version: decode attention (`KITTY_SV`) launches 128 threads on a 16-CU GPU, and the
host packer does a blocking device round-trip per slot, per token. Either could be the real cost,
and nobody has profiled it. The fix is probably splitting the key axis on `KITTY_SV` and batching
the packer, run at R=2 where the numbers get interesting.

There is a roofline, so the expectations are set. The weights are re-read once per loop, which
puts 60 tok/s out of reach at R>=2 no matter how good the kernels get. The target is 27 tok/s at
R=2, about 5x today's packed number.

The full design is in [optimization.md](optimization.md), including the gate and the rules for
deciding which fixes are worth doing. If you want to take it on, that file is the spec. I would
be glad to see it finished.

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

[optimization.md](optimization.md) at the repo root is the design for the unfinished packed-KV
throughput work.

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

One patch, and why. M7's packed KV cache needs two ops, `KITTY_QK` and `KITTY_SV`, which ggml
does not have at the pinned commit. Rather than maintain a fork of ggml, the delta lives in
[`tools/ggml-kitty-ops.patch`](tools/ggml-kitty-ops.patch), 723 lines across 12 files, and CMake
applies it to the fetched tree with `PATCH_COMMAND`. The upstream commit is pinned, so the patch
is deterministic. The addition is a clean, general one, a packed-KV attention pair, so it is a
reasonable candidate to send upstream. At that point the patch goes away.

For an offline or faster rebuild, pass `-DFETCHCONTENT_SOURCE_DIR_GGML=<dir>` to use a local ggml
checkout instead of fetching. That directory must be a clean checkout of the pinned commit,
because the patch is applied to whatever source directory you give it. An already-patched tree
fails the patch step.

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

The server answers these routes:

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
f16 sink length. The default is 32.

## Verification

Correctness is a numeric gate against a pinned HuggingFace `transformers` f32 forward pass. It is
not a look at the output text.

- Graph fidelity. f32 weights and f32 KV, at R=4 and R=2, comparing all four tensors `z_R`, `z_1`,
  `h_1`, and `h_R` tightly. The test uses f32 rather than f16 because ggml re-quantizes
  activations inside f16 and bf16 matmuls, and the 192-layer loop amplifies that error past the
  tolerance.
- Serving stack. Q4_K_M plus q8_0 KV on Vulkan, against the same f32 reference. The enforced
  criterion is the `z_R` argmax at every position, with the logit magnitudes recorded against a
  ceiling. A `z_R` argmax flip is a finding to localize, not something to wave away.
- Decode loop. A greedy continuation must match HuggingFace token for token.
- Unit tests. These cover the q8_0 KV round-trip against ggml, the LoopCD combine math, RoPE,
  RMSNorm, and the sandwich block against numpy, plus tokenizer and chat-template parity.
- Packed KV. Checked against an independent reference rather than itself. The packed attention
  reproduces a from-scratch dequantize-and-attend reference to 7.6e-10 on CPU and 5.5e-10 on
  Vulkan. The C++ emitter reproduces a Python oracle's packed bytes exactly. A cache-level check
  confirms that the packed page's dequantized values differ from what the f16 mode stores by no
  more than fp16 rounding. That last check fails when I break it on purpose, because a test that
  cannot fail is not evidence.

## Limitations

- Ouro only. No other architectures, by design.
- One request at a time. There is no batching, so one slow generation blocks the server.
- Windows with Vulkan or CPU. No CUDA, no Metal.
- The adaptive-depth and `early_exit_gate` path is out of scope.
- The packed 2-bit cache is about 2x slower than q8_0 on long generations. It buys memory and
  context, not speed. The first-cut kernels have no split-K.
- Packed-KV accuracy is smoke-tested only, one item per cell. M7 has no task-accuracy claim yet.
  It has a memory win and six correct items.
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
