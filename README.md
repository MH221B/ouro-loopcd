# loopcd

A from-scratch inference engine for Ouro-2.6B-Thinking with LoopCD contrastive decoding.
C++ on top of [ggml](https://github.com/ggml-org/ggml), running on one machine's
AMD Radeon 890M iGPU through Vulkan, serving an OpenAI-compatible chat endpoint.

It is not a llama.cpp fork. It links ggml and copies a few specific pieces from
[llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823), the PR that added Ouro support: the Ouro graph, the Ouro GGUF
converter, and the BPE tokenizer. The engine handles one architecture, Ouro, and one model.

I built it to check a paper's claim on my own hardware instead of taking it on faith,
and to measure what the technique costs to actually run.

## Background

Ouro is a looped transformer. Instead of 192 distinct layers it has 48 layers applied
4 times, reusing the same weights each pass, so the reasoning happens inside the loop
rather than in emitted chain-of-thought tokens. It comes from
*Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741),
and the model here is [`ByteDance/Ouro-2.6B-Thinking`](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking).

The loop is also what makes the next part possible. At any decode step the model has
already produced an intermediate answer after loop 1 and a final answer after loop 4,
both from the same weights.

LoopCD uses that pair for contrastive decoding. It does not sample from the final logits
`z_R`. It pushes probability away from the shallow answer and toward the deep one:

```
z' = z_R + ω · (z_R − z_1)        # LoopCD-Logits
```

Here `z_k = lm_head(h_k)` and `ω` is the guidance strength, either fixed or annealed up
to a cap. Ouro has no coda, so the hidden form applies `output_norm` after the contrast:

```
z' = lm_head(output_norm(g_R + ω · (g_R − g_1)))   # LoopCD-Hidden
```

The technique is from *Decoding Looped Transformers Better for (Almost) Free*, Apple,
[arXiv 2610.02185](https://arxiv.org/abs/2610.02185). The "almost free" claim is about a fixed loop count, where LoopCD adds
a second `lm_head` pass in logits mode and nothing at all in hidden mode, no extra depth.
The paper's larger computational claim is the reduced-depth regime: run fewer loops, take
some accuracy loss, and let LoopCD recover it, cutting FLOPs by roughly 22 to 48 percent.

Both forms are implemented here, fixed and adaptive ω, and the M5 measurement below tests
both halves of the claim.

## What's implemented

| Milestone | What | State |
|---|---|---|
| M0 | Scaffold, CMake with pinned ggml, Ouro GGUF converter for `arch="ouro"`, Q4_K_M quantize | done |
| M1 | Loader and hand-written ggml forward pass: sandwich RMSNorms, RoPE θ=1e6, SwiGLU FFN, R×48 unrolled | done |
| M2 | Vulkan backend, 192-slot q8_0 KV cache via `ggml_flash_attn_ext`, autoregressive decode | done |
| M3 | LoopCD, logits and hidden forms, fixed and adaptive ω, on by default | done |
| M4 | OpenAI-compatible server: `/v1/chat/completions` streaming and non-streaming, `/v1/completions`, `/health`, `/v1/models` | done |
| M5 | Measurement harness in `tests/eval/`, results in `tests/eval/RESULTS.md` | done |
| M6 | Kitty-KV 2-bit cache as an f16 fake-quant, with a HuggingFace oracle go/no-go | done |
| M7a | Per-slot KV allocation: lifts the ggml 2 GiB buffer and 4 GiB SSBO-range walls | done |
| M7b/c | Packed 2-bit KV cache (Kitty's layout) plus `KITTY_QK`/`KITTY_SV` ops in a ggml fork | done |

## Measured results

The eval is a tiny pinned set: 4 math items, 5 multiple-choice, 2 code-generation, greedy
decoding, one sample each. The shipping stack is `ouro.Q4_K_M.gguf` plus q8_0 KV on Vulkan
at context 4096.

The honest summary is that LoopCD's accuracy effect on this set is inside the noise floor.
One item swings 20 points, so the MCQ columns below flip sign between runs and mean nothing.
The reproducible result is the other half of the paper. Halving the recurrent depth from
R=4 to R=2 runs about 1.8x faster with no accuracy I can measure.

At R=4, Q4_K_M plus q8_0 KV on Vulkan:

| task | condition | accuracy | tok/s |
|---|---|---|---|
| math | off / logits / logits_adaptive / hidden | 100% | 10.4 to 11.7 |
| mc | off | 80% | 11.8 |
| mc | logits / logits_adaptive / hidden | 100% | 8.6 to 11.2 |
| gen | all | 100% | 10.2 to 11.8 |

At R=2:

| task | condition | accuracy | tok/s |
|---|---|---|---|
| math | all | 100% | 18.8 to 21.6 |
| mc | off / logits_adaptive | 100% | 19.6 to 21.2 |
| mc | logits / hidden | 80% | 19.0 to 21.8 |
| gen | all | 100% | 18.6 to 22.0 |

| task | R=4 tok/s | R=2 tok/s | accuracy change |
|---|---|---|---|
| math | 11.5 | 20.8 | none, 100% both |
| mc | ~11.8 | ~21.2 | one item, sign flips, noise |
| gen | 11.5 | 21.5 | none |

A few things worth stating plainly, because the tables flatter the result:

- One item is 20 to 25 accuracy points. The "LoopCD helps MCQ" reading at R=4 and the
  "LoopCD hurts MCQ" reading at R=2 are the same single item moving in opposite directions.
  Neither is a finding.
- LoopCD-Hidden only means anything on f32 weights with f32 KV. On the quantized stack it
  collapses into a repeating token, so its shipping rows are there for completeness. The
  `f32_clean` config exists in `tests/eval/configs.json` but has not been run; on CPU it
  is about 1 token/second.
- These are not the paper's numbers. The paper's AIME pass@1 uses 30 problems and 16
  samples. The reference values in `RESULTS.md` are a direction check, not a bar to pass.

The full per-task tables, deltas, and the R=4 versus R=2 comparison are generated into
[`tests/eval/RESULTS.md`](tests/eval/RESULTS.md) by `tests/eval/report.py`.

## Packed 2-bit KV (M7)

The KV cache is the dominant memory term at long context, so M7 replaces it with a packed
2-bit cache: every channel at 2 bits, the most important *key* channels per page promoted to
4 bits, and the sink plus the pending page kept in f16. It follows Kitty's layout
([arXiv 2511.18643](https://arxiv.org/abs/2511.18643)) but not Kitty's Triton implementation.
The two attention ops mirror its structure (QK, then SV, with softmax in between) and the
kernels stream per-channel tiles instead of staging a full page.

**Memory.** Measured committed memory of the server at context 4096: **3.16 GiB packed against
5.19 GiB for q8_0**, which is 2.03 GiB less, 39%. Absolute figures include the model and the driver, so
the delta is the meaningful number. The arithmetic behind it, about 2.53 bits per element
against q8_0's 8.5, is corroborated by the measured increments.

**Capacity.** M6 capped the cache at context 2048 and I had assumed that was a driver limit. It
was not. ggml rejects any buffer over 2 GiB (`ggml-vulkan-buffers.cpp`), using a `maxBufferSize`
the AMD driver advertises but does not enforce, and separately no tensor may exceed the 4 GiB
SSBO range. Allocating one tensor per slot removes both walls, so **f16, q8_0 and the packed
cache all run at context 8192 now**, with no override. The packed cache reaches furthest because
its planes are small tensors.

**Throughput is a regression, and it is reported rather than hidden.** Packed decode runs at
about 5.6 tok/s on long generations at context 4096 against q8_0's 11.5. The cause is
structural: decode is a single query, so the first-cut kernel gives only 16 workgroups on a
16-CU iGPU. Split-K is the known fix and is not written. The packed cache buys memory and
context, not speed.

**How it is verified.** Against an independent reference, not against itself. The packed
attention (`KITTY_QK` then `KITTY_SV`) reproduces a from-scratch dequantize-and-attend reference
to 7.6e-10 on CPU and 5.5e-10 on Vulkan. The C++ emitter reproduces a Python oracle's packed
bytes exactly. A cache-level check confirms the packed page's dequantized values differ from
what the f16 mode stores by no more than fp16 rounding (2.44e-3 measured against a 9.96e-3
bound), and that check is shown to fail when deliberately broken.

**What is not claimed.** End-to-end packed-versus-f16 token agreement is **0.625**, not the 0.98
I originally specified. That is not a storage or kernel defect: the two modes are different
quantizations. The f16 mode stores `fp16(q*scale + min)`; the packed path dequantizes
`q*fp16(scale) + fp16(min)` in f32. Their attention inputs therefore differ by about half an
fp16 ulp per element, which is smaller than the quantization error itself. I recorded the
number, left the threshold untouched, and moved enforcement to the checks above. The accuracy
table is also a **reduced** condition set, one item per cell, so M7 has no task-accuracy
measurement yet: it has six correct items and a memory win.

**A correction to M6.** Chasing a server crash in the packed path turned up the fact that
`ouro_kv_cache_clear` was never called, so the kitty cursors survived across requests and
**every request after the first skipped quantization entirely**. M6's `kitty_shipping` table was
measured with that bug, so its numbers are an upper bound rather than Kitty-on-Ouro; re-running
it is outstanding. M6's oracle results, which were the go/no-go, never used the server and are
unaffected. The M5 tables above are q8_0 and are unaffected too.

## The optimization I didn't do

Packed decode runs about 2x slower than q8_0, and I never fixed it. It is the one piece of M7 I
left unfinished, and I would rather hand it over than pretend it is fine.

The short version: decode attention (`KITTY_SV`) launches 128 threads on a 16-CU GPU, and the
host packer does a blocking device round-trip per slot, per token. Either could be the real
cost, and nobody has profiled it. The fix is probably splitting the key axis on `KITTY_SV` and
batching the packer, run at R=2 where the numbers get interesting.

There is a roofline, so the expectations are set. The weights are re-read once per loop, which
puts 60 tok/s out of reach at R>=2 no matter how good the kernels get. The target is 27 tok/s at
R=2, about 5x today's packed number.

The full design is in [optimization.md](optimization.md), including the gate and the rules for
deciding which fixes are worth doing. If you want to take it on, that file is the spec. I would
be glad to see it finished.

## Layout

```
src/            engine: model loader, graph, KV cache (incl. the packed 2-bit path),
                kitty packer, loopcd, tokenizer, chat template, sampling, decode loop, server
tools/          asset fetch, Ouro GGUF conversion, GGUF metadata check,
                ggml-kitty-ops.patch (the M7 delta applied to ggml)
tests/
  reference/    HuggingFace reference harness, numeric gate comparison, the kitty pack
                fixture, and the packed-vs-f16 gate script
  unit/         focused unit tests: ops, kv, loopcd math, sampling, tokenizer, cache
                allocation, and the kitty pack / ops / equivalence checks
  eval/         measurement harness: run_eval.py, score.py, report.py, items/, RESULTS.md
third_party/    vendored cpp-httplib and nlohmann/json
```

[optimization.md](optimization.md) at the repo root is the design for the unfinished packed-KV
throughput work.

## Build

Needs CMake 3.20 or newer, a C++17 compiler, Ninja, git, and for the shipping backend the
Vulkan SDK. CMake fetches and pins ggml, then applies one patch to it.

```sh
# Vulkan, the shipping backend
cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk

# CPU only, no Vulkan SDK needed
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLOOPCD_VULKAN=OFF
cmake --build build-cpu
```

**One patch, and why.** M7's packed KV cache needs two ops, `KITTY_QK` and `KITTY_SV`, which
ggml does not have at the pinned commit. Rather than maintain a fork of ggml, the delta lives in
[`tools/ggml-kitty-ops.patch`](tools/ggml-kitty-ops.patch), 723 lines across 12 files, and
CMake applies it to the fetched tree with `PATCH_COMMAND`. Because the upstream commit is pinned,
the patch is deterministic. The addition is a clean, general one (a packed-KV attention pair),
so it is a reasonable candidate to send upstream, at which point the patch goes away.

Offline or faster rebuilds: `-DFETCHCONTENT_SOURCE_DIR_GGML=<dir>` uses a local ggml checkout
instead of fetching. That directory must be a **clean** checkout of the pinned commit, because the
patch is applied to whatever source directory is used, so an already-patched tree will fail the
patch step.

Fetch the model config and tokenizer, convert to GGUF, and quantize:

```sh
python tools/fetch_assets.py                 # config, tokenizer, custom code into models/ouro/
python tools/convert_ouro.py                 # safetensors -> ouro.f16.gguf, arch="ouro"
llama-quantize.exe models/ouro/ouro.f16.gguf models/ouro/ouro.Q4_K_M.gguf Q4_K_M
```

The quantize step needs the Ouro-aware build, which is [llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823). Upstream
`llama-quantize` rejects `arch="ouro"`.

## Run

```sh
# one-shot generation. `run` takes token ids, not text, so tokenize first:
build-vk/loopcd tokenize -m models/ouro/ouro.Q4_K_M.gguf --text "Explain why the sky is blue."
# then pass those ids with --tokens a,b,c (or --tokens-file f), e.g.
build-vk/loopcd run -m models/ouro/ouro.Q4_K_M.gguf --device vulkan --kv-type q8_0 \
    --tokens-file prompt.ids --max-tokens 256 --loopcd logits --loopcd-omega 0.3

# server: text in, text out. This is the usual entry point.
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

LoopCD modes are `off`, `logits`, and `hidden`. `--loopcd-adaptive` anneals ω up to
`--loopcd-omega-max`. `--loops N` sets the recurrent depth, default 4, which is what the
model was trained at.

`--kv-type` is `q8_0` (default), `f16`, `f32`, or `kitty`. `kitty` selects the packed 2-bit
cache from M7; `--kitty-storage f16` selects M6's fake-quant instead. `--kitty-boost N` sets how
many channels per page are promoted to 4 bits (`32` is 25%, `25` is 20%); `--kitty-sink N` sets
the f16 sink length (default 32).

## Verification

Correctness is a numeric gate against a pinned HuggingFace `transformers` f32 forward pass,
not a glance at the output.

- Graph fidelity. f32 weights and f32 KV, at R=4 and R=2, comparing all four tensors
  `z_R`, `z_1`, `h_1`, `h_R` tightly. f32 rather than f16 because ggml re-quantizes
  activations inside f16 and bf16 matmuls, and the 192-layer loop amplifies that error
  past the tolerance.
- Serving stack. Q4_K_M plus q8_0 KV on Vulkan against the same f32 reference. The enforced
  criterion is `z_R` argmax at every position, with the logit magnitudes recorded against a
  ceiling. A `z_R` argmax flip is treated as a finding to localize, not something to wave away.
- Decode loop. A greedy continuation has to match HuggingFace token for token.
- Unit tests cover the q8_0 KV round-trip against ggml, the LoopCD combine math, RoPE,
  RMSNorm, and the sandwich block against numpy, plus tokenizer and chat-template parity.
- Packed KV. Checked against an independent reference rather than itself: the packed attention
  reproduces a from-scratch dequantize-and-attend reference to 7.6e-10 on CPU and 5.5e-10 on
  Vulkan; the C++ emitter reproduces a Python oracle's packed bytes exactly; and a cache-level
  check confirms the packed page's dequantized values differ from what the f16 mode stores by no
  more than fp16 rounding. That last check is shown to fail when deliberately broken, because a
  test that cannot fail is not evidence.

## Limitations

- Ouro only. No other architectures, by design.
- One request at a time. There is no batching, so one slow generation blocks the server.
- Windows with Vulkan or CPU. No CUDA, no Metal.
- The adaptive-depth and `early_exit_gate` path is out of scope.
- The packed 2-bit cache is about 2x slower than q8_0 on long generations. It buys memory and
  context, not speed: the first-cut kernels have no split-K.
- Packed-KV accuracy is smoke-tested only, one item per cell, so M7 has no task-accuracy claim
  yet. It has a memory win and six correct items.
- Prefill runs the stack R times. For `kitty` it is chunked at the page size, so long prompts
  cost proportionally; the other KV types use one dense pass.

## References

- *Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741)
- *Decoding Looped Transformers Better for (Almost) Free*, Apple, [arXiv 2610.02185](https://arxiv.org/abs/2610.02185)
- *Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost*, [arXiv 2511.18643](https://arxiv.org/abs/2511.18643), the layout the packed 2-bit cache follows
- [llama.cpp PR #29823](https://github.com/ggml-org/llama.cpp/pull/29823), the Ouro architecture and converter
- [ggml](https://github.com/ggml-org/ggml)
