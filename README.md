# loopcd

A from-scratch inference engine for Ouro-2.6B-Thinking with LoopCD contrastive decoding.
C++ on top of [ggml](https://github.com/ggml-org/ggml), running on one machine's
AMD Radeon 890M iGPU through Vulkan, serving an OpenAI-compatible chat endpoint.

It is not a llama.cpp fork. It links ggml and copies a few specific pieces from
llama.cpp PR #29823, the PR that added Ouro support: the Ouro graph, the Ouro GGUF
converter, and the BPE tokenizer. The engine handles one architecture, Ouro, and one model.

I built it to check a paper's claim on my own hardware instead of taking it on faith,
and to measure what the technique costs to actually run.

## Background

Ouro is a looped transformer. Instead of 192 distinct layers it has 48 layers applied
4 times, reusing the same weights each pass, so the reasoning happens inside the loop
rather than in emitted chain-of-thought tokens. It comes from
*Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed, arXiv 2510.25741,
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
arXiv 2610.02185. The "almost free" claim is about a fixed loop count, where LoopCD adds
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

## Layout

```
src/            engine: model loader, graph, KV cache, loopcd, tokenizer, chat template,
                sampling, decode loop, server
tools/          asset fetch, Ouro GGUF conversion, GGUF metadata check
tests/
  reference/    HuggingFace reference harness and numeric gate comparison
  unit/         focused unit tests: ops, kv, loopcd math, sampling, tokenizer
  eval/         M5 measurement harness: run_eval.py, score.py, report.py, items/, RESULTS.md
third_party/    vendored cpp-httplib and nlohmann/json
```

## Build

Needs CMake 3.20 or newer, a C++17 compiler, Ninja, and for the shipping backend the
Vulkan SDK. CMake fetches and pins ggml.

```sh
# Vulkan, the shipping backend
cmake -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk

# CPU only, no Vulkan SDK needed
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DLOOPCD_VULKAN=OFF
cmake --build build-cpu
```

Fetch the model config and tokenizer, convert to GGUF, and quantize:

```sh
python tools/fetch_assets.py                 # config, tokenizer, custom code into models/ouro/
python tools/convert_ouro.py                 # safetensors -> ouro.f16.gguf, arch="ouro"
llama-quantize.exe models/ouro/ouro.f16.gguf models/ouro/ouro.Q4_K_M.gguf Q4_K_M
```

The quantize step needs the Ouro-aware build, which is llama.cpp PR #29823. Upstream
`llama-quantize` rejects `arch="ouro"`.

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

LoopCD modes are `off`, `logits`, and `hidden`. `--loopcd-adaptive` anneals ω up to
`--loopcd-omega-max`. `--loops N` sets the recurrent depth, default 4, which is what the
model was trained at.

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

## Limitations

- Ouro only. No other architectures, by design.
- One request at a time. There is no batching, so one slow generation blocks the server.
- Windows with Vulkan or CPU. No CUDA, no Metal.
- The adaptive-depth and `early_exit_gate` path is out of scope.
- Prefill is dense and runs the stack R times, so long prompts are slow. Keep `--ctx`
  no larger than you need.

## References

- *Scaling Latent Reasoning via Looped Language Models*, ByteDance Seed, arXiv 2510.25741
- *Decoding Looped Transformers Better for (Almost) Free*, Apple, arXiv 2610.02185
- llama.cpp PR #29823, the Ouro architecture and converter
- [ggml](https://github.com/ggml-org/ggml)
