# loopcd

A minimal, Ouro-2.6B-Thinking-only inference engine with LoopCD contrastive decoding.

Links [ggml](https://github.com/ggml-org/ggml) (pinned) and copies the Ouro graph and converter
from llama.cpp PR #29823 as reference. Not a llama.cpp fork.

## Layout
- `src/` - engine
- `tools/` - asset fetch + GGUF conversion
- `tests/` - the correctness gate and unit tests
- `docs/superpowers/` - specs and plans

## Build (CPU)
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

## Correctness gate (M1)
`loopcd logits` must match a HuggingFace f32 forward pass at R=4 and R=2. The gate runs on an
**f32** GGUF: ggml re-quantizes f32 activations to the weight type inside f16/bf16 matmuls, and the
looped stack amplifies that per-layer error past the tolerance. Convert with `--outtype f32`,
generate the reference (`tests/reference/harness.py --loops {4,2}`), then
`build/loopcd logits -m models/ouro/ouro.f32.gguf --tokens-file tests/reference/out/<id>.r<N>.tokens.txt --loops <N> --dump out/<id>.r<N>`
and `python tests/reference/compare.py --dump out/<id>.r<N> --ref tests/reference/out/<id>.r<N>.npz`.

## Gate metric (M2)
`tests/reference/compare.py` reads a tolerance file with three roles:

- **`mag`** - gated magnitude: `key` fails if `max|d| > bound`. (The flat M1 `tests/gate_tol.json`
  `{zR,z1,h1,hR}` is accepted and treated as `mag`.)
- **`info`** - informational magnitude: reported against a ceiling, never fails.
- **`argmax`** - enforced **only for the keys in `enforce_argmax`** (default `["zR"]`). `zR` is the
  model's output; `z1` is an intermediate (the LoopCD reference) whose near-tie argmax can flip under
  quantization, so it is reported, not enforced.

The gates:

- **run 1 / run 2 (graph fidelity):** `f32` weights + `f32` KV, R=4 and R=2, `tests/gate_tol.json`
  (all four keys `mag`, ~1e-5..2e-4). This measures the graph, not quantized-KV precision.
- **run 3 (serving stack):** Q4_K_M + q8_0 KV + Vulkan, `tests/gate_tol_serving.json`. Enforced
  criterion: **`zR` argmax at every position**. The zR/z1 magnitudes are informational ceilings
  (Q4_K_M weight quantization; observed worst case `zR max|d| = 4.717`), revisable only with a reason.
- The f16 / q8_0 KV path itself is covered by `test_kv` (exact write-path equality vs ggml's quantizer)
  and by run 3.
