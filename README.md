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
