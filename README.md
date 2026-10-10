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
