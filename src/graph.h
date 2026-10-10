#pragma once

#include "model.h"
#include "kv_cache.h"

#include <string>
#include <vector>

struct ouro_graph_outputs {
    ggml_tensor * zR = nullptr;
    ggml_tensor * z1 = nullptr;
    ggml_tensor * h1 = nullptr;   // post-norm state at end of loop 1
    ggml_tensor * hR = nullptr;   // post-norm state at end of loop R
    ggml_tensor * g1 = nullptr;   // pre-norm layer-stack output at end of loop 1
    ggml_tensor * gR = nullptr;   // pre-norm layer-stack output at end of loop R
};

// Runs `tokens` through the R*48 logical-layer graph. Their K/V are written into
// `kv` at rows [n_past, n_past + tokens.size()) of each logical layer's slot, and
// each layer attends over positions [0, n_past + tokens.size()) via
// ggml_flash_attn_ext. Outputs remain valid until the next ouro_forward_kv call.
// `loops <= 0` means hp.n_loops. The caller must read outputs before the next call.
bool ouro_forward_kv(ouro_model & m, ouro_kv_cache & kv, const std::vector<int32_t> & tokens,
                     int n_past, int loops, ouro_graph_outputs & out, std::string & err);
