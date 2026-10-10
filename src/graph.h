#pragma once

#include "model.h"
#include <vector>

struct ouro_graph_outputs {
    ggml_tensor * zR = nullptr;
    ggml_tensor * z1 = nullptr;
    ggml_tensor * h1 = nullptr;
    ggml_tensor * hR = nullptr;
};

// Builds and runs one full-sequence prefill. Outputs remain valid until the next call.
bool ouro_forward(ouro_model & m, const std::vector<int32_t> & tokens, int loops,
                  ouro_graph_outputs & out, std::string & err);
