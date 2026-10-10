#pragma once

#include "kv_cache.h"
#include "loopcd.h"
#include "sampling.h"

#include <string>

struct ouro_server_config {
    std::string         model_path;
    std::string         device  = "vulkan";
    std::string         kv_type = "q8_0";   // q8_0 | f16 | f32 | kitty
    int                 ctx     = 4096;
    int                 loops   = 0;      // 0 => hp.n_loops
    ouro_loopcd_params  loopcd;           // default: logits, omega 0.3, cap 1.0
    ouro_sampler_params sampling;         // default: greedy (temp 0)
    int                 eos     = 2;
    std::string         host    = "127.0.0.1";
    int                 port    = 8080;
    bool                 kitty      = false;  // M6: fake-quantize the f16 cache
    ouro_kitty_config    kitty_cfg{};         // valid when kitty == true
};

// Loads the model + tokenizer once and serves until the process is stopped.
// Requests are handled serially (one generation at a time).
int ouro_serve(const ouro_server_config & cfg);
