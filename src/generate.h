#pragma once

#include "graph.h"
#include "kv_cache.h"
#include "loopcd.h"
#include "model.h"
#include "sampling.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct ouro_gen_params {
    ouro_loopcd_params  loopcd;
    ouro_sampler_params sampling;
    int loops      = 0;    // 0 => hp.n_loops
    int eos        = 2;
    int max_tokens = 32;
};

// One sequence at a time. `begin` runs nothing yet; the first `next` prefills.
struct ouro_generator {
    ouro_model *   m  = nullptr;
    ouro_kv_cache * kv = nullptr;

    ouro_gen_params gp;
    int  ctx = 0;
    int  R   = 0;

    std::vector<int32_t> prompt, out_ids;
    bool prefilled = false, done = false;
    int  n_past = 0;
    std::mt19937 rng;

    ouro_graph_outputs taps;
    // guided / unguided / z1 logits at the prefill's last position (for the --dump-guided check)
    std::vector<float> first_guided, first_unguided, first_z1;

    bool begin(ouro_model & m, ouro_kv_cache & kv, const std::vector<int32_t> & prompt,
               const ouro_gen_params & gp, int ctx, std::string & err);

    // 1 = produced a token in `out`; 0 = finished (EOS / max_tokens / context full); -1 = error.
    int next(int32_t & out, std::string & err);

  private:
    int sample_from(int col, int32_t & out, std::string & err);
};
