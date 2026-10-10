#pragma once

#include <cstdint>
#include <random>
#include <vector>

// Request/CLI sampling parameters (spec 4.6). temp <= 0 means greedy.
struct ouro_sampler_params {
    float    temp           = 0.0f;
    int      top_k          = 0;      // 0 => disabled
    float    top_p          = 1.0f;   // >= 1 => disabled
    float    repeat_penalty = 1.0f;   // 1 => disabled
    int      repeat_last_n  = 64;
    uint32_t seed           = 1234;
};

// In-place softmax with temperature. temp <= 0 leaves x unchanged.
void ouro_softmax_inplace(std::vector<float> & x, float temp);

// logit = logit < 0 ? logit * p : logit / p for each token id in `recent` (last `repeat_last_n`).
void ouro_apply_repeat_penalty(std::vector<float> & logits, const std::vector<int32_t> & recent,
                               int repeat_last_n, float penalty);

// Zeroes everything outside the top-k / nucleus (top-p); assumes `probs` are non-negative.
void ouro_top_k_top_p(std::vector<float> & probs, int top_k, float top_p);

// Samples one token id from `logits` (not modified). Deterministic given `rng` state.
int32_t ouro_sample(const std::vector<float> & logits, const std::vector<int32_t> & recent,
                    const ouro_sampler_params & sp, std::mt19937 & rng);
