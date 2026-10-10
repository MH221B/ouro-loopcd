#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>

void ouro_softmax_inplace(std::vector<float> & x, float temp) {
    if (temp <= 0.0f || x.empty()) return;
    const float mx = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (float & v : x) { v = std::exp((v - mx) / temp); sum += v; }
    for (float & v : x) v = (float) (v / sum);
}

void ouro_apply_repeat_penalty(std::vector<float> & logits, const std::vector<int32_t> & recent,
                               int repeat_last_n, float penalty) {
    if (penalty == 1.0f || recent.empty()) return;
    const int n = (int) recent.size();
    const int start = repeat_last_n > 0 ? std::max(0, n - repeat_last_n) : 0;
    for (int i = start; i < n; ++i) {
        const int32_t t = recent[i];
        if (t < 0 || t >= (int32_t) logits.size()) continue;
        logits[t] = logits[t] < 0.0f ? logits[t] * penalty : logits[t] / penalty;
    }
}

void ouro_top_k_top_p(std::vector<float> & probs, int top_k, float top_p) {
    const int n = (int) probs.size();
    if (n == 0) return;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    if (top_k > 0 && top_k < n) {
        for (int i = top_k; i < n; ++i) probs[idx[i]] = 0.0f;
    }
    if (top_p < 1.0f) {
        double cum = 0.0;
        for (int i = 0; i < n; ++i) {
            cum += probs[idx[i]];
            if (cum >= top_p) {
                for (int j = i + 1; j < n; ++j) probs[idx[j]] = 0.0f;
                break;
            }
        }
    }
}

int32_t ouro_sample(const std::vector<float> & logits, const std::vector<int32_t> & recent,
                    const ouro_sampler_params & sp, std::mt19937 & rng) {
    const int n = (int) logits.size();
    if (n == 0) return -1;
    std::vector<float> l = logits;
    ouro_apply_repeat_penalty(l, recent, sp.repeat_last_n, sp.repeat_penalty);

    if (sp.temp <= 0.0f) {
        return (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
    }
    ouro_softmax_inplace(l, sp.temp);
    ouro_top_k_top_p(l, sp.top_k, sp.top_p);

    std::discrete_distribution<int> dist(l.begin(), l.end());
    return (int32_t) dist(rng);
}
