// src/kitty_kv.cpp — M6 host-side Kitty-KV fake quantizer.
//
// Mirrors kitty_sim (tests/reference/kitty_quant.py): key pages are quantized per channel
// over the page's positions, with the top `boost_channels` channels by mean|k| promoted to
// `boost_bits`; values are quantized per token (group over head_dim), KIVI-style. The cache
// stays f16; this only rewrites values in place after a forward.
#include "kitty_kv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace {

// element k for (position p, head h, channel d) in the cache row layout is
// p * n_embd_kv + h * head_dim + d (see tests/unit/test_kv.cpp).

// quantize one channel over P positions, returning dequantized values
static void quant_channel(std::vector<float> & x, int stride, int P, int bits) {
    float mn = x[0], mx = x[0];
    for (int t = 1; t < P; ++t) { mn = std::fmin(mn, x[(size_t) t * stride]); mx = std::fmax(mx, x[(size_t) t * stride]); }
    const float range = std::fmax(mx - mn, 1e-4f);
    const int levels = (1 << bits) - 1;
    const float scale = range / (float) levels;
    for (int t = 0; t < P; ++t) {
        float & v = x[(size_t) t * stride];
        float q = std::round((v - mn) / scale);
        q = std::fmin(std::fmax(q, 0.0f), (float) levels);
        v = q * scale + mn;
    }
}

} // namespace

bool ouro_kv_kitty_quantize(ouro_kv_cache & kv, int n_kv, std::string & err) {
    if (!kv.kitty) return true;
    const auto & c = kv.kitty_cfg;
    const int H = kv.n_head_kv, D = kv.head_dim, E = kv.n_embd_kv, ctx = kv.n_ctx, S = kv.n_slots;
    if (c.page % 128 != 0) { err = "kitty: page must be a multiple of 128"; return false; }

    std::mt19937_64 rng(c.seed);
    auto row_bytes = [&]() { return (size_t) E * sizeof(ggml_fp16_t); };

    // ---- Key: quantize completed pages -------------------------------------
    std::vector<ggml_fp16_t> buf((size_t) E * c.page);
    std::vector<float> f((size_t) E * c.page);
    long long a = kv.kitty_k_done;                           // first unquantized position
    while (a < n_kv && ((a - c.sink) % c.page) != 0) ++a;    // align past the sink
    for (; a + c.page <= n_kv; a += c.page) {
        for (int s = 0; s < S; ++s) {
            const size_t off = ((size_t) s * ctx + a) * row_bytes();
            ggml_backend_tensor_get(kv.k, buf.data(), off, (size_t) E * c.page * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < buf.size(); ++i) f[i] = ggml_fp16_to_fp32(buf[i]);
            for (int h = 0; h < H; ++h) {
                // channel scores = mean |k| over the page
                std::vector<std::pair<float,int>> score(D);
                for (int d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (int t = 0; t < c.page; ++t)
                        acc += std::fabs((double) f[(size_t) t * E + h * D + d]);
                    score[d] = { (float) (acc / c.page), d };
                }
                std::vector<char> boost(D, 0);
                int d_boosted = std::min(c.boost_channels, D);
                if (c.random_select) {
                    std::vector<int> idx(D); for (int d = 0; d < D; ++d) idx[d] = d;
                    std::shuffle(idx.begin(), idx.end(), rng);
                    for (int i = 0; i < d_boosted; ++i) boost[idx[i]] = 1;
                } else {
                    std::partial_sort(score.begin(), score.begin() + d_boosted, score.end(),
                                      [](const auto & A, const auto & B){ return A.first > B.first; });
                    for (int i = 0; i < d_boosted; ++i) boost[score[i].second] = 1;
                }
                for (int d = 0; d < D; ++d) {
                    const int bits = boost[d] ? c.boost_bits : c.kbits;
                    // gather this channel's P values (with stride E), quantize, scatter back
                    static thread_local std::vector<float> col;
                    col.resize(c.page);
                    for (int t = 0; t < c.page; ++t) col[t] = f[(size_t) t * E + h * D + d];
                    quant_channel(col, 1, c.page, bits);
                    for (int t = 0; t < c.page; ++t) f[(size_t) t * E + h * D + d] = col[t];
                }
            }
            for (size_t i = 0; i < f.size(); ++i) buf[i] = ggml_fp32_to_fp16(f[i]);
            ggml_backend_tensor_set(kv.k, buf.data(), off, (size_t) E * c.page * sizeof(ggml_fp16_t));
        }
    }
    kv.kitty_k_done = a;

    // ---- Value: per-token (group over head_dim); keep the newest `page` fp16 ----
    const long long vstart = std::max((long long) c.sink, kv.kitty_v_done);
    const long long vstop  = (long long) n_kv - c.page;
    if (vstop > vstart) {
        const int T = (int) (vstop - vstart);
        std::vector<ggml_fp16_t> vbuf((size_t) E * T);
        for (int s = 0; s < S; ++s) {
            const size_t off = ((size_t) s * ctx + vstart) * row_bytes();
            ggml_backend_tensor_get(kv.v, vbuf.data(), off, vbuf.size() * sizeof(ggml_fp16_t));
            // per-token, per-head group over head_dim
            for (int t = 0; t < T; ++t) {
                for (int h = 0; h < H; ++h) {
                    float mn = ggml_fp16_to_fp32(vbuf[(size_t) t * E + h * D]);
                    float mx = mn;
                    for (int d = 1; d < D; ++d) {
                        float vv = ggml_fp16_to_fp32(vbuf[(size_t) t * E + h * D + d]);
                        mn = std::fmin(mn, vv); mx = std::fmax(mx, vv);
                    }
                    const int levels = (1 << c.vbits) - 1;
                    const float scale = std::fmax(mx - mn, 1e-4f) / (float) levels;
                    for (int d = 0; d < D; ++d) {
                        float vv = ggml_fp16_to_fp32(vbuf[(size_t) t * E + h * D + d]);
                        float q = std::round((vv - mn) / scale);
                        q = std::fmin(std::fmax(q, 0.0f), (float) levels);
                        vbuf[(size_t) t * E + h * D + d] = ggml_fp32_to_fp16(q * scale + mn);
                    }
                }
            }
            ggml_backend_tensor_set(kv.v, vbuf.data(), off, vbuf.size() * sizeof(ggml_fp16_t));
        }
        kv.kitty_v_done = vstop;
    }
    return true;
}
