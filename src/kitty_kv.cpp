// src/kitty_kv.cpp — M6 host-side Kitty-KV fake quantizer.
//
// Mirrors kitty_sim (tests/reference/kitty_quant.py): key pages are quantized per channel
// over the page's positions, with the top `boost_channels` channels by mean|k| promoted to
// `boost_bits`; values are quantized per token (group over head_dim), KIVI-style. The cache
// stays f16; this only rewrites values in place after a forward.
#include "kitty_kv.h"
#include "cache_layout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
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
    const int H = kv.n_head_kv, D = kv.head_dim, E = kv.n_embd_kv, S = kv.n_slots;
    if (c.page <= 0) { err = "kitty: page must be > 0"; return false; }

    std::mt19937_64 & rng = kv.kitty_rng;   // seeded at init, advanced (not reseeded) per call
    auto row_bytes = [&]() { return (size_t) E * sizeof(ggml_fp16_t); };

    // ---- Key: quantize completed pages -------------------------------------
    std::vector<ggml_fp16_t> buf((size_t) E * c.page);
    std::vector<float> f((size_t) E * c.page);
    long long a = kv.kitty_k_done;                           // first unquantized position
    while (a < n_kv && ((a - c.sink) % c.page) != 0) ++a;    // align past the sink
    for (; a + c.page <= n_kv; a += c.page) {
        for (int s = 0; s < S; ++s) {
            ggml_backend_tensor_get(kv.k[s], buf.data(), (size_t) a * row_bytes(),
                                    (size_t) E * c.page * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < buf.size(); ++i) f[i] = ggml_fp16_to_fp32(buf[i]);
            for (int h = 0; h < H; ++h) {
                // channel scores = mean |k| over the page. Stored as double to match the
                // packed emitter's select_boost exactly (Both-modes requirement): a float
                // score would rank near-ties differently and promote a different channel set.
                std::vector<std::pair<double,int>> score(D);
                for (int d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (int t = 0; t < c.page; ++t)
                        acc += std::fabs((double) f[(size_t) t * E + h * D + d]);
                    score[d] = { acc / c.page, d };
                }
                std::vector<char> boost(D, 0);
                int d_boosted = std::min(c.boost_channels, D);
                if (c.random_select) {
                    std::vector<int> idx(D); for (int d = 0; d < D; ++d) idx[d] = d;
                    std::shuffle(idx.begin(), idx.end(), rng);
                    for (int i = 0; i < d_boosted; ++i) boost[idx[i]] = 1;
                } else {
                    // Deterministic promotion (spec §7.5): sort by (mean|k| desc, channel asc).
                    std::stable_sort(score.begin(), score.end(),
                                     [](const auto & A, const auto & B){
                                         if (A.first != B.first) return A.first > B.first;
                                         return A.second < B.second;
                                     });
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
            ggml_backend_tensor_set(kv.k[s], buf.data(), (size_t) a * row_bytes(),
                                    (size_t) E * c.page * sizeof(ggml_fp16_t));
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
            ggml_backend_tensor_get(kv.v[s], vbuf.data(), (size_t) vstart * row_bytes(),
                                    vbuf.size() * sizeof(ggml_fp16_t));
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
            ggml_backend_tensor_set(kv.v[s], vbuf.data(), (size_t) vstart * row_bytes(),
                                    vbuf.size() * sizeof(ggml_fp16_t));
        }
        kv.kitty_v_done = vstop;
    }
    return true;
}

// ---------------------------------------------------------------------------
// M7b packed emitter.
// ---------------------------------------------------------------------------

namespace {

using namespace kitty_layout;

// Deterministic top-D_BOOST channel selection for one head's page.
// `col[d]` holds the page's P values for channel d; returns boost_idx[D]
// (sparse slot in [0,D_BOOST) for promoted channels, D_BOOST for the rest)
// and the ranked channel order in `rank[0..D_BOOST-1]`.
static void select_boost(const std::vector<float> & col, int D_, int P_, int d_boosted,
                         std::vector<uint8_t> & boost_idx, std::vector<int> & rank) {
    std::vector<std::pair<double,int>> score(D_);
    for (int d = 0; d < D_; ++d) {
        double acc = 0.0;
        for (int t = 0; t < P_; ++t) acc += std::fabs((double) col[(size_t) d * P_ + t]);
        score[d] = { acc / P_, d };
    }
    std::stable_sort(score.begin(), score.end(), [](const auto & A, const auto & B){
        if (A.first != B.first) return A.first > B.first;
        return A.second < B.second;
    });
    boost_idx.assign(D_, (uint8_t) D_BOOST);
    rank.clear();
    for (int i = 0; i < d_boosted; ++i) {
        rank.push_back(score[i].second);
        boost_idx[score[i].second] = (uint8_t) i;
    }
}

// Quantize one channel (P_ values) to `bits`, returning the integer codes and
// the fp16 (scale, min). Mirrors the M6 C++ math exactly (eps 1e-4, spec decision #5).
static void quant_channel_codes(const std::vector<float> & x, int P_, int bits,
                                std::vector<uint8_t> & q, ggml_fp16_t & scale_h, ggml_fp16_t & min_h) {
    float mn = x[0], mx = x[0];
    for (int t = 1; t < P_; ++t) { mn = std::fmin(mn, x[t]); mx = std::fmax(mx, x[t]); }
    const float range = std::fmax(mx - mn, 1e-4f);
    const int levels = (1 << bits) - 1;
    const float scale = range / (float) levels;
    q.assign(P_, 0);
    for (int t = 0; t < P_; ++t) {
        float v = std::round((x[t] - mn) / scale);
        v = std::fmin(std::fmax(v, 0.0f), (float) levels);
        q[t] = (uint8_t) v;
    }
    scale_h = ggml_fp32_to_fp16(scale);
    min_h   = ggml_fp32_to_fp16(mn);
}

static void put_u16(std::vector<uint8_t> & buf, size_t off, ggml_fp16_t h) {
    std::memcpy(buf.data() + off, &h, sizeof(h));
}

// Pack one K page from staging rows [sink, sink+P) for every slot into page `pg`.
static bool pack_k_page(ouro_kv_cache & kv, int pg) {
    const auto & c = kv.kitty_cfg;
    const int H = kv.n_head_kv, D_ = kv.head_dim, E = kv.n_embd_kv, S = kv.n_slots;
    const size_t row_bytes = (size_t) E * sizeof(ggml_fp16_t);
    const size_t page_bytes = k_page_bytes(H);
    const int d_boosted = std::min(c.boost_channels > 0 ? c.boost_channels : D_ / 4, D_);

    std::vector<ggml_fp16_t> stage((size_t) E * P);
    std::vector<uint8_t> page(page_bytes);
    std::vector<float> col((size_t) D_ * P);   // one head's [D, P] page (indexed d*P + t)
    std::vector<uint8_t> q, boost_idx;
    std::vector<int> rank;

    for (int s = 0; s < S; ++s) {
        std::fill(page.begin(), page.end(), 0);
        ggml_backend_tensor_get(kv.k[s], stage.data(), (size_t) c.sink * row_bytes,
                                (size_t) E * P * sizeof(ggml_fp16_t));
        for (int h = 0; h < H; ++h) {
            const size_t hb = k_head_off(h);
            // gather all channels for this head, then select/quantize
            for (int d = 0; d < D_; ++d)
                for (int t = 0; t < P; ++t)
                    col[(size_t) d * P + t] = ggml_fp16_to_fp32(stage[(size_t) t * E + h * D_ + d]);
            select_boost(col, D_, P, d_boosted, boost_idx, rank);
            for (int d = 0; d < D_; ++d) {
                const bool boosted = boost_idx[d] < (uint8_t) D_BOOST;
                const int bits = boosted ? c.boost_bits : c.kbits;
                ggml_fp16_t sh, mh;
                std::vector<float> x(col.begin() + (size_t) d * P, col.begin() + (size_t) (d + 1) * P);
                quant_channel_codes(x, P, bits, q, sh, mh);
                page[k_idx_off(h, d)] = boost_idx[d];
                put_u16(page, k_meta_off(h, d), sh);
                put_u16(page, k_meta_off(h, d) + 2, mh);
                for (int t = 0; t < P; ++t) {
                    const uint8_t low = q[t] & 0x3;
                    page[k_low_off(h, d, t)] |= (uint8_t) (low << ((t & 3) * 2));
                    if (boosted) {
                        const uint8_t high = (q[t] >> 2) & 0x3;
                        page[k_high_off(h, boost_idx[d], t)] |= (uint8_t) (high << ((t & 3) * 2));
                    }
                }
            }
        }
        const size_t dst_off = (size_t) pg * page_bytes;
        if (dst_off + page_bytes > (size_t) ggml_nbytes(kv.k_planes[s])) return false;
        ggml_backend_tensor_set(kv.k_planes[s], page.data(), dst_off, page_bytes);
    }
    return true;
}

// Pack one V token at absolute position `pos` from its ring row into the packed plane.
static bool pack_v_token(ouro_kv_cache & kv, long long pos) {
    const auto & c = kv.kitty_cfg;
    const int H = kv.n_head_kv, D_ = kv.head_dim, E = kv.n_embd_kv, S = kv.n_slots;
    const size_t row_bytes = (size_t) E * sizeof(ggml_fp16_t);
    const long long rel = pos - c.sink;                 // [0, ...)
    const int pg = (int) (rel / P);
    const int slot = (int) (rel % P);
    const size_t vpage_bytes = v_page_bytes(H);
    const int row = c.sink + (int) (rel % P);           // ring row

    std::vector<ggml_fp16_t> stage((size_t) E);
    std::vector<uint8_t> page(vpage_bytes);
    for (int s = 0; s < S; ++s) {
        std::fill(page.begin(), page.end(), 0);
        ggml_backend_tensor_get(kv.v[s], stage.data(), (size_t) row * row_bytes,
                                (size_t) E * sizeof(ggml_fp16_t));
        for (int h = 0; h < H; ++h) {
            const size_t hb = v_head_off(h);
            float mn = ggml_fp16_to_fp32(stage[h * D_]);
            float mx = mn;
            for (int d = 1; d < D_; ++d) {
                float vv = ggml_fp16_to_fp32(stage[h * D_ + d]);
                mn = std::fmin(mn, vv); mx = std::fmax(mx, vv);
            }
            const int levels = (1 << c.vbits) - 1;
            const float scale = std::fmax(mx - mn, 1e-4f) / (float) levels;
            put_u16(page, hb + V_OFF_META + (size_t) slot * 4,     ggml_fp32_to_fp16(scale));
            put_u16(page, hb + V_OFF_META + (size_t) slot * 4 + 2, ggml_fp32_to_fp16(mn));
            for (int d = 0; d < D_; ++d) {
                float vv = ggml_fp16_to_fp32(stage[h * D_ + d]);
                float qq = std::round((vv - mn) / scale);
                qq = std::fmin(std::fmax(qq, 0.0f), (float) levels);
                page[hb + (size_t) slot * (D_ / 4) + (size_t) (d / 4)] |=
                    (uint8_t) (((uint8_t) qq & 0x3) << ((d & 3) * 2));
            }
        }
        const size_t dst_off = (size_t) pg * vpage_bytes;
        if (dst_off + vpage_bytes > (size_t) ggml_nbytes(kv.v_planes[s])) return false;
        ggml_backend_tensor_set(kv.v_planes[s], page.data(), dst_off, vpage_bytes);
    }
    return true;
}

} // namespace

bool ouro_kv_kitty_pack_evict(ouro_kv_cache & kv, int n_kv_new, std::string & err) {
    if (!kv.kitty || !kv.kitty_cfg.packed) return true;
    const auto & c = kv.kitty_cfg;
    const long long target = (long long) n_kv_new - c.page;   // packed V covers [sink, target)
    if (target <= c.sink) return true;
    while (kv.kitty_v_packed < target - c.sink) {
        if (!pack_v_token(kv, c.sink + kv.kitty_v_packed)) { err = "kitty: V plane overflow"; return false; }
        ++kv.kitty_v_packed;
    }
    return true;
}

bool ouro_kv_kitty_pack_pages(ouro_kv_cache & kv, int n_kv_new, std::string & err) {
    if (!kv.kitty || !kv.kitty_cfg.packed) return true;
    const auto & c = kv.kitty_cfg;
    const long long n_kpages = ((long long) n_kv_new - c.sink) / c.page;   // pages fully within [sink,n_kv)
    if (n_kpages - kv.kitty_k_pages > 1) { err = "kitty: more than one page completed in a chunk"; return false; }
    while (kv.kitty_k_pages < n_kpages) {
        if (!pack_k_page(kv, (int) kv.kitty_k_pages)) { err = "kitty: K plane overflow"; return false; }
        ++kv.kitty_k_pages;
    }
    return true;
}
