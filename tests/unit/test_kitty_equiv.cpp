// tests/unit/test_kitty_equiv.cpp — M7b/c cache-level equivalence (§12.3 enforced item 3).
//
// The §12.3 gate enforces "packed ops vs an independent reference" + the byte fixture.
// This adds the statement that licenses attributing the packed-vs-f16 end-to-end
// difference to rounding rather than to a storage bug: for the SAME input page, the
// dequantized packed values and the values the f16 mode STORES must differ by no more
// than fp16 rounding.
//
//   packed : stores q + fp16(scale) + fp16(min), dequantizes in f32 -> d_p = q*fp16(s) + fp16(m)
//   f16    : stores fp16(q*s + m) directly                          -> d_f = fp16(q*s + m)
//
// so |d_p - d_f| <= 2^-11*(range + |m|)   [fp16 rounding of the metadata]
//                 + 2^-11*|d|            [fp16 rounding of the stored value]
// We assert a 2x-conservative 2^-10 scale. A real storage defect (wrong plane offset,
// wrong index, wrong code assembly) exceeds it by orders of magnitude.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "kv_cache.h"
#include "kitty_kv.h"
#include "cache_layout.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace kitty_layout;

static int g_fail = 0;

static ggml_fp16_t rd16(const uint8_t * p, size_t off) {
    ggml_fp16_t h; std::memcpy(&h, p + off, 2); return h;
}

// Dequantize one packed K page into token-major values [(t*H + h)*D + d] (the cache layout).
static void dequant_page(const uint8_t * page, int H, int D_, int P_,
                         std::vector<float> & out) {
    out.assign((size_t) P_ * H * D_, 0.0f);
    for (int h = 0; h < H; ++h) {
        const uint8_t * head = page + k_head_off(h);
        for (int d = 0; d < D_; ++d) {
            const int idx = head[K_OFF_IDX + d];
            const float s = ggml_fp16_to_fp32(rd16(head, K_OFF_META + (size_t) d * 4));
            const float m = ggml_fp16_to_fp32(rd16(head, K_OFF_META + (size_t) d * 4 + 2));
            for (int t = 0; t < P_; ++t) {
                const int lo = (head[(size_t) d * (P_ / 4) + t / 4] >> ((t & 3) * 2)) & 3;
                int code = lo;
                if (idx < D_BOOST) {
                    const int hi = (head[K_OFF_HIGH + (size_t) idx * (P_ / 4) + t / 4]
                                    >> ((t & 3) * 2)) & 3;
                    code = lo | (hi << 2);
                }
                out[((size_t) t * H + h) * D_ + d] = (float) code * s + m;
            }
        }
    }
}

int main() {
    const int H = 16, D_ = 128, P_ = 128, E = D_ * H;
    const int sink = 0;
    std::string err;

    // Deterministic input page, token-major rows [P][E] (the cache row layout).
    std::vector<ggml_fp16_t> in((size_t) P_ * E);
    for (int t = 0; t < P_; ++t)
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D_; ++d) {
                const float v = 1.7f * std::sin(0.013f * (float)(t * 7 + h * 31 + d * 101))
                              + 0.9f * std::cos(0.021f * (float)(t + h * 3 + d * 5));
                in[(size_t) t * E + h * D_ + d] = ggml_fp32_to_fp16(v);
            }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { printf("FAIL cpu backend\n"); return 1; }

    // ---- f16 mode: quantize the page, read what it STORES ----
    std::vector<float> d_f;
    {
        ouro_kv_cache kv; ouro_kitty_config c;
        c.sink = sink; c.page = P_; c.boost_channels = D_BOOST; c.kbits = 2;
        c.vbits = 2; c.boost_bits = 4; c.packed = false;
        if (!ouro_kv_cache_init(kv, cpu, D_, H, P_, 1, GGML_TYPE_F16, &c, err)) {
            printf("FAIL f16 init: %s\n", err.c_str()); return 1;
        }
        ggml_backend_tensor_set(kv.k[0], in.data(), 0, in.size() * sizeof(ggml_fp16_t));
        if (!ouro_kv_kitty_quantize(kv, P_, err)) { printf("FAIL f16 quantize: %s\n", err.c_str()); return 1; }
        std::vector<ggml_fp16_t> stored((size_t) P_ * E);
        ggml_backend_tensor_get(kv.k[0], stored.data(), 0, stored.size() * sizeof(ggml_fp16_t));
        d_f.resize(stored.size());
        for (size_t i = 0; i < stored.size(); ++i) d_f[i] = ggml_fp16_to_fp32(stored[i]);
        ouro_kv_cache_free(kv);
    }

    // ---- packed mode: pack the page, dequantize it ----
    std::vector<float> d_p;
    {
        ouro_kv_cache kv; ouro_kitty_config c;
        c.sink = sink; c.page = P_; c.boost_channels = D_BOOST; c.kbits = 2;
        c.vbits = 2; c.boost_bits = 4; c.packed = true;
        if (!ouro_kv_cache_init(kv, cpu, D_, H, P_, 1, GGML_TYPE_F16, &c, err)) {
            printf("FAIL packed init: %s\n", err.c_str()); return 1;
        }
        ggml_backend_tensor_set(kv.k[0], in.data(), 0, in.size() * sizeof(ggml_fp16_t));
        if (!ouro_kv_kitty_pack_pages(kv, P_, err)) { printf("FAIL packed pack: %s\n", err.c_str()); return 1; }
        std::vector<uint8_t> page(k_page_bytes(H));
        ggml_backend_tensor_get(kv.k_planes[0], page.data(), 0, page.size());
        dequant_page(page.data(), H, D_, P_, d_p);
        ouro_kv_cache_free(kv);
    }

    // ---- compare against the analytic bound ----
    double maxd = 0.0;
    for (size_t i = 0; i < d_f.size(); ++i)
        maxd = std::fmax(maxd, std::fabs((double) d_p[i] - d_f[i]));

    double range_max = 0.0, mabs_max = 0.0, vabs_max = 0.0;
    for (int h = 0; h < H; ++h)
        for (int d = 0; d < D_; ++d) {
            double mn = 1e30, mx = -1e30;
            for (int t = 0; t < P_; ++t) {
                const double v = ggml_fp16_to_fp32(in[(size_t) t * E + h * D_ + d]);
                mn = std::fmin(mn, v); mx = std::fmax(mx, v);
                vabs_max = std::fmax(vabs_max, std::fabs(v));
            }
            range_max = std::fmax(range_max, mx - mn);
            mabs_max  = std::fmax(mabs_max, std::fabs(mn));
        }
    const double bound = std::pow(2.0, -10.0) * (range_max + mabs_max + vabs_max);  // 2x-conservative

    printf("max|d_packed - d_f16| = %.4g   bound = %.4g   (range=%.3g |m|=%.3g |v|=%.3g)\n",
           maxd, bound, range_max, mabs_max, vabs_max);

    if (maxd <= bound) printf("ok   packed dequant vs f16 stored within the fp16-rounding bound\n");
    else { printf("FAIL packed dequant vs f16 stored EXCEEDS the fp16-rounding bound\n"); g_fail = 1; }

    ggml_backend_free(cpu);
    printf(g_fail ? "KITTY_EQUIV: FAIL\n" : "KITTY_EQUIV: PASS\n");
    return g_fail;
}
