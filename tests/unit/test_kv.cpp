// Focused tests for the M2 KV cache path:
//  A. the f32 -> q8_0 write (ggml_cpy) matches ggml's reference quantizer and the
//     round-trip error stays inside the q8_0 step;
//  B. the cache read view + causal mask feed ggml_flash_attn_ext the values a naive
//     softmax attention reference expects.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "kv_cache.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const char * name, double got, double want, double tol) {
    const double d = std::fabs(got - want);
    if (d > tol) { printf("FAIL %s: got %.6g want %.6g (d=%.3g)\n", name, got, want, d); g_fail = 1; }
    else         { printf("ok   %s (d=%.3g)\n", name, d); }
}

static void test_quant_roundtrip() {
    const int n_per_row = 2048;  // head_dim * n_head_kv
    const int nrows = 3;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> src((size_t) n_per_row * nrows);
    for (auto & f : src) f = dist(rng);

    std::vector<uint8_t> ref(ggml_row_size(GGML_TYPE_Q8_0, n_per_row) * nrows);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), ref.data(), 0, nrows, n_per_row, nullptr);

    ggml_init_params ip = { 32 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  n_per_row, nrows);
    ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, n_per_row, nrows);
    memcpy(s->data, src.data(), src.size() * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, s, d));
    ggml_backend_graph_compute(cpu, gf);

    std::vector<uint8_t> got(ggml_nbytes(d));
    memcpy(got.data(), d->data, got.size());

    std::vector<float> ref_f(src.size()), got_f(src.size());
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(ref.data(), ref_f.data(), (int64_t) src.size());
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(got.data(), got_f.data(), (int64_t) src.size());

    double max_src = 0.0;
    for (float f : src) max_src = std::max(max_src, (double) std::fabs(f));

    double max_diff = 0.0, max_rt = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        max_diff = std::max(max_diff, (double) std::fabs(got_f[i] - ref_f[i]));
        max_rt   = std::max(max_rt,   (double) std::fabs(got_f[i] - src[i]));
    }
    check("q8_0 write path == ggml reference quantizer", max_diff / max_src, 0.0, 1e-3);
    check("q8_0 round-trip max rel err",                max_rt   / max_src, 0.0, 0.02);

    ggml_backend_free(cpu);
    ggml_free(ctx);
}

static void test_masked_attn() {
    const int hd = 8, nh = 2, n = 4, n_slots = 1, n_ctx = 8;
    const int n_kv = 4, n_past = 0;
    const float scale = 1.0f / std::sqrt((float) hd);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ouro_kv_cache kv;
    std::string err;
    const bool ok = ouro_kv_cache_init(kv, cpu, hd, nh, n_ctx, n_slots, GGML_TYPE_F16, nullptr, err);
    check("kv_cache_init f16", ok ? 1.0 : 0.0, 1.0, 0.0);
    if (!ok) { printf("err: %s\n", err.c_str()); g_fail = 1; ggml_backend_free(cpu); return; }

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> qd((size_t) hd * nh * n), kd((size_t) hd * nh * n), vd((size_t) hd * nh * n);
    for (auto & f : qd) f = dist(rng);
    for (auto & f : kd) f = dist(rng);
    for (auto & f : vd) f = dist(rng);

    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, n);
    ggml_tensor * K = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, n);
    ggml_tensor * V = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, n);
    memcpy(Q->data, qd.data(), qd.size() * sizeof(float));
    memcpy(K->data, kd.data(), kd.size() * sizeof(float));
    memcpy(V->data, vd.data(), vd.size() * sizeof(float));

    // write K/V into slot 0 through the exact view path the graph uses
    {
        ggml_cgraph * wg = ggml_new_graph(ctx);
        ggml_tensor * K2 = ggml_view_2d(ctx, K, hd * nh, n, K->nb[2], 0);
        ggml_tensor * V2 = ggml_view_2d(ctx, V, hd * nh, n, V->nb[2], 0);
        ggml_build_forward_expand(wg, ggml_cpy(ctx, K2, ouro_kv_k_write(kv, ctx, 0, 0, n)));
        ggml_build_forward_expand(wg, ggml_cpy(ctx, V2, ouro_kv_v_write(kv, ctx, 0, 0, n)));
        ggml_backend_graph_compute(cpu, wg);
    }

    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n);
    {
        ggml_fp16_t * md = (ggml_fp16_t *) mask->data;
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n_kv; ++j)
                md[(size_t) i * n_kv + j] = (j <= n_past + i) ? keep : drop;
    }

    ggml_tensor * Qp = ggml_permute(ctx, Q, 0, 2, 1, 3);
    ggml_tensor * Kc = ggml_permute(ctx, ouro_kv_k(kv, ctx, 0, n_kv), 0, 2, 1, 3);
    ggml_tensor * Vc = ggml_permute(ctx, ouro_kv_v(kv, ctx, 0, n_kv), 0, 2, 1, 3);

    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qp, Kc, Vc, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(attn, GGML_PREC_F32);
    ggml_tensor * res = ggml_reshape_2d(ctx, attn, hd * nh, n);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> got((size_t) hd * nh * n);
    memcpy(got.data(), res->data, got.size() * sizeof(float));

    // read the f16 K/V back, in the cache row layout: row for position p holds h*head_dim + d
    std::vector<ggml_fp16_t> kraw((size_t) kv.n_embd_kv * n_ctx * n_slots);
    std::vector<ggml_fp16_t> vraw((size_t) kv.n_embd_kv * n_ctx * n_slots);
    ggml_backend_tensor_get(kv.k, kraw.data(), 0, ggml_nbytes(kv.k));
    ggml_backend_tensor_get(kv.v, vraw.data(), 0, ggml_nbytes(kv.v));

    double max_d = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int h = 0; h < nh; ++h) {
            double score[n_kv];
            double smax = -INFINITY;
            for (int j = 0; j < n_kv; ++j) {
                double acc = 0.0;
                for (int d2 = 0; d2 < hd; ++d2) {
                    const double qv = qd[(size_t) h * hd + d2 + (size_t) hd * nh * i];
                    const double kc = ggml_fp16_to_fp32(kraw[(size_t) j * kv.n_embd_kv + h * hd + d2]);
                    acc += qv * kc;
                }
                score[j] = acc * scale;
                if (j > n_past + i) score[j] = -INFINITY;
                smax = std::max(smax, score[j]);
            }
            double denom = 0.0;
            for (int j = 0; j < n_kv; ++j) { score[j] = std::exp(score[j] - smax); denom += score[j]; }
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < n_kv; ++j) {
                    const double vc = ggml_fp16_to_fp32(vraw[(size_t) j * kv.n_embd_kv + h * hd + d]);
                    acc += (score[j] / denom) * vc;
                }
                const double e = got[(size_t) i * hd * nh + h * hd + d];
                max_d = std::max(max_d, std::fabs(e - acc));
            }
        }
    }
    check("masked flash_attn_ext vs naive softmax", max_d, 0.0, 5e-3);

    ouro_kv_cache_free(kv);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main() {
    test_quant_roundtrip();
    test_masked_attn();
    printf(g_fail ? "KV: FAIL\n" : "KV: PASS\n");
    return g_fail;
}
