// tests/unit/test_kitty_ops.cpp — M7c: the packed ops (CPU, and Vulkan when available)
// must match an independent dequantize-and-attend reference on the same cache.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
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

// Plain-C++ reference: dequantize the packed K/V planes for one slot and run
// softmax(q*k^T) * v over the concatenated [sink | pages | qbuf] axis.
static void reference_attention(const std::vector<float> & q, int hd, int nh, int nq,
                                const std::vector<ggml_fp16_t> & kstage, const std::vector<ggml_fp16_t> & vstage,
                                const std::vector<uint8_t> & kpl, const std::vector<uint8_t> & vpl,
                                int n_sink, int n_kpending, int n_kpages, int n_vp, int n_vqbuf,
                                int stage_rows, float scale, std::vector<float> & out) {
    (void) stage_rows;
    const int n_kv = n_sink + n_kpages * P + n_kpending;
    std::vector<float> kd((size_t) hd * nh * n_kv), vd((size_t) hd * nh * n_kv);
    for (int h = 0; h < nh; ++h) {
        // sink (rows [0, n_sink), key == row) and K pending (rows [n_sink, n_sink+n_kpending),
        // whose keys sit AFTER the packed pages: key = n_sink + n_kpages*P + i).
        for (int key = 0; key < n_sink; ++key)
            for (int d = 0; d < hd; ++d)
                kd[((size_t) h * n_kv + key) * hd + d] = ggml_fp16_to_fp32(kstage[(size_t) key * nh * hd + h * hd + d]);
        for (int i = 0; i < n_kpending; ++i) {
            const int key = n_sink + n_kpages * P + i;
            for (int d = 0; d < hd; ++d)
                kd[((size_t) h * n_kv + key) * hd + d] =
                    ggml_fp16_to_fp32(kstage[(size_t) (n_sink + i) * nh * hd + h * hd + d]);
        }
        for (int key = 0; key < n_sink; ++key)
            for (int d = 0; d < hd; ++d)
                vd[((size_t) h * n_kv + key) * hd + d] = ggml_fp16_to_fp32(vstage[(size_t) key * nh * hd + h * hd + d]);
        for (int i = 0; i < n_vqbuf; ++i) {
            const int slot = (n_vp % P + i) % P;
            for (int d = 0; d < hd; ++d)
                vd[((size_t) h * n_kv + n_sink + n_vp + i) * hd + d] =
                    ggml_fp16_to_fp32(vstage[(size_t) (n_sink + slot) * nh * hd + h * hd + d]);
        }
        // packed K pages
        for (int pg = 0; pg < n_kpages; ++pg) {
            const uint8_t * page = kpl.data() + (size_t) pg * k_page_bytes(nh);
            const uint8_t * head = page + (size_t) h * K_HEAD_BYTES;
            for (int t = 0; t < P; ++t) for (int d = 0; d < hd; ++d) {
                const int low = (head[d * (P/4) + t/4] >> ((t&3)*2)) & 3;
                const int idx = head[K_OFF_IDX + d];
                int code = low;
                if (idx < D_BOOST) code |= ((head[K_OFF_HIGH + (size_t) idx * (P/4) + t/4] >> ((t&3)*2)) & 3) << 2;
                float s, m; ggml_fp16_t sh, mh;
                memcpy(&sh, head + K_OFF_META + d*4, 2); memcpy(&mh, head + K_OFF_META + d*4 + 2, 2);
                s = ggml_fp16_to_fp32(sh); m = ggml_fp16_to_fp32(mh);
                kd[((size_t) h * n_kv + n_sink + pg*P + t) * hd + d] = (float) code * s + m;
            }
        }
        // packed V pages
        for (int pg = 0; pg < (n_vp + P - 1) / P; ++pg) {
            const uint8_t * page = vpl.data() + (size_t) pg * v_page_bytes(nh);
            const uint8_t * head = page + (size_t) h * V_HEAD_BYTES;
            const int t_hi = std::min(P, n_vp - pg * P);
            for (int t = 0; t < t_hi; ++t) {
                ggml_fp16_t sh, mh; memcpy(&sh, head + V_OFF_META + t*4, 2); memcpy(&mh, head + V_OFF_META + t*4 + 2, 2);
                const float s = ggml_fp16_to_fp32(sh), m = ggml_fp16_to_fp32(mh);
                for (int d = 0; d < hd; ++d) {
                    const int qq = (head[(size_t) t*(hd/4) + d/4] >> ((d&3)*2)) & 3;
                    vd[((size_t) h * n_kv + n_sink + pg*P + t) * hd + d] = (float) qq * s + m;
                }
            }
        }
        // attention per query
        for (int qi = 0; qi < nq; ++qi) {
            const float * qh = q.data() + ((size_t) qi * nh + h) * hd;
            std::vector<float> logit(n_kv);
            float mx = -INFINITY;
            for (int key = 0; key < n_kv; ++key) {
                float acc = 0.0f;
                for (int d = 0; d < hd; ++d) acc += qh[d] * kd[((size_t) h * n_kv + key) * hd + d];
                logit[key] = acc * scale; mx = std::fmax(mx, logit[key]);
            }
            float sum = 0.0f;
            for (int key = 0; key < n_kv; ++key) { logit[key] = std::exp(logit[key] - mx); sum += logit[key]; }
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int key = 0; key < n_kv; ++key) acc += logit[key] / sum * vd[((size_t) h * n_kv + key) * hd + d];
                out[((size_t) qi * nh + h) * hd + d] = acc;
            }
        }
    }
}

// Build and run the graph on `backend`; returns the [hd, nh, nq] output.
static bool run_ops(ggml_backend_t backend, ouro_kv_cache & kv, int slot,
                    const std::vector<float> & q, int hd, int nh, int nq,
                    int n_sink, int n_kpending, int n_kpages, int n_vp, int n_vqbuf,
                    float scale, std::vector<float> & out) {
    std::vector<uint8_t> buf(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params ip = { buf.size(), buf.data(), /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nh, nq, 1);
    ggml_set_input(qt);
    ggml_tensor * kf = ouro_kv_kitty_f16(kv, ctx, slot, n_sink + n_kpending, false);
    ggml_tensor * vf = ouro_kv_kitty_f16(kv, ctx, slot, n_sink + 128, true);
    ggml_tensor * kp = ouro_kv_kitty_planes(kv, slot, false);
    ggml_tensor * vp = ouro_kv_kitty_planes(kv, slot, true);

    const int n_kv = n_sink + n_kpages * P + n_kpending;
    ggml_tensor * logits = ggml_kitty_qk(ctx, qt, kf, kp, n_sink, n_kpending, n_kpages, scale);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, nq);
    ggml_set_input(mask);
    ggml_tensor * probs = ggml_soft_max_ext(ctx, logits, mask, 1.0f, 0.0f);
    ggml_tensor * sv = ggml_kitty_sv(ctx, probs, vf, vp, n_sink, n_vp, n_vqbuf);
    ggml_build_forward_expand(gf, sv);

    if (!ggml_backend_alloc_ctx_tensors(ctx, backend)) {
        ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(qt, q.data(), 0, q.size() * sizeof(float));
    std::vector<ggml_fp16_t> mdata((size_t) n_kv * nq, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(mask, mdata.data(), 0, mdata.size() * sizeof(ggml_fp16_t));

    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (ok) { out.resize((size_t) hd * nh * nq); ggml_backend_tensor_get(sv, out.data(), 0, out.size() * sizeof(float)); }
    ggml_free(ctx);
    return ok;
}

int main() {
    const int hd = 128, nh = 16, nq = 3, sink = 32, P_ = P;   // nq > 1 exercises the (head,query) indexing
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { printf("FAIL cpu\n"); return 1; }

    ouro_kv_cache kv; std::string err;
    ouro_kitty_config c; c.sink = sink; c.page = P_; c.packed = true;
    // two packed pages -> ctx must cover sink + 3 pages
    if (!ouro_kv_cache_init(kv, cpu, hd, nh, sink + 3 * P_, /*n_slots=*/1, GGML_TYPE_F16, &c, err)) {
        printf("FAIL init: %s\n", err.c_str()); return 1;
    }

    // Deterministic staging: fill all [0, sink+P) rows of both K and V, then pack one
    // K page. V has no packed page yet and a full ring -- the state right after the
    // first K page completes -- so n_kv = sink + P, n_vp = 0, n_vqbuf = P.
    std::vector<ggml_fp16_t> stage((size_t) (sink + P_) * hd * nh);
    for (size_t i = 0; i < stage.size(); ++i) stage[i] = ggml_fp32_to_fp16(0.05f * std::sin((float) i * 0.37f));
    {
        ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, sink + P_);
        memcpy(src->data, stage.data(), stage.size() * sizeof(ggml_fp16_t));
        ggml_cgraph * wg = ggml_new_graph(ctx);
        ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_kitty_write(kv, ctx, 0, 0, sink + P_, false)));
        ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_kitty_write(kv, ctx, 0, 0, sink + P_, true)));
        ggml_backend_graph_compute(cpu, wg);
        ggml_free(ctx);
        if (!ouro_kv_kitty_pack_pages(kv, sink + P_, err)) { printf("FAIL pack: %s\n", err.c_str()); return 1; }
    }

    const int n_sink = sink, n_kpages = 1, n_kpending = 0, n_vp = 0, n_vqbuf = P_;
    const float scale = 1.0f / std::sqrt((float) hd);
    std::vector<float> q((size_t) hd * nh * nq);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.02f * std::cos((float) i * 0.11f);

    std::vector<float> got;
    if (!run_ops(cpu, kv, 0, q, hd, nh, nq, n_sink, n_kpending, n_kpages, n_vp, n_vqbuf, scale, got)) {
        printf("FAIL cpu graph compute\n"); return 1;
    }

    // reference needs the current staging rows [0, sink+P) too; read them back
    std::vector<ggml_fp16_t> kstage((size_t) (sink + P_) * hd * nh), vstage(kstage.size());
    ggml_backend_tensor_get(kv.k[0], kstage.data(), 0, kstage.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(kv.v[0], vstage.data(), 0, vstage.size() * sizeof(ggml_fp16_t));
    std::vector<uint8_t> kpl((size_t) n_kpages * k_page_bytes(nh)), vpl((size_t) n_kpages * v_page_bytes(nh));
    ggml_backend_tensor_get(kv.k_planes[0], kpl.data(), 0, kpl.size());
    ggml_backend_tensor_get(kv.v_planes[0], vpl.data(), 0, vpl.size());

    std::vector<float> ref((size_t) hd * nh * nq);
    reference_attention(q, hd, nh, nq, kstage, vstage, kpl, vpl,
                        n_sink, n_kpending, n_kpages, n_vp, n_vqbuf, sink + P_, scale, ref);

    double maxd = 0.0;
    for (size_t i = 0; i < got.size(); ++i) maxd = std::fmax(maxd, std::fabs((double) got[i] - ref[i]));
    if (maxd > 5e-3) { printf("FAIL ops vs reference: max|d|=%.4g\n", maxd); g_fail = 1; }
    else             { printf("ok   ops vs reference (d=%.3g)\n", maxd); }

    // Vulkan-vs-reference on the SAME (base) cache contents.
    // The device here is an integrated GPU, so query IGPU as well as GPU.
    ggml_backend_t vk = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!vk) vk = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (vk) {
        std::vector<float> vgot;
        ouro_kv_cache kv2; ouro_kitty_config c2 = c;
        if (ouro_kv_cache_init(kv2, vk, hd, nh, sink + 3 * P_, 1, GGML_TYPE_F16, &c2, err)) {
            // copy the CPU cache contents into the Vulkan cache
            std::vector<uint8_t> rawk(ggml_nbytes(kv.k[0])), rawv(ggml_nbytes(kv.v[0]));
            std::vector<uint8_t> rk((size_t) n_kpages * k_page_bytes(nh)), rv((size_t) n_kpages * v_page_bytes(nh));
            ggml_backend_tensor_get(kv.k[0], rawk.data(), 0, rawk.size());
            ggml_backend_tensor_get(kv.v[0], rawv.data(), 0, rawv.size());
            ggml_backend_tensor_get(kv.k_planes[0], rk.data(), 0, rk.size());
            ggml_backend_tensor_get(kv.v_planes[0], rv.data(), 0, rv.size());
            ggml_backend_tensor_set(kv2.k[0], rawk.data(), 0, rawk.size());
            ggml_backend_tensor_set(kv2.v[0], rawv.data(), 0, rawv.size());
            ggml_backend_tensor_set(kv2.k_planes[0], rk.data(), 0, rk.size());
            ggml_backend_tensor_set(kv2.v_planes[0], rv.data(), 0, rv.size());
            if (run_ops(vk, kv2, 0, q, hd, nh, nq, n_sink, n_kpending, n_kpages, n_vp, n_vqbuf, scale, vgot)) {
                double vd = 0.0;
                for (size_t i = 0; i < vgot.size(); ++i) vd = std::fmax(vd, std::fabs((double) vgot[i] - ref[i]));
                if (vd > 5e-3) { printf("FAIL vulkan vs reference: max|d|=%.4g\n", vd); g_fail = 1; }
                else           { printf("ok   vulkan vs reference (d=%.3g)\n", vd); }
            }
        }
        ouro_kv_cache_free(kv2);
        ggml_backend_free(vk);
    } else {
        printf("skip vulkan (no GPU backend available)\n");
    }

    // MIXED-STATE CASE (added after review): n_kpages >= 1 AND n_kpending > 0.
    // This is the state chunked prefill and every post-page decode step are in, and it is
    // the ONLY state that exercises the QK pending branch — the case above uses
    // n_kpending = 0, which is exactly why the key->row mapping bug hid there.
    {
        const int nk = 8;
        // The K pending staging is contiguous at [sink, sink+P) and RESETS when a page
        // packs, so the next tokens land at row = sink (see Task 10's stage_row formula).
        std::vector<ggml_fp16_t> extra((size_t) nk * hd * nh);
        for (size_t i = 0; i < extra.size(); ++i)
            extra[i] = ggml_fp32_to_fp16(0.05f * std::sin((float) i * 0.07f));
        {
            ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hd * nh, nk);
            memcpy(src->data, extra.data(), extra.size() * sizeof(ggml_fp16_t));
            ggml_cgraph * wg = ggml_new_graph(ctx);
            ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_kitty_write(kv, ctx, 0, sink, nk, false)));
            ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_kitty_write(kv, ctx, 0, sink, nk, true)));
            ggml_backend_graph_compute(cpu, wg);
            ggml_free(ctx);
        }

        std::vector<float> got2;
        // n_kv = n_sink + n_kpages*P + nk = 168, so SV needs n_sink + n_vp + n_vqbuf == 168:
        // with the V window at n_vqbuf = P = 128, V has packed the nk=8 positions that left
        // its window, i.e. n_vp = nk. (The plan passed n_vp = 0, which fails the SV assert.)
        const int n_vp_mixed = nk;
        if (!run_ops(cpu, kv, 0, q, hd, nh, nq, n_sink, nk, n_kpages, n_vp_mixed, n_vqbuf, scale, got2)) {
            printf("FAIL cpu graph compute (mixed state)\n"); return 1;
        }
        ggml_backend_tensor_get(kv.k[0], kstage.data(), 0, kstage.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(kv.v[0], vstage.data(), 0, vstage.size() * sizeof(ggml_fp16_t));
        std::vector<float> ref2((size_t) hd * nh * nq);
        reference_attention(q, hd, nh, nq, kstage, vstage, kpl, vpl,
                            n_sink, nk, n_kpages, n_vp_mixed, n_vqbuf, sink + P_, scale, ref2);
        double maxd2 = 0.0;
        for (size_t i = 0; i < got2.size(); ++i) maxd2 = std::fmax(maxd2, std::fabs((double) got2[i] - ref2[i]));
        if (maxd2 > 5e-3) { printf("FAIL ops vs reference (mixed state): max|d|=%.4g\n", maxd2); g_fail = 1; }
        else              { printf("ok   ops vs reference (mixed state n_kpages=1 n_kpending=%d, d=%.3g)\n", nk, maxd2); }
    }

    ouro_kv_cache_free(kv);
    ggml_backend_free(cpu);
    printf(g_fail ? "KITTY_OPS: FAIL\n" : "KITTY_OPS: PASS\n");
    return g_fail;
}
