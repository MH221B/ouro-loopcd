// tests/unit/test_kitty_pack.cpp — M7b: the C++ packed emitter must reproduce the
// byte layout and the dequantized values of tests/reference/kitty_pack_fixture.py.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "kv_cache.h"
#include "kitty_kv.h"
#include "cache_layout.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

using namespace kitty_layout;
static int g_fail = 0;

int main() {
    FILE * fp = fopen("tests/unit/fixtures/kitty_pack.bin", "rb");
    if (!fp) { printf("FAIL: fixture missing (run tests/reference/kitty_pack_fixture.py)\n"); return 1; }
    int32_t hd[6]; if (fread(hd, sizeof(int32_t), 6, fp) != 6) return 1;
    const int Hh = hd[0], Dd = hd[1], Pp = hd[2], KB = hd[3], BB = hd[4], DB = hd[5];
    const int E = Hh * Dd;
    std::vector<ggml_fp16_t> in((size_t) E * Pp);
    std::vector<uint8_t> want_page((size_t) Hh * (Dd * (Pp/4) + DB*(Pp/4) + Dd + Dd*4));
    std::vector<ggml_fp16_t> want_deq((size_t) E * Pp);
    if (fread(in.data(), sizeof(ggml_fp16_t), in.size(), fp) != in.size()) return 1;
    if (fread(want_page.data(), 1, want_page.size(), fp) != want_page.size()) return 1;
    if (fread(want_deq.data(), sizeof(ggml_fp16_t), want_deq.size(), fp) != want_deq.size()) return 1;
    fclose(fp);

    // The fixture uses the real fixed layout (16 heads, D=128, P=128), so the emitter
    // is driven through its production path with the production geometry.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ouro_kv_cache kv; std::string err;
    ouro_kitty_config c; c.sink = 0; c.page = Pp; c.boost_channels = DB;
    c.kbits = KB; c.boost_bits = BB; c.vbits = 16; c.packed = true;
    const bool ok = ouro_kv_cache_init(kv, cpu, Dd, Hh, /*n_ctx=*/Pp, /*n_slots=*/1,
                                       GGML_TYPE_F16, &c, err);
    printf(ok ? "ok   init\n" : "FAIL init: %s\n", err.c_str());
    if (!ok) { ggml_backend_free(cpu); return 1; }

    {
        ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, E, Pp);
        memcpy(src->data, in.data(), in.size() * sizeof(ggml_fp16_t));
        ggml_cgraph * wg = ggml_new_graph(ctx);
        ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_kitty_write(kv, ctx, 0, 0, Pp, false)));
        ggml_backend_graph_compute(cpu, wg);
        ggml_free(ctx);
    }

    // pack page 0
    if (!ouro_kv_kitty_pack_pages(kv, Pp, err)) { printf("FAIL pack: %s\n", err.c_str()); return 1; }

    std::vector<uint8_t> got_page(want_page.size());
    ggml_backend_tensor_get(kv.k_planes[0], got_page.data(), 0, got_page.size());

    size_t first_bad = (size_t) -1;
    for (size_t i = 0; i < got_page.size(); ++i)
        if (got_page[i] != want_page[i]) { first_bad = i; break; }
    if (first_bad != (size_t) -1) { printf("FAIL packed bytes differ at %zu\n", first_bad); g_fail = 1; }
    else                          { printf("ok   packed bytes match fixture\n"); }

    // dequantize got_page with the C++ reference and compare
    double maxd = 0.0;
    for (int h = 0; h < Hh; ++h) {
        for (int d = 0; d < Dd; ++d) {
            const size_t hb = (size_t) h * (Dd*(Pp/4) + DB*(Pp/4) + Dd + Dd*4);
            ggml_fp16_t sh, mh;
            memcpy(&sh, got_page.data() + hb + Dd*(Pp/4) + DB*(Pp/4) + Dd + d*4,     2);
            memcpy(&mh, got_page.data() + hb + Dd*(Pp/4) + DB*(Pp/4) + Dd + d*4 + 2, 2);
            const float scale = ggml_fp16_to_fp32(sh), mn = ggml_fp16_to_fp32(mh);
            const uint8_t s = got_page[hb + Dd*(Pp/4) + DB*(Pp/4) + d];
            for (int t = 0; t < Pp; ++t) {
                int lo = (got_page[hb + (size_t) d*(Pp/4) + t/4] >> ((t&3)*2)) & 3;
                int hi = 0;
                if (s < DB) hi = (got_page[hb + Dd*(Pp/4) + (size_t) s*(Pp/4) + t/4] >> ((t&3)*2)) & 3;
                const float got = (float)(lo | (hi<<2)) * scale + mn;
                const float want = ggml_fp16_to_fp32(want_deq[(size_t) t*E + h*Dd + d]);
                maxd = std::fmax(maxd, std::fabs(got - want));
            }
        }
    }
    if (maxd > 2e-3) { printf("FAIL dequant vs fixture: max|d|=%.4g\n", maxd); g_fail = 1; }
    else             { printf("ok   dequant vs fixture (d=%.3g)\n", maxd); }

    // M7b Both-modes requirement (plan decisions #8/#9): the f16 fake-quant path and the
    // packed emitter must select the SAME promoted channels. Run the f16 path on a second
    // (non-packed) cache holding the identical page and compare its boosted set to the
    // packed idx plane. The f16 path stores only the quantized values, so a channel's
    // resolution reveals its bit width: a 2-bit channel has <= 2^kbits distinct values,
    // a 4-bit channel more than that.
    {
        ouro_kv_cache kvf; ouro_kitty_config cf = c; cf.packed = false;
        std::string ferr;
        if (!ouro_kv_cache_init(kvf, cpu, Dd, Hh, Pp, 1, GGML_TYPE_F16, &cf, ferr)) {
            printf("FAIL both-modes: f16 init: %s\n", ferr.c_str()); g_fail = 1;
        } else {
            {
                ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, E, Pp);
                memcpy(src->data, in.data(), in.size() * sizeof(ggml_fp16_t));
                ggml_cgraph * wg = ggml_new_graph(ctx);
                ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_k_write(kvf, ctx, 0, 0, Pp)));
                ggml_backend_graph_compute(cpu, wg);
                ggml_free(ctx);
            }
            if (!ouro_kv_kitty_quantize(kvf, Pp, ferr)) { printf("FAIL both-modes quantize: %s\n", ferr.c_str()); g_fail = 1; }
            else {
                std::vector<ggml_fp16_t> f16out((size_t) E * Pp);
                ggml_backend_tensor_get(kvf.k[0], f16out.data(), 0, f16out.size() * sizeof(ggml_fp16_t));
                int sel_diff = 0;
                for (int h = 0; h < Hh; ++h) {
                    for (int d = 0; d < Dd; ++d) {
                        std::vector<float> vals(Pp);
                        for (int t = 0; t < Pp; ++t)
                            vals[t] = ggml_fp16_to_fp32(f16out[(size_t) t*E + h*Dd + d]);
                        std::sort(vals.begin(), vals.end());
                        const int distinct = (int) (std::unique(vals.begin(), vals.end()) - vals.begin());
                        const bool f16_boosted = distinct > (1 << KB);   // 2-bit => <= 4 distinct
                        const size_t hb = (size_t) h * (Dd*(Pp/4) + DB*(Pp/4) + Dd + Dd*4);
                        const bool packed_boosted = got_page[hb + Dd*(Pp/4) + DB*(Pp/4) + d] < (uint8_t) DB;
                        if (f16_boosted != packed_boosted) ++sel_diff;
                    }
                }
                if (sel_diff) { printf("FAIL both-modes: %d channels differ in promotion\n", sel_diff); g_fail = 1; }
                else          { printf("ok   both-modes: f16 and packed select identical channels\n"); }
            }
            ouro_kv_cache_free(kvf);
        }
    }

    ouro_kv_cache_free(kv);
    ggml_backend_free(cpu);
    printf(g_fail ? "KITTY_PACK: FAIL\n" : "KITTY_PACK: PASS\n");
    return g_fail;
}
