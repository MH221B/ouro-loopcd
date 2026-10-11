// tests/unit/test_kitty.cpp — M6: the C++ Kitty page quantizer must match the Python
// oracle's fake_quant_page (tests/unit/fixtures/kitty_page.bin).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "kv_cache.h"
#include "kitty_kv.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

int main() {
    FILE * fp = fopen("tests/unit/fixtures/kitty_page.bin", "rb");
    if (!fp) { printf("FAIL: fixture missing (run tests/reference/kitty_fixture.py)\n"); return 1; }
    int32_t hd[6]; size_t nr = fread(hd, sizeof(int32_t), 6, fp); (void) nr;
    const int Hh = hd[0], Dd = hd[1], Pp = hd[2], KB = hd[3], BB = hd[4], DB = hd[5];
    const int E = Hh * Dd;
    std::vector<ggml_fp16_t> in((size_t) E * Pp), want((size_t) E * Pp);
    nr = fread(in.data(),   sizeof(ggml_fp16_t), in.size(), fp);   (void) nr;
    nr = fread(want.data(), sizeof(ggml_fp16_t), want.size(), fp); (void) nr;
    fclose(fp);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ouro_kv_cache kv; std::string err;
    ouro_kitty_config c; c.sink = 0; c.page = Pp; c.boost_channels = DB; c.kbits = KB; c.boost_bits = BB; c.vbits = 16;
    const bool ok = ouro_kv_cache_init(kv, cpu, Dd, Hh, /*n_ctx=*/Pp, /*n_slots=*/1, GGML_TYPE_F16, &c, err);
    printf(ok ? "ok   init\n" : "FAIL init: %s\n", err.c_str());
    if (!ok) { ggml_backend_free(cpu); return 1; }

    // write the page through the same cpy path the graph uses: [E, Pp] f16
    {
        ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, E, Pp);
        memcpy(src->data, in.data(), in.size() * sizeof(ggml_fp16_t));
        ggml_cgraph * wg = ggml_new_graph(ctx);
        ggml_build_forward_expand(wg, ggml_cpy(ctx, src, ouro_kv_k_write(kv, ctx, 0, 0, Pp)));
        ggml_backend_graph_compute(cpu, wg);
        ggml_free(ctx);
    }

    if (!ouro_kv_kitty_quantize(kv, Pp, err)) { printf("FAIL quantize: %s\n", err.c_str()); g_fail = 1; }

    std::vector<ggml_fp16_t> got((size_t) E * Pp);
    ggml_backend_tensor_get(kv.k[0], got.data(), 0, got.size() * sizeof(ggml_fp16_t));

    double maxd = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        maxd = std::max(maxd, std::fabs((double) ggml_fp16_to_fp32(got[i]) - (double) ggml_fp16_to_fp32(want[i])));
    if (maxd > 2e-3) { printf("FAIL kitty page vs oracle: max|d|=%.4g\n", maxd); g_fail = 1; }
    else             { printf("ok   kitty page vs oracle (d=%.3g)\n", maxd); }

    ouro_kv_cache_free(kv);
    ggml_backend_free(cpu);
    printf(g_fail ? "KITTY: FAIL\n" : "KITTY: PASS\n");
    return g_fail;
}
