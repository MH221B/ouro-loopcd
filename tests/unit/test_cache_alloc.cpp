// M7a: the KV cache is one tensor per slot, so ggml's allocator can group slots
// into buffers smaller than the Vulkan per-allocation limit. This checks the
// structure; the actual limit is verified by the Vulkan smoke runs in the plan.
#include "ggml.h"
#include "ggml-cpu.h"        // declares ggml_backend_cpu_init (see tests/unit/test_kv.cpp:7)
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "kv_cache.h"

#include <cstdio>
#include <string>

static int g_fail = 0;
static void check(bool ok, const char * name) {
    if (ok) printf("ok   %s\n", name);
    else    { printf("FAIL %s\n", name); g_fail = 1; }
}

int main() {
    const int head_dim = 128, n_head_kv = 16;
    const int n_embd_kv = head_dim * n_head_kv;   // 2048
    const int n_ctx = 512, n_slots = 4;           // small: structure only

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { printf("FAIL cpu backend\n"); return 1; }

    ouro_kv_cache kv;
    std::string err;
    if (!ouro_kv_cache_init(kv, cpu, head_dim, n_head_kv, n_ctx, n_slots,
                            GGML_TYPE_F16, nullptr, err)) {
        printf("FAIL init: %s\n", err.c_str());
        return 1;
    }

    check((int) kv.k.size() == n_slots && (int) kv.v.size() == n_slots,
          "one tensor per slot");
    check(kv.k[0] != nullptr && kv.k[n_slots - 1] != nullptr &&
          kv.v[0] != nullptr && kv.v[n_slots - 1] != nullptr,
          "every slot tensor is allocated");

    const size_t expect = (size_t) n_embd_kv * n_ctx * sizeof(ggml_fp16_t);
    check(ggml_nbytes(kv.k[0]) == expect, "K per-slot byte size");
    check(ggml_nbytes(kv.v[n_slots - 1]) == expect, "V per-slot byte size");

    // The read/write views must be valid on a per-slot tensor.
    std::vector<uint8_t> ctxbuf(ggml_tensor_overhead() * 8 + 4096);
    ggml_init_params ip = { ctxbuf.size(), ctxbuf.data(), /*no_alloc=*/ true };
    ggml_context * vctx = ggml_init(ip);
    ggml_tensor * r = ouro_kv_k(kv, vctx, 1, n_ctx);
    check(r->ne[0] == head_dim && r->ne[1] == n_head_kv && r->ne[2] == n_ctx,
          "read view shape [head_dim, n_head_kv, n_kv]");
    ggml_tensor * w = ouro_kv_k_write(kv, vctx, 3, 16, 8);
    check(w->ne[0] == n_embd_kv && w->ne[1] == 8, "write view shape [n_embd_kv, n_tokens]");
    ggml_free(vctx);

    ouro_kv_cache_free(kv);
    ggml_backend_free(cpu);
    printf(g_fail ? "\nFAILED\n" : "\nPASSED\n");
    return g_fail;
}
