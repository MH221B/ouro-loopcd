#include "kv_cache.h"
#include "ggml-alloc.h"

#include <cstdio>

static ggml_tensor * kv_read_view(ggml_context * ctx, ggml_tensor * cache, ggml_type type,
                                  int head_dim, int n_embd_kv, int n_ctx,
                                  int slot, int n_kv) {
    const size_t rs_head = ggml_row_size(type, head_dim);
    const size_t rs_row  = ggml_row_size(type, n_embd_kv);
    const size_t rs_slot = rs_row * (size_t) n_ctx;
    return ggml_view_4d(ctx, cache, head_dim, n_embd_kv / head_dim, n_kv, 1,
                        rs_head, rs_row, rs_slot, rs_slot * (size_t) slot);
}

static ggml_tensor * kv_write_view(ggml_context * ctx, ggml_tensor * cache, ggml_type type,
                                   int n_embd_kv, int n_ctx, int slot, int n_past, int n_tokens) {
    const size_t rs_row = ggml_row_size(type, n_embd_kv);
    const size_t off    = ((size_t) slot * n_ctx + n_past) * rs_row;
    return ggml_view_2d(ctx, cache, n_embd_kv, n_tokens, rs_row, off);
}

bool ouro_kv_cache_init(ouro_kv_cache & kv, ggml_backend_t backend,
                        int head_dim, int n_head_kv, int n_ctx, int n_slots,
                        ggml_type type, std::string & err) {
    if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_F16) {
        err = "kv cache type must be q8_0 or f16";
        return false;
    }
    kv.head_dim  = head_dim;
    kv.n_head_kv = n_head_kv;
    kv.n_embd_kv = head_dim * n_head_kv;
    kv.n_ctx     = n_ctx;
    kv.n_slots   = n_slots;
    kv.type      = type;

    if (kv.n_embd_kv % ggml_blck_size(type) != 0) {
        err = "kv cache row length is not a multiple of the block size";
        return false;
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 4 + 4096, nullptr, /*no_alloc=*/ true };
    kv.ctx = ggml_init(ip);
    if (!kv.ctx) { err = "kv cache: ggml_init failed"; return false; }

    kv.k = ggml_new_tensor_3d(kv.ctx, type, kv.n_embd_kv, n_ctx, n_slots);
    kv.v = ggml_new_tensor_3d(kv.ctx, type, kv.n_embd_kv, n_ctx, n_slots);
    ggml_set_name(kv.k, "kv_cache.k");
    ggml_set_name(kv.v, "kv_cache.v");

    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf) { err = "kv cache: backend allocation failed"; ouro_kv_cache_free(kv); return false; }

    ggml_backend_buffer_clear(kv.buf, 0);
    return true;
}

void ouro_kv_cache_free(ouro_kv_cache & kv) {
    if (kv.buf) { ggml_backend_buffer_free(kv.buf); kv.buf = nullptr; }
    if (kv.ctx) { ggml_free(kv.ctx);                 kv.ctx = nullptr; }
    kv.k = nullptr;
    kv.v = nullptr;
}

void ouro_kv_cache_clear(ouro_kv_cache & kv) {
    if (kv.buf) ggml_backend_buffer_clear(kv.buf, 0);
}

ggml_tensor * ouro_kv_k(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_kv) {
    return kv_read_view(ctx, kv.k, kv.type, kv.head_dim, kv.n_embd_kv, kv.n_ctx, slot, n_kv);
}

ggml_tensor * ouro_kv_v(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_kv) {
    return kv_read_view(ctx, kv.v, kv.type, kv.head_dim, kv.n_embd_kv, kv.n_ctx, slot, n_kv);
}

ggml_tensor * ouro_kv_k_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_past, int n_tokens) {
    return kv_write_view(ctx, kv.k, kv.type, kv.n_embd_kv, kv.n_ctx, slot, n_past, n_tokens);
}

ggml_tensor * ouro_kv_v_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_past, int n_tokens) {
    return kv_write_view(ctx, kv.v, kv.type, kv.n_embd_kv, kv.n_ctx, slot, n_past, n_tokens);
}
