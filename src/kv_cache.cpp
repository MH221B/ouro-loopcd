#include "kv_cache.h"
#include "ggml-alloc.h"

#include <cstdio>

// Per-head view of ONE slot tensor: [head_dim, n_head_kv, n_kv, 1].
// nb1 = stride between heads (they are contiguous inside a row);
// nb2 = stride between positions (one full row per position).
static ggml_tensor * kv_read_view(ggml_context * ctx, ggml_tensor * slot_tensor, ggml_type type,
                                  int head_dim, int n_embd_kv, int n_kv) {
    const size_t rs_head = ggml_row_size(type, head_dim);
    const size_t rs_row  = ggml_row_size(type, n_embd_kv);
    return ggml_view_4d(ctx, slot_tensor, head_dim, n_embd_kv / head_dim, n_kv, 1,
                        rs_head, rs_row, rs_row, 0);
}

// Flat write view of ONE slot tensor, positions [n_past, n_past + n_tokens):
// [n_embd_kv, n_tokens]. Contiguous.
static ggml_tensor * kv_write_view(ggml_context * ctx, ggml_tensor * slot_tensor, ggml_type type,
                                   int n_embd_kv, int n_past, int n_tokens) {
    const size_t rs_row = ggml_row_size(type, n_embd_kv);
    return ggml_view_2d(ctx, slot_tensor, n_embd_kv, n_tokens, rs_row, (size_t) n_past * rs_row);
}

bool ouro_kv_cache_init(ouro_kv_cache & kv, ggml_backend_t backend,
                        int head_dim, int n_head_kv, int n_ctx, int n_slots,
                        ggml_type type, const ouro_kitty_config * kitty, std::string & err) {
    if (kitty) {
        if (type != GGML_TYPE_F16) { err = "kitty kv requires --kv-type kitty (f16 storage)"; return false; }
        kv.kitty = true;
        kv.kitty_cfg = *kitty;
        if (kv.kitty_cfg.boost_channels <= 0) kv.kitty_cfg.boost_channels = head_dim / 4;
        if (kv.kitty_cfg.sink < 0) kv.kitty_cfg.sink = 0;
        if (kv.kitty_cfg.page <= 0) kv.kitty_cfg.page = 128;
        kv.kitty_k_done = kv.kitty_cfg.sink;
        kv.kitty_v_done = kv.kitty_cfg.sink;
    } else if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_F16 && type != GGML_TYPE_F32) {
        err = "kv cache type must be q8_0, f16 or f32 (or --kv-type kitty)";
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

    // M7a: 2 * n_slots tensors, each [n_embd_kv, n_ctx]. The allocator groups
    // them into buffers at suballocation_block_size (1 GiB by default), so no
    // single buffer approaches the Vulkan max_buffer_size limit.
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t)(2 * n_slots + 8) + 4096,
                            nullptr, /*no_alloc=*/ true };
    kv.ctx = ggml_init(ip);
    if (!kv.ctx) { err = "kv cache: ggml_init failed"; return false; }

    kv.k.resize(n_slots);
    kv.v.resize(n_slots);
    for (int s = 0; s < n_slots; ++s) {
        kv.k[s] = ggml_new_tensor_2d(kv.ctx, type, kv.n_embd_kv, n_ctx);
        kv.v[s] = ggml_new_tensor_2d(kv.ctx, type, kv.n_embd_kv, n_ctx);
        char name[32];
        std::snprintf(name, sizeof(name), "kv_cache.k[%d]", s); ggml_set_name(kv.k[s], name);
        std::snprintf(name, sizeof(name), "kv_cache.v[%d]", s); ggml_set_name(kv.v[s], name);
    }

    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf) { err = "kv cache: backend allocation failed"; ouro_kv_cache_free(kv); return false; }

    ggml_backend_buffer_clear(kv.buf, 0);
    return true;
}

void ouro_kv_cache_free(ouro_kv_cache & kv) {
    if (kv.buf) { ggml_backend_buffer_free(kv.buf); kv.buf = nullptr; }
    if (kv.ctx) { ggml_free(kv.ctx);                 kv.ctx = nullptr; }
    kv.k.clear();
    kv.v.clear();
}

void ouro_kv_cache_clear(ouro_kv_cache & kv) {
    if (kv.buf) ggml_backend_buffer_clear(kv.buf, 0);
    if (kv.kitty) {
        kv.kitty_k_done = kv.kitty_cfg.sink;
        kv.kitty_v_done = kv.kitty_cfg.sink;
    }
}

ggml_tensor * ouro_kv_k(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_kv) {
    return kv_read_view(ctx, kv.k[slot], kv.type, kv.head_dim, kv.n_embd_kv, n_kv);
}

ggml_tensor * ouro_kv_v(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_kv) {
    return kv_read_view(ctx, kv.v[slot], kv.type, kv.head_dim, kv.n_embd_kv, n_kv);
}

ggml_tensor * ouro_kv_k_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_past, int n_tokens) {
    return kv_write_view(ctx, kv.k[slot], kv.type, kv.n_embd_kv, n_past, n_tokens);
}

ggml_tensor * ouro_kv_v_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_past, int n_tokens) {
    return kv_write_view(ctx, kv.v[slot], kv.type, kv.n_embd_kv, n_past, n_tokens);
}
