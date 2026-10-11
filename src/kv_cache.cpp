#include "kv_cache.h"
#include "ggml-alloc.h"
#include "cache_layout.h"

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
        kv.kitty_rng.seed(kv.kitty_cfg.seed);
        kv.kitty_k_pages = 0;
        kv.kitty_v_packed = 0;
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

    // M7a: one tensor per slot so the allocator can split the cache into buffers
    // smaller than the Vulkan per-allocation limit. For plain/q8_0/f16 each slot is
    // [n_embd_kv, n_ctx]; for kitty-packed the per-slot tensors are the small f16
    // staging regions [n_embd_kv, sink + P] (spec §5.1) plus the packed I8 blobs.
    const bool packed = kv.kitty && kv.kitty_cfg.packed;
    const int  sink   = kv.kitty_cfg.sink;
    int        n_pages_max = 0;
    size_t     k_plane_bytes = 0, v_plane_bytes = 0;
    int        stage_rows = 0;
    if (packed) {
        if (type != GGML_TYPE_F16) { err = "kitty packed requires f16 staging"; return false; }
        if (kv.kitty_cfg.page != kitty_layout::P || head_dim != kitty_layout::D) {
            err = "kitty packed requires page=128 and head_dim=128"; return false;
        }
        stage_rows    = sink + kitty_layout::P;
        n_pages_max   = (n_ctx - sink + kitty_layout::P - 1) / kitty_layout::P;
        if (n_pages_max < 0) n_pages_max = 0;
        k_plane_bytes = (size_t) n_pages_max * kitty_layout::k_page_bytes(n_head_kv);
        v_plane_bytes = (size_t) n_pages_max * kitty_layout::v_page_bytes(n_head_kv);
    }

    const size_t n_tensors = packed ? (size_t) 4 * n_slots + 8 : (size_t) 2 * n_slots + 8;
    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors + 4096, nullptr, /*no_alloc=*/ true };
    kv.ctx = ggml_init(ip);
    if (!kv.ctx) { err = "kv cache: ggml_init failed"; return false; }

    kv.k.resize(n_slots);
    kv.v.resize(n_slots);
    if (packed) {
        kv.k_planes.resize(n_slots);
        kv.v_planes.resize(n_slots);
        for (int s = 0; s < n_slots; ++s) {
            kv.k[s] = ggml_new_tensor_2d(kv.ctx, GGML_TYPE_F16, kv.n_embd_kv, stage_rows);
            kv.v[s] = ggml_new_tensor_2d(kv.ctx, GGML_TYPE_F16, kv.n_embd_kv, stage_rows);
            kv.k_planes[s] = k_plane_bytes ? ggml_new_tensor_1d(kv.ctx, GGML_TYPE_I8, (int64_t) k_plane_bytes) : nullptr;
            kv.v_planes[s] = v_plane_bytes ? ggml_new_tensor_1d(kv.ctx, GGML_TYPE_I8, (int64_t) v_plane_bytes) : nullptr;
            char name[32];
            std::snprintf(name, sizeof(name), "kv_cache.k[%d]",  s); ggml_set_name(kv.k[s], name);
            std::snprintf(name, sizeof(name), "kv_cache.v[%d]",  s); ggml_set_name(kv.v[s], name);
            std::snprintf(name, sizeof(name), "kv_cache.kp[%d]", s); if (kv.k_planes[s]) ggml_set_name(kv.k_planes[s], name);
            std::snprintf(name, sizeof(name), "kv_cache.vp[%d]", s); if (kv.v_planes[s]) ggml_set_name(kv.v_planes[s], name);
        }
    } else {
        for (int s = 0; s < n_slots; ++s) {
            kv.k[s] = ggml_new_tensor_2d(kv.ctx, type, kv.n_embd_kv, n_ctx);
            kv.v[s] = ggml_new_tensor_2d(kv.ctx, type, kv.n_embd_kv, n_ctx);
            char name[32];
            std::snprintf(name, sizeof(name), "kv_cache.k[%d]", s); ggml_set_name(kv.k[s], name);
            std::snprintf(name, sizeof(name), "kv_cache.v[%d]", s); ggml_set_name(kv.v[s], name);
        }
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
    kv.k_planes.clear();
    kv.v_planes.clear();
}

void ouro_kv_cache_clear(ouro_kv_cache & kv) {
    if (kv.buf) ggml_backend_buffer_clear(kv.buf, 0);
    if (kv.kitty) {
        kv.kitty_k_done = kv.kitty_cfg.sink;
        kv.kitty_v_done = kv.kitty_cfg.sink;
        kv.kitty_k_pages  = 0;
        kv.kitty_v_packed = 0;
        kv.kitty_rng.seed(kv.kitty_cfg.seed);
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

ggml_tensor * ouro_kv_kitty_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot,
                                  int row, int n_tokens, bool is_v) {
    ggml_tensor * t = is_v ? kv.v[slot] : kv.k[slot];
    const size_t rs_row = ggml_row_size(t->type, kv.n_embd_kv);
    return ggml_view_2d(ctx, t, kv.n_embd_kv, n_tokens, rs_row, (size_t) row * rs_row);
}

ggml_tensor * ouro_kv_kitty_f16(const ouro_kv_cache & kv, ggml_context * ctx, int slot,
                                int n_rows, bool is_v) {
    ggml_tensor * t = is_v ? kv.v[slot] : kv.k[slot];
    const size_t rs_head = ggml_row_size(t->type, kv.head_dim);
    const size_t rs_row  = ggml_row_size(t->type, kv.n_embd_kv);
    return ggml_view_4d(ctx, t, kv.head_dim, kv.n_head_kv, n_rows, 1,
                        rs_head, rs_row, rs_row, 0);
}

ggml_tensor * ouro_kv_kitty_planes(const ouro_kv_cache & kv, int slot, bool is_v) {
    return is_v ? kv.v_planes[slot] : kv.k_planes[slot];
}
