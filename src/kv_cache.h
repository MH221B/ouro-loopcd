#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>

// The cached K/V for one logical layer occupies one slot, because K/V differ per loop.
// Ouro has n_layer * n_loops slots (48 * 4 = 192); slot index = loop * n_layer + layer.
//
// Layout: `k` and `v` are each one [head_dim * n_head_kv, n_ctx, n_slots] tensor.
// One row of head_dim * n_head_kv contiguous values is the K (or V) for one token in
// one slot, laid out head-major (element h*head_dim + d). The per-head read view splits
// dim 0 into [head_dim, n_head_kv].
struct ouro_kv_cache {
    int head_dim    = 0;   // 128
    int n_head_kv   = 0;   // 16
    int n_embd_kv   = 0;   // head_dim * n_head_kv = 2048
    int n_ctx       = 0;   // each slot is n_ctx rows tall
    int n_slots     = 0;   // 192
    ggml_type type  = GGML_TYPE_Q8_0; // q8_0 (default) or f16
    ggml_context * ctx = nullptr;     // owns the k/v tensor structs
    ggml_backend_buffer_t buf = nullptr; // owns the k/v data
    ggml_tensor * k = nullptr;        // [n_embd_kv, n_ctx, n_slots]
    ggml_tensor * v = nullptr;        // [n_embd_kv, n_ctx, n_slots]
};

// Allocates both cache tensors on `backend`. type must be GGML_TYPE_Q8_0 or GGML_TYPE_F16.
bool ouro_kv_cache_init(ouro_kv_cache & kv, ggml_backend_t backend,
                        int head_dim, int n_head_kv, int n_ctx, int n_slots,
                        ggml_type type, std::string & err);

void ouro_kv_cache_free(ouro_kv_cache & kv);

// Zero the cache. Call before a fresh sequence.
void ouro_kv_cache_clear(ouro_kv_cache & kv);

// Per-head read view for `slot` over the first `n_kv` positions: [head_dim, n_head_kv, n_kv, 1].
// Feed to ggml_permute(0,2,1,3) before ggml_flash_attn_ext. Fully contiguous when n_kv == n_ctx.
ggml_tensor * ouro_kv_k(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_kv);
ggml_tensor * ouro_kv_v(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_kv);

// Flat write view for `slot`, positions [n_past, n_past + n_tokens): [n_embd_kv, n_tokens].
// Feed to ggml_cpy with a [n_embd_kv, n_tokens] f32 source. Contiguous.
ggml_tensor * ouro_kv_k_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_past, int n_tokens);
ggml_tensor * ouro_kv_v_write(const ouro_kv_cache & kv, ggml_context * ctx, int slot, int n_past, int n_tokens);
