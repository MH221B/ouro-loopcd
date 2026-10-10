#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>

// M6 Kitty-KV config. nullptr (or kitty=false) means the cache is a plain f16/q8_0 cache.
struct ouro_kitty_config {
    int sink          = 32;   // leading tokens kept fp16
    int page          = 128;  // page size for key quantization
    int boost_channels= 0;    // 0 => head_dim//4, computed at init
    int kbits         = 2;
    int vbits         = 2;
    int boost_bits    = 4;
    bool random_select= false;
    unsigned long long seed = 1234;
};

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
    ggml_type type  = GGML_TYPE_Q8_0; // q8_0 (default), f16, or f32
    ggml_context * ctx = nullptr;     // owns the k/v tensor structs
    ggml_backend_buffer_t buf = nullptr; // owns the k/v data
    ggml_tensor * k = nullptr;        // [n_embd_kv, n_ctx, n_slots]
    ggml_tensor * v = nullptr;        // [n_embd_kv, n_ctx, n_slots]

    // M6: fake-quantize the f16 cache after each forward.
    bool kitty = false;              // valid when kitty == true
    ouro_kitty_config kitty_cfg{};
    long long kitty_k_done = 0;      // positions whose K page is already quantized
    long long kitty_v_done = 0;      // positions whose V is already quantized
};

// Allocates both cache tensors on `backend`. type must be GGML_TYPE_Q8_0, GGML_TYPE_F16 or GGML_TYPE_F32.
// When `kitty` is non-null the cache is f16 storage with the Kitty quantizing write path enabled.
bool ouro_kv_cache_init(ouro_kv_cache & kv, ggml_backend_t backend,
                        int head_dim, int n_head_kv, int n_ctx, int n_slots,
                        ggml_type type, const ouro_kitty_config * kitty, std::string & err);

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
