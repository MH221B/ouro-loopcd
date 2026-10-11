// src/cache_layout.h — M7b packed Kitty-KV byte layout (spec §5.2).
//
// Single source of truth for the packed plane geometry, shared by the host packer
// (src/kitty_kv.cpp), the loopcd graph builder (src/graph.cpp), and the unit tests.
// The ggml fork's CPU/Vulkan kernels hard-code the same fixed values because they
// cannot include a loopcd header; the shader-free byte fixture (tests/unit/test_kitty_pack.cpp)
// is what catches any drift.
//
// Fixed (spec §5.5): P=128, D=128, D_BOOST=32, kbits=2, boost=4, vbits=2.
// Runtime (rides in op_params / flags): n_sink, n_pages, n_qbuf, n_vp, scale, seed.
//
// One I8 blob per slot per K and per V. Layout per page (P tokens), per head:
//
//   K head (5760 B): [ low 4096 | high 1024 | idx 128 | meta 512 ]
//     low  : channel-major, 4 tokens/byte. byte d*32 + t/4, bits (t%4)*2
//     high : sparse (slot) -major, same token packing. byte slot*32 + t/4
//     idx  : 1 byte/channel: dense->sparse LUT, sentinel == D_BOOST
//     meta : fp16 (scale, min) per channel, 4 B/channel (u32 == unpackHalf2x16)
//   V head (4608 B): [ data 4096 | meta 512 ]
//     data : token-major, 4 channels/byte. byte t*32 + d/4, bits (d%4)*2
//     meta : fp16 (scale, min) per token, 4 B/token
#pragma once
#include <cstddef>

namespace kitty_layout {

constexpr int P       = 128;   // page size (tokens)
constexpr int D       = 128;   // head_dim (channels)
constexpr int D_BOOST = 32;    // promoted channels reserved per head (sparse slots)

// ---- K ----
constexpr size_t K_LOW_BYTES  = (size_t) D       * (P / 4);   // 4096
constexpr size_t K_HIGH_BYTES = (size_t) D_BOOST * (P / 4);   // 1024
constexpr size_t K_IDX_BYTES  = (size_t) D;                   // 128
constexpr size_t K_META_BYTES = (size_t) D * 2 * 2;           // 512
constexpr size_t K_HEAD_BYTES = K_LOW_BYTES + K_HIGH_BYTES + K_IDX_BYTES + K_META_BYTES; // 5760
constexpr size_t K_OFF_HIGH   = K_LOW_BYTES;                  // 4096
constexpr size_t K_OFF_IDX    = K_LOW_BYTES + K_HIGH_BYTES;    // 5120
constexpr size_t K_OFF_META   = K_OFF_IDX + K_IDX_BYTES;       // 5248

// ---- V ----
constexpr size_t V_DATA_BYTES = (size_t) D * (P / 4);         // 4096
constexpr size_t V_META_BYTES = (size_t) P * 2 * 2;           // 512
constexpr size_t V_HEAD_BYTES = V_DATA_BYTES + V_META_BYTES;  // 4608
constexpr size_t V_OFF_META   = V_DATA_BYTES;                 // 4096

inline size_t k_page_bytes(int n_head_kv) { return (size_t) n_head_kv * K_HEAD_BYTES; }
inline size_t v_page_bytes(int n_head_kv) { return (size_t) n_head_kv * V_HEAD_BYTES; }

inline size_t k_head_off(int h) { return (size_t) h * K_HEAD_BYTES; }
inline size_t v_head_off(int h) { return (size_t) h * V_HEAD_BYTES; }

// K low-plane byte for (head, channel, token-in-page).
inline size_t k_low_off(int h, int d, int t) {
    return k_head_off(h) + (size_t) d * (P / 4) + (size_t) (t / 4);
}
// K high-plane byte for (head, sparse slot, token-in-page).
inline size_t k_high_off(int h, int slot, int t) {
    return k_head_off(h) + K_OFF_HIGH + (size_t) slot * (P / 4) + (size_t) (t / 4);
}
inline size_t k_idx_off(int h, int d) { return k_head_off(h) + K_OFF_IDX + (size_t) d; }
inline size_t k_meta_off(int h, int d) { return k_head_off(h) + K_OFF_META + (size_t) d * 4; }

// V data byte for (head, token-in-page, channel).
inline size_t v_data_off(int h, int t, int d) {
    return v_head_off(h) + (size_t) t * (D / 4) + (size_t) (d / 4);
}
inline size_t v_meta_off(int h, int t) { return v_head_off(h) + V_OFF_META + (size_t) t * 4; }

} // namespace kitty_layout
