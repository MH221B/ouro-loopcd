#include "graph.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#define OURO_MAX_NODES 16384

static ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// ggml_flash_attn_ext over one slot. q is [head_dim, n, n_head], k/v are
// [head_dim, n_kv, n_head], mask is the F16 [n_kv, n, 1, 1] additive mask.
static ggml_tensor * flash_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                ggml_tensor * mask, int head_dim, int n_head, int n, float scale) {
    ggml_tensor * cur = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, cur, head_dim * n_head, n);
}

// Causal additive mask: keep key j when j <= n_past + query_index.
static void fill_mask(ggml_tensor * mask, int n_kv, int n, int n_past) {
    std::vector<ggml_fp16_t> data((size_t) n_kv * n);
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n; ++i) {
        const int q_pos = n_past + i;
        for (int j = 0; j < n_kv; ++j) {
            data[(size_t) i * n_kv + j] = (j <= q_pos) ? keep : drop;
        }
    }
    ggml_backend_tensor_set(mask, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
}

bool ouro_forward_kv(ouro_model & m, ouro_kv_cache & kv, const std::vector<int32_t> & tokens,
                     int n_past, int loops, ouro_graph_outputs & out, std::string & err) {
    const auto & hp = m.hp;
    const int n    = (int) tokens.size();
    const int hd   = hp.n_embd_head;
    const int nh   = hp.n_head;
    const int R    = loops > 0 ? loops : hp.n_loops;
    const int n_kv = n_past + n;

    if (n <= 0)                    { err = "empty token list"; return false; }
    if (nh != hp.n_head_kv)        { err = "M2 supports MHA only (n_head != n_head_kv)"; return false; }
    if (kv.n_slots != hp.n_layer * R) { err = "KV slots do not match n_layer * loops"; return false; }
    if (kv.n_embd_kv != hd * nh)   { err = "KV row width does not match head_dim * n_head"; return false; }
    if (n_kv > kv.n_ctx)           { err = "context overflow: n_past + n > n_ctx"; return false; }

    static std::vector<uint8_t> graph_buf;
    if (graph_buf.empty()) {
        graph_buf.resize(ggml_tensor_overhead() * OURO_MAX_NODES + ggml_graph_overhead_custom(OURO_MAX_NODES, false));
    }
    ggml_init_params ip = { graph_buf.size(), graph_buf.data(), /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, OURO_MAX_NODES, false);

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor * inp_pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(inp_tokens);
    ggml_set_input(inp_pos);

    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n);
    ggml_set_input(mask);
    ggml_set_name(mask, "attn_mask");

    const float scale = 1.0f / std::sqrt((float) hd);

    ggml_tensor * x = ggml_cast(ctx, ggml_get_rows(ctx, m.tok_embd, inp_tokens), GGML_TYPE_F32);

    ggml_tensor * h1 = nullptr;
    ggml_tensor * hR = nullptr;
    ggml_tensor * g1 = nullptr;
    ggml_tensor * gR = nullptr;

    for (int j = 0; j < R; ++j) {
        for (int i = 0; i < hp.n_layer; ++i) {
            const auto & L = m.layers[i];
            const int slot = j * hp.n_layer + i;

            ggml_tensor * inpSA = x;
            ggml_tensor * cur = rms_norm(ctx, x, L.attn_norm, hp.eps);

            ggml_tensor * Q = ggml_mul_mat(ctx, L.attn_q, cur);
            ggml_tensor * K = ggml_mul_mat(ctx, L.attn_k, cur);
            ggml_tensor * V = ggml_mul_mat(ctx, L.attn_v, cur);

            ggml_tensor * Q3 = ggml_reshape_3d(ctx, Q, hd, nh, n);
            ggml_tensor * K3 = ggml_reshape_3d(ctx, K, hd, nh, n);
            ggml_tensor * V3 = ggml_reshape_3d(ctx, V, hd, nh, n);

            Q3 = ggml_rope_ext(ctx, Q3, inp_pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                               0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K3 = ggml_rope_ext(ctx, K3, inp_pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                               0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // write the new rows into this slot; expand now so the write precedes the read
            {
                ggml_tensor * K2 = ggml_view_2d(ctx, K3, kv.n_embd_kv, n, K3->nb[2], 0);
                ggml_tensor * V2 = ggml_view_2d(ctx, V3, kv.n_embd_kv, n, V3->nb[2], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, K2, ouro_kv_k_write(kv, ctx, slot, n_past, n)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, V2, ouro_kv_v_write(kv, ctx, slot, n_past, n)));
            }

            // attend over the whole slot [0, n_kv)
            ggml_tensor * Qp = ggml_permute(ctx, Q3, 0, 2, 1, 3);                        // [hd, n, nh, 1]
            ggml_tensor * Kc = ggml_permute(ctx, ouro_kv_k(kv, ctx, slot, n_kv), 0, 2, 1, 3); // [hd, n_kv, nh, 1]
            ggml_tensor * Vc = ggml_permute(ctx, ouro_kv_v(kv, ctx, slot, n_kv), 0, 2, 1, 3); // [hd, n_kv, nh, 1]

            cur = flash_attn(ctx, Qp, Kc, Vc, mask, hd, nh, n, scale);
            cur = ggml_mul_mat(ctx, L.attn_output, cur);
            cur = rms_norm(ctx, cur, L.attn_post_norm, hp.eps);

            ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            cur = rms_norm(ctx, ffn_inp, L.ffn_norm, hp.eps);
            ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, L.ffn_gate, cur));
            ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up, cur);
            cur = ggml_mul(ctx, gate, up);
            cur = ggml_mul_mat(ctx, L.ffn_down, cur);
            cur = rms_norm(ctx, cur, L.ffn_post_norm, hp.eps);

            x = ggml_add(ctx, cur, ffn_inp);
        }

        // Pre-norm tap: the layer-stack output before output_norm (used by LoopCD-Hidden).
        if (j == 0)     g1 = x;
        if (j == R - 1) gR = x;

        // The reference applies model.norm at the end of every loop, including the last.
        x = rms_norm(ctx, x, m.output_norm, hp.eps);
        if (j == 0)     h1 = x;
        if (j == R - 1) hR = x;
    }

    ggml_tensor * zR = ggml_mul_mat(ctx, m.output, hR);
    ggml_tensor * z1 = ggml_mul_mat(ctx, m.output, h1);

    ggml_build_forward_expand(gf, zR);
    ggml_build_forward_expand(gf, z1);

    ggml_backend_sched_reset(m.sched);
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) { err = "scheduler failed to allocate the graph"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(inp_tokens, tokens.data(), 0, tokens.size() * sizeof(int32_t));
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = n_past + i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    fill_mask(mask, n_kv, n, n_past);

    if (ggml_backend_sched_graph_compute(m.sched, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; ggml_free(ctx); return false; }

    out.zR = zR; out.z1 = z1; out.h1 = h1; out.hR = hR; out.g1 = g1; out.gR = gR;
    ggml_free(ctx);
    return true;
}
