#include "graph.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#define OURO_MAX_NODES 16384

static ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Full-sequence causal attention. q,k,v are [head_dim, n_head, n] after RoPE.
static ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                               int n_head, int head_dim, int n, float scale) {
    ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3);                 // [hd, n, nh]
    ggml_tensor * K = ggml_permute(ctx, k, 0, 2, 1, 3);                 // [hd, n, nh]
    ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);                        // [n, n, nh]
    KQ = ggml_scale(ctx, KQ, scale);
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);
    ggml_tensor * Vt = ggml_cont_3d(ctx, ggml_permute(ctx, v, 1, 2, 0, 3), n, head_dim, n_head);
    ggml_tensor * KQV = ggml_mul_mat(ctx, Vt, KQ);                     // [hd, n, nh]
    return ggml_cont_2d(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3), head_dim * n_head, n);
}

bool ouro_forward(ouro_model & m, const std::vector<int32_t> & tokens, int loops,
                  ouro_graph_outputs & out, std::string & err) {
    const auto & hp = m.hp;
    const int n      = (int) tokens.size();
    const int hd     = hp.n_embd_head;
    const int nh     = hp.n_head;
    const int R      = loops > 0 ? loops : hp.n_loops;
    if (n <= 0) { err = "empty token list"; return false; }
    if (nh != hp.n_head_kv) { err = "GQA is not supported in M1 (n_head != n_head_kv)"; return false; }

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

    const float scale = 1.0f / std::sqrt((float) hd);

    ggml_tensor * x = ggml_cast(ctx, ggml_get_rows(ctx, m.tok_embd, inp_tokens), GGML_TYPE_F32);

    ggml_tensor * h1 = nullptr;
    ggml_tensor * hR = nullptr;

    for (int j = 0; j < R; ++j) {
        for (int i = 0; i < hp.n_layer; ++i) {
            const auto & L = m.layers[i];

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

            cur = attention(ctx, Q3, K3, V3, nh, hd, n, scale);
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
    for (int i = 0; i < n; ++i) pos[i] = i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(m.sched, gf) != GGML_STATUS_SUCCESS) { err = "graph compute failed"; ggml_free(ctx); return false; }

    out.zR = zR; out.z1 = z1; out.h1 = h1; out.hR = hR;
    ggml_free(ctx); // tensor structs are referenced by out; caller must read before next forward
    return true;
}
