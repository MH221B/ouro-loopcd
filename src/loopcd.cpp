#include "loopcd.h"
#include "graph.h"

#include <cmath>
#include <cstdint>
#include <vector>

void ouro_loopcd_combine_logits(const float * zR, const float * z1,
                                int n_vocab, int n, float omega, float * out) {
    const size_t total = (size_t) n_vocab * (size_t) n;
    for (size_t i = 0; i < total; ++i) {
        out[i] = zR[i] + omega * (zR[i] - z1[i]);
    }
}

void ouro_loopcd_combine_hidden(const float * gR, const float * g1,
                                int n_embd, int n, float omega, float * out) {
    const size_t total = (size_t) n_embd * (size_t) n;
    for (size_t i = 0; i < total; ++i) {
        out[i] = gR[i] + omega * (gR[i] - g1[i]);
    }
}

float ouro_loopcd_adaptive_omega(const float * zR_col, int n_vocab, float omega_max) {
    // stable softmax + top-two margin (== Eq. 4)
    float mx = -INFINITY;
    for (int v = 0; v < n_vocab; ++v) if (zR_col[v] > mx) mx = zR_col[v];

    double sum = 0.0, p1 = 0.0, p2 = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        const double p = std::exp((double) zR_col[v] - (double) mx);
        sum += p;
        if (p > p1)      { p2 = p1; p1 = p; }
        else if (p > p2) { p2 = p; }
    }
    const double margin = (p1 - p2) / sum;              // p1, p2 normalized by sum
    float omega = (float) ((double) omega_max * (1.0 - margin));
    if (omega < 0.0f) omega = 0.0f;
    return omega;
}

bool ouro_loopcd_hidden_head(ouro_model & m, const float * g, int n, std::vector<float> & z_out, std::string & err) {
    const int H = m.hp.n_embd;
    const int V = m.hp.n_vocab;
    if (n <= 0) { err = "loopcd_hidden_head: empty input"; return false; }

    static std::vector<uint8_t> graph_buf;
    if (graph_buf.empty()) {
        graph_buf.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    }
    ggml_init_params ip = { graph_buf.size(), graph_buf.data(), /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor * inp_g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n);
    ggml_set_input(inp_g);
    ggml_set_name(inp_g, "loopcd_g");

    // lm_head = output_norm (RMSNorm) then the vocab projection; the norm sits AFTER the combine
    ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, inp_g, m.hp.eps), m.output_norm); // [H, n]
    ggml_tensor * z  = ggml_mul_mat(ctx, m.output, hn);                                  // [V, n]
    ggml_build_forward_expand(gf, z);

    ggml_backend_sched_reset(m.sched);
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) {
        err = "loopcd_hidden_head: scheduler alloc failed"; ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(inp_g, g, 0, (size_t) H * n * sizeof(float));
    if (ggml_backend_sched_graph_compute(m.sched, gf) != GGML_STATUS_SUCCESS) {
        err = "loopcd_hidden_head: graph compute failed"; ggml_free(ctx); return false;
    }

    z_out.resize((size_t) V * n);
    ggml_backend_tensor_get(z, z_out.data(), 0, z_out.size() * sizeof(float));

    ggml_free(ctx);
    return true;
}

static void tap_col(ggml_tensor * t, int col, std::vector<float> & dst) {
    const int D = (int) t->ne[0];
    dst.resize(D);
    ggml_backend_tensor_get(t, dst.data(), (size_t) col * D * sizeof(float), D * sizeof(float));
}

bool ouro_loopcd_guided_logits(ouro_model & m, const ouro_loopcd_params & lp, ouro_graph_outputs & out,
                               int col, std::vector<float> & z_out, std::string & err) {
    const int V = m.hp.n_vocab;
    const int H = m.hp.n_embd;

    std::vector<float> zR, z1;
    tap_col(out.zR, col, zR);
    tap_col(out.z1, col, z1);   // read before any head pass can clobber the taps

    if (lp.mode == OURO_LOOPCD_OFF) { z_out = zR; return true; }

    if (lp.mode == OURO_LOOPCD_HIDDEN) {
        std::vector<float> gR, g1;
        tap_col(out.gR, col, gR);
        tap_col(out.g1, col, g1);
        std::vector<float> gp(H);
        ouro_loopcd_combine_hidden(gR.data(), g1.data(), H, 1, lp.omega, gp.data());
        return ouro_loopcd_hidden_head(m, gp.data(), 1, z_out, err);
    }

    const float omega = lp.adaptive ? ouro_loopcd_adaptive_omega(zR.data(), V, lp.omega_max) : lp.omega;
    z_out.resize(V);
    ouro_loopcd_combine_logits(zR.data(), z1.data(), V, 1, omega, z_out.data());
    return true;
}
