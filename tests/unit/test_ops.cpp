// Validates how loopcd uses three ggml primitives against naive double-precision references.
// This locks eps, RoPE mode and the SwiGLU order before the full graph depends on them.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;

static void check(const char * name, double got, double want, double tol) {
    const double d = std::fabs(got - want);
    if (d > tol) {
        printf("FAIL %s: got %.8f want %.8f (d=%.3g)\n", name, got, want, d);
        g_fail = 1;
    } else {
        printf("ok   %s (d=%.3g)\n", name, d);
    }
}

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int H = 8;   // head dim
    const int N = 4;   // tokens
    const int NH = 2;  // heads
    const float eps = 1e-6f;
    const float theta_base = 1e6f;

    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_backend_t cpu = ggml_backend_cpu_init();

    // ---- RMSNorm over ne0, times a weight ----
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    std::vector<double> xd(H), wd(H);
    for (int i = 0; i < H; ++i) { xd[i] = dist(rng); wd[i] = dist(rng); }
    for (int i = 0; i < H; ++i) { ((float *) x->data)[i] = (float) xd[i]; ((float *) w->data)[i] = (float) wd[i]; }

    ggml_tensor * rn = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
    ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, rn);
    ggml_backend_graph_compute(cpu, g1);

    double ss = 0.0;
    for (int i = 0; i < H; ++i) ss += xd[i] * xd[i];
    const double rms = std::sqrt(ss / H + eps);
    for (int i = 0; i < H; ++i) {
        char nm[64];
        snprintf(nm, sizeof(nm), "rms_norm[%d]", i);
        check(nm, ((float *) rn->data)[i], xd[i] / rms * wd[i], 1e-5);
    }

    // ---- RoPE, GGML_ROPE_TYPE_NORMAL (consecutive pairs), full head dim ----
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, NH, N);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    std::vector<double> qd(H * NH * N);
    for (int i = 0; i < H * NH * N; ++i) { qd[i] = dist(rng); ((float *) q->data)[i] = (float) qd[i]; }
    for (int i = 0; i < N; ++i) ((int32_t *) pos->data)[i] = i;

    ggml_tensor * roped = ggml_rope_ext(ctx, q, pos, nullptr, H, GGML_ROPE_TYPE_NORMAL,
                                        0, theta_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, roped);
    ggml_backend_graph_compute(cpu, g2);

    for (int t = 0; t < N; ++t) {
        for (int h = 0; h < NH; ++h) {
            for (int k = 0; k < H / 2; ++k) {
                const double freq = std::pow((double) theta_base, -2.0 * k / H);
                const double ang = t * freq;
                const double c = std::cos(ang), s = std::sin(ang);
                const size_t base = (size_t) t * NH * H + (size_t) h * H;
                const double x0 = qd[base + 2 * k], x1 = qd[base + 2 * k + 1];
                char nm[64];
                snprintf(nm, sizeof(nm), "rope[%d,%d,%d].0", t, h, k);
                check(nm, ((float *) roped->data)[base + 2 * k], x0 * c - x1 * s, 1e-4);
                snprintf(nm, sizeof(nm), "rope[%d,%d,%d].1", t, h, k);
                check(nm, ((float *) roped->data)[base + 2 * k + 1], x0 * s + x1 * c, 1e-4);
            }
        }
    }

    // ---- SwiGLU: silu(gate) * up ----
    ggml_tensor * gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * up   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    std::vector<double> gd(H), ud(H);
    for (int i = 0; i < H; ++i) { gd[i] = dist(rng); ud[i] = dist(rng); }
    for (int i = 0; i < H; ++i) { ((float *) gate->data)[i] = (float) gd[i]; ((float *) up->data)[i] = (float) ud[i]; }

    ggml_tensor * swiglu = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_cgraph * g3 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g3, swiglu);
    ggml_backend_graph_compute(cpu, g3);

    for (int i = 0; i < H; ++i) {
        const double silu = gd[i] / (1.0 + std::exp(-gd[i]));
        char nm[64];
        snprintf(nm, sizeof(nm), "swiglu[%d]", i);
        check(nm, ((float *) swiglu->data)[i], silu * ud[i], 1e-5);
    }

    ggml_backend_free(cpu);
    ggml_free(ctx);

    printf(g_fail ? "OPS: FAIL\n" : "OPS: PASS\n");
    return g_fail;
}
