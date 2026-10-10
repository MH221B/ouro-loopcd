// Pure-function tests for the LoopCD combine math (spec 7.3 / paper Eqs. 2-4).
// Written before src/loopcd.cpp exists: the first build must fail, then pass after Step 3.
#include "loopcd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;

static void check(const char * name, double got, double want, double tol) {
    const double d = std::fabs(got - want);
    if (d > tol) { printf("FAIL %s: got %.8f want %.8f (d=%.3g)\n", name, got, want, d); g_fail = 1; }
    else         { printf("ok   %s (d=%.3g)\n", name, d); }
}

static void test_combine_logits() {
    // Eq. (3): z' = zR + omega*(zR - z1)
    const int V = 4;
    const float zR[4] = { 1.0f, 0.9f, 0.0f, 0.0f };
    const float z1[4] = { 0.8f, 0.0f, 0.0f, 0.0f };
    float out[4];
    ouro_loopcd_combine_logits(zR, z1, V, 1, 0.5f, out);
    check("logits z'[0]", out[0], 1.10, 1e-6);
    check("logits z'[1]", out[1], 1.35, 1e-6);
    check("logits z'[2]", out[2], 0.00, 1e-6);
    check("logits z'[3]", out[3], 0.00, 1e-6);

    // omega = 0 recovers the unguided logits exactly
    float out0[4];
    ouro_loopcd_combine_logits(zR, z1, V, 1, 0.0f, out0);
    for (int i = 0; i < V; ++i) check("logits omega=0 recovers zR", out0[i], zR[i], 0.0);
}

// tiny independent double references for the synthetic hidden-form construction
static void ref_rms_norm(const double * x, const double * w, int H, double eps, double * y) {
    double ss = 0.0;
    for (int i = 0; i < H; ++i) ss += x[i] * x[i];
    const double rms = std::sqrt(ss / H + eps);
    for (int i = 0; i < H; ++i) y[i] = x[i] / rms * w[i];
}
static void ref_head(const double * W, int V, int H, const double * x, double * z) {
    for (int v = 0; v < V; ++v) {
        double acc = 0.0;
        for (int k = 0; k < H; ++k) acc += W[v * H + k] * x[k];
        z[v] = acc;
    }
}

static void test_combine_hidden_pre_norm() {
    // Eq. (2) on the PRE-norm loop outputs: g' = gR + omega*(gR - g1)
    const int H = 4;
    const float gR[4] = { 1.0f, 0.9f, 0.0f, 0.0f };
    const float g1[4] = { 0.8f, 0.0f, 0.0f, 0.0f };
    float out[4];
    ouro_loopcd_combine_hidden(gR, g1, H, 1, 0.5f, out);
    check("hidden g'[0]", out[0], 1.10, 1e-6);
    check("hidden g'[1]", out[1], 1.35, 1e-6);
    check("hidden g'[2]", out[2], 0.00, 1e-6);

    // omega = 0 recovers the pre-norm final state exactly
    float out0[4];
    ouro_loopcd_combine_hidden(gR, g1, H, 1, 0.0f, out0);
    for (int i = 0; i < H; ++i) check("hidden omega=0 recovers gR", out0[i], gR[i], 0.0);
}

static void test_hidden_norm_after_combine() {
    // Verify the pre-norm placement: hidden = N(g_R + omega*(g_R - g1)) then the head,
    // while combining the POST-norm states collapses to the logits form.
    const int H = 3, V = 3;
    const double g1d[H] = { 1.0, 0.5, -0.5 };
    const double gRd[H] = { 1.2, 0.9, -0.1 };
    const double w[H]   = { 1.0, 1.0, 1.0 };
    const double W[V * H] = {  1.0, 0.5, -0.5,
                               0.0, 1.0,  0.25,
                              -0.25, 0.0, 1.0 };
    const double eps = 1e-6;
    const double omega = 0.5;

    double h1d[H], hRd[H], z1d[V], zRd[V];
    ref_rms_norm(g1d, w, H, eps, h1d);
    ref_rms_norm(gRd, w, H, eps, hRd);
    ref_head(W, V, H, h1d, z1d);
    ref_head(W, V, H, hRd, zRd);

    float g1f[H], gRf[H], gpf[H];
    for (int i = 0; i < H; ++i) { g1f[i] = (float) g1d[i]; gRf[i] = (float) gRd[i]; }

    // logits form
    double logits[V];
    for (int v = 0; v < V; ++v) logits[v] = zRd[v] + omega * (zRd[v] - z1d[v]);

    // hidden form: combine pre-norm, norm, head
    ouro_loopcd_combine_hidden(gRf, g1f, H, 1, (float) omega, gpf);
    double gpd[H], hpd[H], hidden[V];
    for (int i = 0; i < H; ++i) gpd[i] = (double) gpf[i];
    ref_rms_norm(gpd, w, H, eps, hpd);
    ref_head(W, V, H, hpd, hidden);

    double max_diff = 0.0;
    for (int v = 0; v < V; ++v) max_diff = std::max(max_diff, std::fabs(hidden[v] - logits[v]));
    check("hidden != logits at omega=0.5", max_diff > 1e-4 ? 1.0 : 0.0, 1.0, 0.0);

    // at omega = 0 both forms equal zR
    ouro_loopcd_combine_hidden(gRf, g1f, H, 1, 0.0f, gpf);
    for (int i = 0; i < H; ++i) gpd[i] = (double) gpf[i];
    ref_rms_norm(gpd, w, H, eps, hpd);
    double hidden0[V];
    ref_head(W, V, H, hpd, hidden0);
    double max_h0 = 0.0;
    for (int v = 0; v < V; ++v) max_h0 = std::max(max_h0, std::fabs(hidden0[v] - zRd[v]));
    check("hidden omega=0 == zR", max_h0, 0.0, 1e-5);

    // the wrong path (combine post-norm h, then the linear head) reproduces the logits form
    double hcomb[H], z_lin[V], maxc = 0.0;
    for (int i = 0; i < H; ++i) hcomb[i] = hRd[i] + omega * (hRd[i] - h1d[i]);
    ref_head(W, V, H, hcomb, z_lin);
    for (int v = 0; v < V; ++v) maxc = std::max(maxc, std::fabs(z_lin[v] - logits[v]));
    check("post-norm combine + linear head == logits (collapse)", maxc, 0.0, 1e-9);
}

static void test_adaptive_omega() {
    const int V = 4;
    const float zR[4] = { 1.0f, 0.9f, 0.0f, 0.0f };

    // independent double reference: softmax + top-two margin, Eq. (4)
    double mx = -1e300;
    for (int i = 0; i < V; ++i) mx = std::max(mx, (double) zR[i]);
    double p[4], sum = 0.0;
    for (int i = 0; i < V; ++i) { p[i] = std::exp((double) zR[i] - mx); sum += p[i]; }
    const double a = p[0] / sum, b = p[1] / sum;
    const double want = 1.0 * (1.0 - (a - b));
    check("adaptive omega (cap 1.0)", ouro_loopcd_adaptive_omega(zR, V, 1.0f), want, 1e-6);

    // ties on the top two -> the cap
    const float tie[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
    check("adaptive omega tie -> cap", ouro_loopcd_adaptive_omega(tie, V, 1.5f), 1.5, 1e-4);

    // a dominant top token -> guidance withheld (omega near 0)
    const float dom[4] = { 20.0f, 0.0f, 0.0f, 0.0f };
    check("adaptive omega dominant -> ~0", ouro_loopcd_adaptive_omega(dom, V, 1.0f), 0.0, 1e-4);
}

static void test_seeded_three_modes() {
    // A fixed ("seeded") synthetic step. Unguided the top token is 0; guidance must flip it to 1.
    const int V = 4;
    const float zR[4] = { 1.0f, 0.9f, 0.0f, 0.0f };
    const float z1[4] = { 0.8f, 0.0f, 0.0f, 0.0f };
    const float omega = 0.5f;

    int off = 0;
    for (int i = 1; i < V; ++i) if (zR[i] > zR[off]) off = i;

    float zp[4];
    ouro_loopcd_combine_logits(zR, z1, V, 1, omega, zp);
    int guided = 0;
    for (int i = 1; i < V; ++i) if (zp[i] > zp[guided]) guided = i;

    check("seeded off argmax",    off,    0, 0);
    check("seeded logits argmax", guided, 1, 0);
    check("seeded guided z'[0]",  zp[0], 1.10, 1e-6);
    check("seeded guided z'[1]",  zp[1], 1.35, 1e-6);
}

int main() {
    test_combine_logits();
    test_combine_hidden_pre_norm();
    test_hidden_norm_after_combine();
    test_adaptive_omega();
    test_seeded_three_modes();
    printf(g_fail ? "LOOPCD: FAIL\n" : "LOOPCD: PASS\n");
    return g_fail;
}
