// Pure-function tests for sampling (spec 4.6): temp/top-k/top-p/repetition penalty/greedy.
// Written first, against src/sampling.h which does not exist yet.
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;

static void check(const char * name, double got, double want, double tol) {
    const double d = std::fabs(got - want);
    if (d > tol) { printf("FAIL %s: got %.8f want %.8f (d=%.3g)\n", name, got, want, d); g_fail = 1; }
    else         { printf("ok   %s (d=%.3g)\n", name, d); }
}

static void test_greedy() {
    const std::vector<float> logits = { 1.0f, 3.0f, 2.0f, 0.0f };
    ouro_sampler_params sp; sp.temp = 0.0f; sp.seed = 7;
    std::mt19937 rng(sp.seed);
    check("greedy argmax", ouro_sample(logits, {}, sp, rng), 1, 0);

    sp.top_k = 1; sp.temp = 1.0f;
    std::mt19937 rng2(sp.seed);
    check("top_k=1 argmax", ouro_sample(logits, {}, sp, rng2), 1, 0);
}

static void test_repeat_penalty() {
    // a positive logit that was just used is pushed down by the penalty
    std::vector<float> logits = { 5.0f, 0.0f };
    ouro_apply_repeat_penalty(logits, { 0 }, 8, 2.0f);
    check("repeat penalty divides positive by p", logits[0], 2.5, 1e-6);

    std::vector<float> logits2 = { -4.0f, 0.0f };
    ouro_apply_repeat_penalty(logits2, { 0 }, 8, 2.0f);
    check("repeat penalty multiplies negative by p", logits2[0], -8.0, 1e-6);
}

static void test_softmax() {
    std::vector<float> x = { 0.0f, 0.0f, 0.0f };
    ouro_softmax_inplace(x, 1.0f);
    for (int i = 0; i < 3; ++i) check("softmax uniform", x[i], 1.0 / 3.0, 1e-6);

    std::vector<float> y = { 0.0f, 1.0f };
    ouro_softmax_inplace(y, 1.0f);
    const double e = std::exp(1.0);
    check("softmax y[0]", y[0], 1.0 / (1.0 + e), 1e-6);
    check("softmax y[1]", y[1], e / (1.0 + e), 1e-6);
}

static void test_topk_topp() {
    std::vector<float> p = { 0.1f, 0.4f, 0.3f, 0.2f };
    ouro_top_k_top_p(p, 2, 1.0f);   // keep the two largest (idx 1, 2)
    check("top_k keeps idx1", p[1] > 0 ? 1.0 : 0.0, 1.0, 0);
    check("top_k keeps idx2", p[2] > 0 ? 1.0 : 0.0, 1.0, 0);
    check("top_k drops idx0", p[0], 0.0, 0);
    check("top_k drops idx3", p[3], 0.0, 0);

    std::vector<float> q = { 0.5f, 0.3f, 0.2f, 0.0f };
    ouro_top_k_top_p(q, 0, 0.8f);   // nucleus {0.5,0.3} = 0.8 -> keep idx 0 and 1
    check("top_p keeps idx0", q[0] > 0 ? 1.0 : 0.0, 1.0, 0);
    check("top_p keeps idx1", q[1] > 0 ? 1.0 : 0.0, 1.0, 0);
    check("top_p drops idx2", q[2], 0.0, 0);
}

static void test_determinism() {
    const std::vector<float> logits = { 1.0f, 1.5f, 0.2f, 0.1f };
    ouro_sampler_params sp; sp.temp = 1.0f; sp.top_p = 0.95f; sp.seed = 42;
    std::mt19937 a(sp.seed), b(sp.seed);
    int same = 1;
    for (int i = 0; i < 16; ++i) if (ouro_sample(logits, {}, sp, a) != ouro_sample(logits, {}, sp, b)) same = 0;
    check("seeded sampling is deterministic", same, 1, 0);
}

int main() {
    test_greedy();
    test_repeat_penalty();
    test_softmax();
    test_topk_topp();
    test_determinism();
    printf(g_fail ? "SAMPLING: FAIL\n" : "SAMPLING: PASS\n");
    return g_fail;
}
