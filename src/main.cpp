#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    ggml_time_init();
    printf("loopcd: ggml %s\n", ggml_version());

    // smoke test: a 2x2 matmul must compute
    ggml_init_params ip = { 4 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    float ad[4] = { 1, 2, 3, 4 };
    float bd[4] = { 5, 6, 7, 8 };
    for (int i = 0; i < 4; ++i) { ((float *) a->data)[i] = ad[i]; ((float *) b->data)[i] = bd[i]; }

    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_graph_compute(cpu, gf);

    printf("smoke: %g %g %g %g\n",
        ((float *) c->data)[0], ((float *) c->data)[1], ((float *) c->data)[2], ((float *) c->data)[3]);

    ggml_backend_free(cpu);
    ggml_free(ctx);
    return 0;
}
