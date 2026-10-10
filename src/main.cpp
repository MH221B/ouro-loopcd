#include "model.h"
#include "graph.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> parse_token_list(const std::string & s) {
    std::vector<int32_t> out;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!cur.empty()) { out.push_back((int32_t) std::strtol(cur.c_str(), nullptr, 10)); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    return out;
}

static std::vector<int32_t> read_tokens_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path.c_str()); std::exit(1); }
    std::stringstream ss; ss << f.rdbuf();
    return parse_token_list(ss.str());
}

static std::string arg_value(int argc, char ** argv, int & i) {
    if (i + 1 >= argc) { fprintf(stderr, "error: missing value for %s\n", argv[i]); std::exit(1); }
    return argv[++i];
}

static int cmd_logits(int argc, char ** argv);
static int cmd_run(int argc, char ** argv);

static int cmd_inspect(const std::string & model_path, const std::string & device) {
    ouro_model m;
    std::string err;
    if (!ouro_backend_init(m, device, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!ouro_model_load(model_path, m, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const auto & hp = m.hp;
    printf("arch        : ouro\n");
    printf("n_layer     : %d\n", hp.n_layer);
    printf("n_loops     : %d\n", hp.n_loops);
    printf("n_embd      : %d\n", hp.n_embd);
    printf("n_ff        : %d\n", hp.n_ff);
    printf("n_head      : %d\n", hp.n_head);
    printf("n_head_kv   : %d\n", hp.n_head_kv);
    printf("head_dim    : %d\n", hp.n_embd_head);
    printf("n_vocab     : %d\n", hp.n_vocab);
    printf("n_ctx       : %d\n", hp.n_ctx);
    printf("eps         : %g\n", hp.eps);
    printf("rope_theta  : %g\n", hp.rope_theta);

    ouro_model_free(m);
    return 0;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    if (argc < 2) {
        fprintf(stderr, "usage: loopcd <inspect|logits|run> [args]\n");
        return 1;
    }
    const std::string cmd = argv[1];

    if (cmd == "inspect") {
        std::string model_path;
        std::string device = "cpu";
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if      (a == "-m" || a == "--model") model_path = arg_value(argc, argv, i);
            else if (a == "--device")             device     = arg_value(argc, argv, i);
            else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
        }
        if (model_path.empty()) { fprintf(stderr, "usage: loopcd inspect -m <model.gguf>\n"); return 1; }
        return cmd_inspect(model_path, device);
    }

    if (cmd == "logits") {
        return cmd_logits(argc - 2, argv + 2);
    }

    if (cmd == "run") {
        return cmd_run(argc - 2, argv + 2);
    }

    fprintf(stderr, "error: unknown command '%s'\n", cmd.c_str());
    return 1;
}

static void dump_f32(const std::string & path, ggml_tensor * t) {
    std::vector<float> buf((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) buf.data(), (std::streamsize) (buf.size() * sizeof(float)));
}

// Escape backslashes/quotes so a Windows path is valid inside a JSON string.
static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '\\' || c == '"') o.push_back('\\');
        o.push_back(c);
    }
    return o;
}

static int cmd_logits(int argc, char ** argv) {
    std::string model_path, tokens_str, tokens_file, dump;
    int loops = 0;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model")        model_path  = value();
        else if (a == "--tokens")               tokens_str  = value();
        else if (a == "--tokens-file")          tokens_file = value();
        else if (a == "--loops")                loops       = std::atoi(value().c_str());
        else if (a == "--dump")                 dump        = value();
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) { fprintf(stderr, "usage: loopcd logits -m <model.gguf> (--tokens a,b,c | --tokens-file f) [--loops N] [--dump prefix]\n"); return 1; }

    std::vector<int32_t> tokens = !tokens_file.empty() ? read_tokens_file(tokens_file) : parse_token_list(tokens_str);
    if (tokens.empty()) { fprintf(stderr, "error: no tokens\n"); return 1; }

    ouro_model m;
    std::string err;
    if (!ouro_backend_init(m, "cpu", err))     { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!ouro_model_load(model_path, m, err))  { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    ouro_graph_outputs out;
    if (!ouro_forward(m, tokens, loops, out, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const int R = loops > 0 ? loops : m.hp.n_loops;
    const int n = (int) tokens.size();
    const int V = m.hp.n_vocab;
    const int H = m.hp.n_embd;

    // report the last-token argmax of zR as a sanity check
    std::vector<float> zR_last(V);
    ggml_backend_tensor_get(out.zR, zR_last.data(), (size_t) (V * (n - 1)) * sizeof(float), V * sizeof(float));
    int best = 0;
    for (int i = 1; i < V; ++i) if (zR_last[i] > zR_last[best]) best = i;
    printf("loops=%d n_tokens=%d argmax(zR[-1])=%d\n", R, n, best);

    if (!dump.empty()) {
        dump_f32(dump + ".zR.bin", out.zR);
        dump_f32(dump + ".z1.bin", out.z1);
        dump_f32(dump + ".h1.bin", out.h1);
        dump_f32(dump + ".hR.bin", out.hR);
        std::ofstream j(dump + ".meta.json");
        const std::string d = json_escape(dump);
        j << "{\n";
        j << "  \"n_vocab\": " << V << ",\n";
        j << "  \"n_embd\": "  << H << ",\n";
        j << "  \"n_tokens\": " << n << ",\n";
        j << "  \"loops\": " << R << ",\n";
        j << "  \"tensors\": {\n";
        j << "    \"zR\": {\"shape\": [" << V << ", " << n << "], \"file\": \"" << d << ".zR.bin\"},\n";
        j << "    \"z1\": {\"shape\": [" << V << ", " << n << "], \"file\": \"" << d << ".z1.bin\"},\n";
        j << "    \"h1\": {\"shape\": [" << H << ", " << n << "], \"file\": \"" << d << ".h1.bin\"},\n";
        j << "    \"hR\": {\"shape\": [" << H << ", " << n << "], \"file\": \"" << d << ".hR.bin\"}\n";
        j << "  }\n}\n";
        printf("dumped to %s.*\n", dump.c_str());
    }

    ouro_model_free(m);
    return 0;
}

static int cmd_run(int argc, char ** argv) {
    (void) argc; (void) argv;
    fprintf(stderr, "run: not implemented yet\n");
    return 1;
}
