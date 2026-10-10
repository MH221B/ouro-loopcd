#include "model.h"
#include "graph.h"
#include "loopcd.h"
#include "tokenizer.h"
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

static ggml_type parse_kv_type(const std::string & s, std::string & err) {
    if (s == "q8_0" || s == "q8") return GGML_TYPE_Q8_0;
    if (s == "f16")               return GGML_TYPE_F16;
    if (s == "f32")               return GGML_TYPE_F32;
    err = "unknown --kv-type '" + s + "' (use q8_0, f16 or f32)";
    return GGML_TYPE_Q8_0;
}

static int argmax_col(ggml_tensor * z, int col) {
    const int V = (int) z->ne[0];
    std::vector<float> buf(V);
    ggml_backend_tensor_get(z, buf.data(), (size_t) col * V * sizeof(float), V * sizeof(float));
    int best = 0;
    for (int i = 1; i < V; ++i) if (buf[i] > buf[best]) best = i;
    return best;
}

static ouro_loopcd_mode parse_loopcd_mode(const std::string & s, std::string & err) {
    if (s == "off")    return OURO_LOOPCD_OFF;
    if (s == "logits") return OURO_LOOPCD_LOGITS;
    if (s == "hidden") return OURO_LOOPCD_HIDDEN;
    err = "unknown --loopcd '" + s + "' (use off|logits|hidden)";
    return OURO_LOOPCD_OFF;
}

// Copy one column (token position) of a [dim, n_tokens] tap to host.
static void tensor_col_to_host(ggml_tensor * t, int col, std::vector<float> & dst) {
    const int D = (int) t->ne[0];
    dst.resize(D);
    ggml_backend_tensor_get(t, dst.data(), (size_t) col * D * sizeof(float), D * sizeof(float));
}

static int argmax_host(const std::vector<float> & v) {
    int best = 0;
    for (int i = 1; i < (int) v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

static void dump_f32_vec(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(float)));
}

// Applies the selected LoopCD mode to one token column and returns the guided argmax.
// Reads zR and z1 up front (before any head pass can clobber the taps); the optional out
// pointers receive z' / zR / z1 for the on-vs-off check. Returns -1 on error.
static int loopcd_guided_argmax(ouro_model & m, const ouro_loopcd_params & lp, ouro_graph_outputs & out,
                                int col, std::vector<float> * guided, std::vector<float> * unguided,
                                std::vector<float> * ref1, std::string & err) {
    const int V = m.hp.n_vocab;
    const int H = m.hp.n_embd;

    std::vector<float> zR, z1;
    tensor_col_to_host(out.zR, col, zR);
    tensor_col_to_host(out.z1, col, z1);            // read before the hidden head pass
    if (unguided) *unguided = zR;
    if (ref1)     *ref1     = z1;

    if (lp.mode == OURO_LOOPCD_OFF) {
        if (guided) *guided = zR;
        return argmax_host(zR);
    }

    if (lp.mode == OURO_LOOPCD_HIDDEN) {
        // combine the PRE-norm loop outputs, then apply output_norm and the head
        std::vector<float> gR, g1;
        tensor_col_to_host(out.gR, col, gR);
        tensor_col_to_host(out.g1, col, g1);
        std::vector<float> gp(H);
        ouro_loopcd_combine_hidden(gR.data(), g1.data(), H, 1, lp.omega, gp.data());
        std::vector<float> zp;
        if (!ouro_loopcd_hidden_head(m, gp.data(), 1, zp, err)) return -1;
        if (guided) *guided = zp;
        return argmax_host(zp);
    }

    // logits form
    const float omega = lp.adaptive ? ouro_loopcd_adaptive_omega(zR.data(), V, lp.omega_max) : lp.omega;
    std::vector<float> zp(V);
    ouro_loopcd_combine_logits(zR.data(), z1.data(), V, 1, omega, zp.data());
    if (guided) *guided = zp;
    return argmax_host(zp);
}

static int cmd_logits(int argc, char ** argv);
static int cmd_run(int argc, char ** argv);
static int cmd_tokenize(int argc, char ** argv);

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

    if (cmd == "tokenize") {
        return cmd_tokenize(argc - 2, argv + 2);
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
    std::string model_path, tokens_str, tokens_file, dump, device = "cpu", kv_str = "q8_0";
    int loops = 0, ctx = 512;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-m" || a == "--model")        model_path  = arg_value(argc, argv, i);
        else if (a == "--tokens")                    tokens_str  = arg_value(argc, argv, i);
        else if (a == "--tokens-file")               tokens_file = arg_value(argc, argv, i);
        else if (a == "--loops")                     loops       = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--dump")                      dump        = arg_value(argc, argv, i);
        else if (a == "--device")                    device      = arg_value(argc, argv, i);
        else if (a == "--kv-type")                   kv_str      = arg_value(argc, argv, i);
        else if (a == "--ctx")                       ctx         = std::atoi(arg_value(argc, argv, i).c_str());
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: loopcd logits -m <model.gguf> (--tokens a,b,c | --tokens-file f) "
                        "[--loops N] [--ctx N] [--device cpu|vulkan] [--kv-type q8_0|f16] [--dump prefix]\n");
        return 1;
    }

    std::vector<int32_t> tokens = !tokens_file.empty() ? read_tokens_file(tokens_file) : parse_token_list(tokens_str);
    if (tokens.empty()) { fprintf(stderr, "error: no tokens\n"); return 1; }

    ouro_model m;
    std::string err;
    if (!ouro_backend_init(m, device, err))   { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!ouro_model_load(model_path, m, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const int R = loops > 0 ? loops : m.hp.n_loops;
    const ggml_type kv_type = parse_kv_type(kv_str, err);
    if (!err.empty()) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    ouro_kv_cache kv;
    if (!ouro_kv_cache_init(kv, m.backend, m.hp.n_embd_head, m.hp.n_head_kv, ctx,
                            m.hp.n_layer * R, kv_type, err)) {
        fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }

    ouro_graph_outputs out;
    if (!ouro_forward_kv(m, kv, tokens, /*n_past=*/0, R, out, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const int n = (int) tokens.size();
    const int V = m.hp.n_vocab;
    const int H = m.hp.n_embd;

    const int best = argmax_col(out.zR, n - 1);
    printf("device=%s kv=%s loops=%d n_tokens=%d argmax(zR[-1])=%d\n", device.c_str(), kv_str.c_str(), R, n, best);

    if (!dump.empty()) {
        dump_f32(dump + ".zR.bin", out.zR);
        dump_f32(dump + ".z1.bin", out.z1);
        dump_f32(dump + ".h1.bin", out.h1);
        dump_f32(dump + ".hR.bin", out.hR);
        std::ofstream j(dump + ".meta.json");
        const std::string d = json_escape(dump);   // keep M1's Windows-path escaping
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

    ouro_kv_cache_free(kv);
    ouro_model_free(m);
    return 0;
}

static int cmd_run(int argc, char ** argv) {
    std::string model_path, tokens_str, tokens_file, dump_ids, device = "vulkan", kv_str = "q8_0";
    std::string loopcd_str = "logits", loopcd_preset = "generation", dump_guided;
    bool  loopcd_adaptive = false, omega_set = false, omega_max_set = false;
    float loopcd_omega = 0.0f, loopcd_omega_max = 0.0f;
    int   loopcd_ref = 1;
    int   loops = 0, ctx = 4096, max_tokens = 32, eos = 2;

    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-m" || a == "--model")  model_path  = arg_value(argc, argv, i);
        else if (a == "--tokens")              tokens_str  = arg_value(argc, argv, i);
        else if (a == "--tokens-file")         tokens_file = arg_value(argc, argv, i);
        else if (a == "--loops")               loops       = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--ctx")                 ctx         = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--device")              device      = arg_value(argc, argv, i);
        else if (a == "--kv-type")             kv_str      = arg_value(argc, argv, i);
        else if (a == "--max-tokens")          max_tokens  = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--eos")                 eos         = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--dump-ids")            dump_ids    = arg_value(argc, argv, i);
        else if (a == "--loopcd")              loopcd_str  = arg_value(argc, argv, i);
        else if (a == "--loopcd-adaptive")     loopcd_adaptive = true;
        else if (a == "--loopcd-omega")        { loopcd_omega     = (float) std::atof(arg_value(argc, argv, i).c_str()); omega_set = true; }
        else if (a == "--loopcd-omega-max")    { loopcd_omega_max = (float) std::atof(arg_value(argc, argv, i).c_str()); omega_max_set = true; }
        else if (a == "--loopcd-ref")          loopcd_ref  = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--loopcd-preset")       loopcd_preset = arg_value(argc, argv, i);
        else if (a == "--dump-guided")         dump_guided = arg_value(argc, argv, i);
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: loopcd run -m <model.gguf> (--tokens a,b,c | --tokens-file f) "
                        "[--device cpu|vulkan] [--kv-type q8_0|f16|f32] [--ctx N] [--loops N] "
                        "[--max-tokens N] [--eos N] [--dump-ids file] "
                        "[--loopcd off|logits|hidden] [--loopcd-preset generation|reasoning|mc] "
                        "[--loopcd-omega F] [--loopcd-omega-max F] [--loopcd-adaptive] [--loopcd-ref N] "
                        "[--dump-guided prefix]\n");
        return 1;
    }

    // ---- resolve LoopCD parameters (paper Appendix B.4 presets) ----
    ouro_loopcd_params lp;
    {
        std::string lerr;
        lp.mode = parse_loopcd_mode(loopcd_str, lerr);
        if (!lerr.empty()) { fprintf(stderr, "error: %s\n", lerr.c_str()); return 1; }

        float preset_omega = 0.3f, preset_cap = 1.0f;   // generation default
        if      (loopcd_preset == "generation") { preset_omega = 0.3f; preset_cap = 1.0f; }
        else if (loopcd_preset == "reasoning")  { preset_omega = 0.5f; preset_cap = 1.5f; }
        else if (loopcd_preset == "mc")         { preset_omega = 0.5f; preset_cap = 1.0f; }
        else { fprintf(stderr, "error: unknown --loopcd-preset '%s' (generation|reasoning|mc)\n", loopcd_preset.c_str()); return 1; }

        // hidden form has its own default strength (paper D.3: Ouro-2.6B hidden peaks at omega=0.5)
        lp.omega     = omega_set     ? loopcd_omega     : (lp.mode == OURO_LOOPCD_HIDDEN ? 0.5f : preset_omega);
        lp.omega_max = omega_max_set ? loopcd_omega_max : preset_cap;
        lp.adaptive  = loopcd_adaptive;
        lp.ref_iter  = loopcd_ref;

        if (lp.mode == OURO_LOOPCD_HIDDEN && lp.adaptive) {
            fprintf(stderr, "error: adaptive strength is LoopCD-Logits only (paper 3.2)\n"); return 1;
        }
        if (lp.mode != OURO_LOOPCD_OFF && lp.ref_iter != 1) {
            fprintf(stderr, "error: only --loopcd-ref 1 is available (the M2 taps expose loop 1)\n"); return 1;
        }
    }

    std::vector<int32_t> prompt = !tokens_file.empty() ? read_tokens_file(tokens_file) : parse_token_list(tokens_str);
    if (prompt.empty()) { fprintf(stderr, "error: no tokens\n"); return 1; }
    if (max_tokens <= 0) { fprintf(stderr, "error: --max-tokens must be > 0\n"); return 1; }

    ouro_model m;
    std::string err;
    if (!ouro_backend_init(m, device, err))   { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!ouro_model_load(model_path, m, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const int R = loops > 0 ? loops : m.hp.n_loops;
    const ggml_type kv_type = parse_kv_type(kv_str, err);
    if (!err.empty()) { fprintf(stderr, "error: %s\n", err.c_str()); ouro_model_free(m); return 1; }

    if ((int) prompt.size() + max_tokens > ctx) {
        fprintf(stderr, "error: ctx=%d too small for %d prompt + %d generated tokens\n",
                ctx, (int) prompt.size(), max_tokens);
        ouro_model_free(m);
        return 1;
    }

    ouro_kv_cache kv;
    if (!ouro_kv_cache_init(kv, m.backend, m.hp.n_embd_head, m.hp.n_head_kv, ctx,
                            m.hp.n_layer * R, kv_type, err)) {
        fprintf(stderr, "error: %s\n", err.c_str()); ouro_model_free(m); return 1;
    }

    printf("device=%s kv=%s loops=%d loopcd=%s omega=%.3f omega_max=%.3f adaptive=%d ref=%d n_prompt=%d ctx=%d\n",
           device.c_str(), kv_str.c_str(), R, loopcd_str.c_str(), lp.omega, lp.omega_max,
           lp.adaptive ? 1 : 0, lp.ref_iter, (int) prompt.size(), ctx);

    ouro_graph_outputs out;

    // prefill: fill all slots' rows [0, prompt.size())
    const int64_t t_pre0 = ggml_time_us();
    if (!ouro_forward_kv(m, kv, prompt, /*n_past=*/0, R, out, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const int64_t t_pre1 = ggml_time_us();

    std::vector<int32_t> gen;
    {
        std::vector<float> gz, gu, gz1;   // guided, unguided (zR), reference (z1)
        const int tok = loopcd_guided_argmax(m, lp, out, (int) prompt.size() - 1, &gz, &gu, &gz1, err);
        if (tok < 0) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        if (!dump_guided.empty()) {
            dump_f32_vec(dump_guided + ".guided.bin",   gz);
            dump_f32_vec(dump_guided + ".unguided.bin", gu);
            dump_f32_vec(dump_guided + ".z1.bin",       gz1);
        }
        gen.push_back(tok);
    }

    // decode: one graph compute per token, appending K/V and advancing n_past
    int n_past = (int) prompt.size();
    const int64_t t_dec0 = ggml_time_us();
    while ((int) gen.size() < max_tokens && gen.back() != eos && n_past < ctx) {
        std::vector<int32_t> one = { gen.back() };
        if (!ouro_forward_kv(m, kv, one, n_past, R, out, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        const int tok = loopcd_guided_argmax(m, lp, out, 0, nullptr, nullptr, nullptr, err);
        if (tok < 0) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        n_past += 1;
        gen.push_back(tok);
    }
    const int64_t t_dec1 = ggml_time_us();

    const double pre_s = (t_pre1 - t_pre0) / 1e6;
    const double dec_s = (t_dec1 - t_dec0) / 1e6;
    printf("gen_ids:");
    for (size_t i = 0; i < gen.size(); ++i) printf("%s%d", i ? "," : "", gen[i]);
    printf("\n");
    printf("prefill: %d tok in %.2fs (%.2f tok/s)\n", (int) prompt.size(), pre_s, prompt.size() / (pre_s > 0 ? pre_s : 1e-9));
    printf("decode : %d tok in %.2fs (%.2f tok/s)\n", (int) gen.size(), dec_s, gen.size() / (dec_s > 0 ? dec_s : 1e-9));

    if (!dump_ids.empty()) {
        std::ofstream f(dump_ids);
        for (size_t i = 0; i < gen.size(); ++i) f << (i ? "," : "") << gen[i];
        f << "\n";
    }

    ouro_kv_cache_free(kv);
    ouro_model_free(m);
    return 0;
}

static int cmd_tokenize(int argc, char ** argv) {
    std::string model_path, text, ids_str;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-m" || a == "--model") model_path = arg_value(argc, argv, i);
        else if (a == "--text")               text       = arg_value(argc, argv, i);
        else if (a == "--ids")                ids_str    = arg_value(argc, argv, i);
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: loopcd tokenize -m <model.gguf> --text \"...\" | --ids a,b,c\n");
        return 1;
    }

    ouro_tokenizer tok;
    std::string err;
    if (!ouro_tokenizer_load(tok, model_path, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    if (!ids_str.empty()) {
        std::vector<int32_t> ids = parse_token_list(ids_str);
        const std::string out = ouro_detokenize(tok, ids, /*skip_special=*/false);
        printf("%s\n", out.c_str());
        return 0;
    }

    std::vector<int32_t> ids;
    if (!ouro_tokenize(tok, text, /*add_special=*/false, ids, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    printf("ids:");
    for (size_t i = 0; i < ids.size(); ++i) printf("%s%d", i ? "," : "", ids[i]);
    printf("\n");
    return 0;
}
