#include "model.h"
#include "graph.h"
#include "loopcd.h"
#include "tokenizer.h"
#include "chat_template.h"
#include "generate.h"
#include "sampling.h"
#include "server.h"
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

static void dump_f32_vec(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(float)));
}

static int cmd_logits(int argc, char ** argv);
static int cmd_run(int argc, char ** argv);
static int cmd_tokenize(int argc, char ** argv);
static int cmd_render(int argc, char ** argv);
static int cmd_serve(int argc, char ** argv);

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

    if (cmd == "render") {
        return cmd_render(argc - 2, argv + 2);
    }

    if (cmd == "serve") {
        return cmd_serve(argc - 2, argv + 2);
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
    float temp = 0.0f, top_p = 1.0f, repeat_penalty = 1.0f;
    int   top_k = 0, repeat_last_n = 64;
    uint32_t seed = 1234;
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
        else if (a == "--temp")                temp        = (float) std::atof(arg_value(argc, argv, i).c_str());
        else if (a == "--top-k")               top_k       = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--top-p")               top_p       = (float) std::atof(arg_value(argc, argv, i).c_str());
        else if (a == "--repeat-penalty")      repeat_penalty = (float) std::atof(arg_value(argc, argv, i).c_str());
        else if (a == "--repeat-last-n")       repeat_last_n = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--seed")                seed        = (uint32_t) std::strtoul(arg_value(argc, argv, i).c_str(), nullptr, 10);
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: loopcd run -m <model.gguf> (--tokens a,b,c | --tokens-file f) "
                        "[--device cpu|vulkan] [--kv-type q8_0|f16|f32] [--ctx N] [--loops N] "
                        "[--max-tokens N] [--eos N] [--dump-ids file] [--temp F] [--top-k N] [--top-p F] "
                        "[--repeat-penalty F] [--repeat-last-n N] [--seed N] "
                        "[--loopcd off|logits|hidden] [--loopcd-preset generation|reasoning|mc] "
                        "[--loopcd-omega F] [--loopcd-omega-max F] [--loopcd-adaptive] [--loopcd-ref N] "
                        "[--dump-guided prefix]\n");
        return 1;
    }

    ouro_gen_params gp;
    {
        std::string lerr;
        gp.loopcd.mode = parse_loopcd_mode(loopcd_str, lerr);
        if (!lerr.empty()) { fprintf(stderr, "error: %s\n", lerr.c_str()); return 1; }

        float preset_omega = 0.3f, preset_cap = 1.0f;
        if      (loopcd_preset == "generation") { preset_omega = 0.3f; preset_cap = 1.0f; }
        else if (loopcd_preset == "reasoning")  { preset_omega = 0.5f; preset_cap = 1.5f; }
        else if (loopcd_preset == "mc")         { preset_omega = 0.5f; preset_cap = 1.0f; }
        else { fprintf(stderr, "error: unknown --loopcd-preset '%s'\n", loopcd_preset.c_str()); return 1; }

        gp.loopcd.omega     = omega_set     ? loopcd_omega     : (gp.loopcd.mode == OURO_LOOPCD_HIDDEN ? 0.5f : preset_omega);
        gp.loopcd.omega_max = omega_max_set ? loopcd_omega_max : preset_cap;
        gp.loopcd.adaptive  = loopcd_adaptive;
        gp.loopcd.ref_iter  = loopcd_ref;
        if (gp.loopcd.mode == OURO_LOOPCD_HIDDEN && gp.loopcd.adaptive) {
            fprintf(stderr, "error: adaptive strength is LoopCD-Logits only (paper 3.2)\n"); return 1;
        }
        if (gp.loopcd.mode != OURO_LOOPCD_OFF && gp.loopcd.ref_iter != 1) {
            fprintf(stderr, "error: only --loopcd-ref 1 is available (the M2 taps expose loop 1)\n"); return 1;
        }
        gp.sampling.temp           = temp;
        gp.sampling.top_k          = top_k;
        gp.sampling.top_p          = top_p;
        gp.sampling.repeat_penalty = repeat_penalty;
        gp.sampling.repeat_last_n  = repeat_last_n;
        gp.sampling.seed           = seed;
        gp.loops                   = loops;
        gp.eos                     = eos;
        gp.max_tokens              = max_tokens;
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

    printf("device=%s kv=%s loops=%d loopcd=%s omega=%.3f omega_max=%.3f adaptive=%d ref=%d "
           "temp=%.3f top_k=%d top_p=%.3f repeat_penalty=%.3f seed=%u n_prompt=%d ctx=%d\n",
           device.c_str(), kv_str.c_str(), R, loopcd_str.c_str(), gp.loopcd.omega, gp.loopcd.omega_max,
           gp.loopcd.adaptive ? 1 : 0, gp.loopcd.ref_iter, gp.sampling.temp, gp.sampling.top_k,
           gp.sampling.top_p, gp.sampling.repeat_penalty, gp.sampling.seed, (int) prompt.size(), ctx);

    ouro_generator gen;
    if (!gen.begin(m, kv, prompt, gp, ctx, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    std::vector<int32_t> out_ids;
    const int64_t t0 = ggml_time_us();
    while (true) {
        int32_t tok = -1;
        const int r = gen.next(tok, err);
        if (r < 0) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        if (r == 0) break;
        out_ids.push_back(tok);
    }
    const int64_t t1 = ggml_time_us();

    printf("gen_ids:");
    for (size_t i = 0; i < out_ids.size(); ++i) printf("%s%d", i ? "," : "", out_ids[i]);
    printf("\n");
    printf("decode : %d tok in %.2fs (%.2f tok/s)\n", (int) out_ids.size(), (t1 - t0) / 1e6,
           out_ids.size() / ((t1 - t0) > 0 ? (t1 - t0) / 1e6 : 1e-9));

    if (!dump_guided.empty()) {
        dump_f32_vec(dump_guided + ".guided.bin",   gen.first_guided);
        dump_f32_vec(dump_guided + ".unguided.bin", gen.first_unguided);
        dump_f32_vec(dump_guided + ".z1.bin",       gen.first_z1);
    }
    if (!dump_ids.empty()) {
        std::ofstream f(dump_ids);
        for (size_t i = 0; i < out_ids.size(); ++i) f << (i ? "," : "") << out_ids[i];
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

static int cmd_render(int argc, char ** argv) {
    std::vector<ouro_chat_message> msgs;
    bool no_think = false;
    int add_gen = 1;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--msg") {
            const std::string v = arg_value(argc, argv, i);
            const size_t eq = v.find('=');
            if (eq == std::string::npos) { fprintf(stderr, "error: --msg must be role=content\n"); return 1; }
            msgs.push_back({ v.substr(0, eq), v.substr(eq + 1) });
        } else if (a == "--no-generation-prompt") add_gen = 0;
        else if (a == "--no-think")               no_think = true;
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (msgs.empty()) { fprintf(stderr, "usage: loopcd render --msg role=content [--msg ...] [--no-think]\n"); return 1; }
    printf("%s", ouro_render_chat(msgs, add_gen != 0, !no_think).c_str());
    return 0;
}

static int cmd_serve(int argc, char ** argv) {
    ouro_server_config cfg;
    std::string loopcd_str = "logits", loopcd_preset = "generation";
    bool omega_set = false, omega_max_set = false, loopcd_adaptive = false;
    float omega = 0.0f, omega_max = 0.0f;
    int loopcd_ref = 1;

    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "-m" || a == "--model")  cfg.model_path = arg_value(argc, argv, i);
        else if (a == "--device")              cfg.device     = arg_value(argc, argv, i);
        else if (a == "--kv-type")             cfg.kv_type    = arg_value(argc, argv, i);
        else if (a == "--ctx")                 cfg.ctx        = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--loops")               cfg.loops      = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--eos")                 cfg.eos        = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--host")                cfg.host       = arg_value(argc, argv, i);
        else if (a == "--port")                cfg.port       = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--temp")                cfg.sampling.temp           = (float) std::atof(arg_value(argc, argv, i).c_str());
        else if (a == "--top-k")               cfg.sampling.top_k          = std::atoi(arg_value(argc, argv, i).c_str());
        else if (a == "--top-p")               cfg.sampling.top_p          = (float) std::atof(arg_value(argc, argv, i).c_str());
        else if (a == "--repeat-penalty")      cfg.sampling.repeat_penalty = (float) std::atof(arg_value(argc, argv, i).c_str());
        else if (a == "--loopcd")              loopcd_str  = arg_value(argc, argv, i);
        else if (a == "--loopcd-preset")       loopcd_preset = arg_value(argc, argv, i);
        else if (a == "--loopcd-adaptive")     loopcd_adaptive = true;
        else if (a == "--loopcd-omega")        { omega = (float) std::atof(arg_value(argc, argv, i).c_str()); omega_set = true; }
        else if (a == "--loopcd-omega-max")    { omega_max = (float) std::atof(arg_value(argc, argv, i).c_str()); omega_max_set = true; }
        else if (a == "--loopcd-ref")          loopcd_ref = std::atoi(arg_value(argc, argv, i).c_str());
        else { fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); return 1; }
    }
    if (cfg.model_path.empty()) {
        fprintf(stderr, "usage: loopcd serve -m <model.gguf> [--device cpu|vulkan] [--kv-type q8_0|f16|f32] "
                        "[--ctx N] [--loops N] [--host H] [--port N] [--temp F] [--top-k N] [--top-p F] "
                        "[--repeat-penalty F] [--loopcd off|logits|hidden] [--loopcd-preset generation|reasoning|mc] "
                        "[--loopcd-omega F] [--loopcd-omega-max F] [--loopcd-adaptive] [--loopcd-ref N]\n");
        return 1;
    }

    std::string lerr;
    cfg.loopcd.mode = parse_loopcd_mode(loopcd_str, lerr);
    if (!lerr.empty()) { fprintf(stderr, "error: %s\n", lerr.c_str()); return 1; }
    float preset_omega = 0.3f, preset_cap = 1.0f;
    if      (loopcd_preset == "generation") { preset_omega = 0.3f; preset_cap = 1.0f; }
    else if (loopcd_preset == "reasoning")  { preset_omega = 0.5f; preset_cap = 1.5f; }
    else if (loopcd_preset == "mc")         { preset_omega = 0.5f; preset_cap = 1.0f; }
    else { fprintf(stderr, "error: unknown --loopcd-preset '%s'\n", loopcd_preset.c_str()); return 1; }
    cfg.loopcd.omega     = omega_set     ? omega     : (cfg.loopcd.mode == OURO_LOOPCD_HIDDEN ? 0.5f : preset_omega);
    cfg.loopcd.omega_max = omega_max_set ? omega_max : preset_cap;
    cfg.loopcd.adaptive  = loopcd_adaptive;
    cfg.loopcd.ref_iter  = loopcd_ref;
    // Keep the same hard rejections and exact messages as the landed M3 `run` command.
    if (cfg.loopcd.mode == OURO_LOOPCD_HIDDEN && cfg.loopcd.adaptive) {
        fprintf(stderr, "error: adaptive strength is LoopCD-Logits only (paper 3.2)\n"); return 1;
    }
    if (cfg.loopcd.mode != OURO_LOOPCD_OFF && cfg.loopcd.ref_iter != 1) {
        fprintf(stderr, "error: only --loopcd-ref 1 is available (the M2 taps expose loop 1)\n"); return 1;
    }
    return ouro_serve(cfg);
}
