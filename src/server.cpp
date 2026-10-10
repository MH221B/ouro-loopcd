#include "server.h"

#include "chat_template.h"
#include "generate.h"
#include "graph.h"
#include "kv_cache.h"
#include "model.h"
#include "tokenizer.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

static const char * k_model_id = "ouro-2.6b-thinking";

static json err_body(const std::string & msg, const std::string & type) {
    return json{{"error", {{"message", msg}, {"type", type}, {"code", nullptr}}}};
}

static ouro_loopcd_mode loopcd_mode_from_str(const std::string & s) {
    if (s == "off")    return OURO_LOOPCD_OFF;
    if (s == "hidden") return OURO_LOOPCD_HIDDEN;
    return OURO_LOOPCD_LOGITS;
}

// Detokenizes one token for streaming; special tokens are dropped.
static std::string piece_of(const ouro_tokenizer & tok, int32_t id) {
    return ouro_detokenize(tok, { id }, /*skip_special=*/true);
}

struct server_state {
    ouro_model     m;
    ouro_kv_cache  kv;
    ouro_tokenizer tok;
    ouro_server_config cfg;
    std::mutex     mu;   // serializes generations (also enforced by the 1-thread task queue)
};

// Incremental text with stop-string hold-back (the last (max_stop-1) bytes are held).
struct stream_text {
    std::vector<std::string> stops;
    size_t max_stop = 0;
    std::string all;          // everything generated so far
    size_t emitted = 0;       // bytes already handed to the client

    void init(const std::vector<std::string> & s) {
        stops = s;
        for (const auto & x : stops) max_stop = std::max(max_stop, x.size());
    }
    // Returns (new_text_to_emit, stopped). Appends `piece` to `all`.
    std::pair<std::string, bool> push(const std::string & piece) {
        all += piece;
        size_t stop_at = std::string::npos;
        for (const auto & s : stops) {
            const size_t p = all.find(s);
            if (p != std::string::npos) { stop_at = std::min(stop_at, p); }
        }
        const bool stopped = stop_at != std::string::npos;
        size_t cut = stopped ? stop_at : all.size();
        if (!stopped && max_stop > 0 && all.size() >= max_stop - 1) {
            cut = all.size() - (max_stop - 1);   // hold back a possible partial stop
        }
        if (cut < emitted) cut = emitted;
        const std::string out = all.substr(emitted, cut - emitted);
        emitted = cut;
        return { out, stopped };
    }
    std::string finish() {
        const std::string out = all.substr(emitted);
        emitted = all.size();
        return out;
    }
};

static json chat_chunk(const char * id, const std::string & model, const json & delta, const char * finish) {
    return json{
        {"id", id}, {"object", "chat.completion.chunk"}, {"created", 0}, {"model", model},
        {"choices", json::array({ json{{"index", 0}, {"delta", delta}, {"finish_reason", finish ? json(finish) : json(nullptr)}} })}
    };
}

int ouro_serve(const ouro_server_config & cfg) {
    auto st = std::make_shared<server_state>();
    st->cfg = cfg;

    std::string err;
    if (!ouro_backend_init(st->m, cfg.device, err))   { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!ouro_model_load(cfg.model_path, st->m, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!ouro_tokenizer_load(st->tok, cfg.model_path, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    const int R = cfg.loops > 0 ? cfg.loops : st->m.hp.n_loops;
    const bool kitty = cfg.kitty || cfg.kv_type == "kitty";
    const ggml_type kv_type = (kitty || cfg.kv_type == "f16") ? GGML_TYPE_F16
                            : cfg.kv_type == "f32" ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
    ouro_kitty_config kcfg = cfg.kitty_cfg;
    if (!ouro_kv_cache_init(st->kv, st->m.backend, st->m.hp.n_embd_head, st->m.hp.n_head_kv,
                            cfg.ctx, st->m.hp.n_layer * R, kv_type, kitty ? &kcfg : nullptr, err)) {
        fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }

    httplib::Server svr;
    svr.new_task_queue = [] { return new httplib::ThreadPool(1); };   // serial serving

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        json data = json::array({ json{{"id", k_model_id}, {"object", "model"}, {"created", 0}, {"owned_by", "loopcd"}} });
        res.set_content(json{{"object", "list"}, {"data", data}}.dump(), "application/json");
    });

    auto make_gen_params = [&](const json & body) {
        ouro_gen_params gp;
        gp.loopcd = st->cfg.loopcd;
        gp.sampling = st->cfg.sampling;
        gp.loops = cfg.loops;
        gp.eos = cfg.eos;
        gp.max_tokens = body.value("max_tokens", 64);
        gp.sampling.temp           = body.value("temperature", gp.sampling.temp);
        gp.sampling.top_p          = body.value("top_p", gp.sampling.top_p);
        gp.sampling.top_k          = body.value("top_k", gp.sampling.top_k);
        gp.sampling.repeat_penalty = body.value("repetition_penalty", gp.sampling.repeat_penalty);
        gp.sampling.seed           = (uint32_t) body.value("seed", (int) gp.sampling.seed);
        // Per-request LoopCD override. Default mode is `logits` (server default). NOTE: on the quantized
        // shipping stack (Q4_K_M + q8_0) the `hidden` form degenerates to a repeating token (M3
        // observation); it is only well-behaved on f32 weights + f32 KV.
        if (body.contains("extra_body") && body["extra_body"].is_object() && body["extra_body"].contains("loopcd")) {
            const json & lj = body["extra_body"]["loopcd"];
            if (lj.is_object()) {
                if (lj.contains("mode"))       gp.loopcd.mode = loopcd_mode_from_str(lj["mode"].get<std::string>());
                if (lj.contains("omega"))      gp.loopcd.omega = lj["omega"].get<float>();
                if (lj.contains("omega_max"))  gp.loopcd.omega_max = lj["omega_max"].get<float>();
                if (lj.contains("adaptive"))   gp.loopcd.adaptive = lj["adaptive"].get<bool>();
            }
        }
        return gp;
    };

    auto run_request = [&](const json & body, std::vector<int32_t> & prompt_ids,
                           std::vector<int32_t> & out_ids, std::string & out_err) -> bool {
        ouro_gen_params gp = make_gen_params(body);
        const std::vector<int32_t> prompt = prompt_ids;
        if ((int) prompt.size() >= cfg.ctx) { out_err = "context_length_exceeded: prompt does not fit ctx"; return false; }
        std::lock_guard<std::mutex> lock(st->mu);
        ouro_generator gen;
        if (!gen.begin(st->m, st->kv, prompt, gp, cfg.ctx, out_err)) return false;
        while (true) {
            int32_t tok = -1;
            const int r = gen.next(tok, out_err);
            if (r < 0) return false;
            if (r == 0) break;
            out_ids.push_back(tok);
        }
        return true;
    };

    auto parse_stops = [](const json & body, std::vector<std::string> & stops) {
        if (!body.contains("stop")) return;
        const json & s = body["stop"];
        if (s.is_string()) stops.push_back(s.get<std::string>());
        else if (s.is_array()) for (const auto & x : s) if (x.is_string()) stops.push_back(x.get<std::string>());
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(err_body("invalid JSON body", "invalid_request_error").dump(), "application/json"); return; }
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            res.status = 400; res.set_content(err_body("missing 'messages'", "invalid_request_error").dump(), "application/json"); return;
        }
        std::vector<ouro_chat_message> msgs;
        for (const auto & m : body["messages"]) {
            const std::string role = m.value("role", "user");
            const std::string content = (m.contains("content") && m["content"].is_string())
                                      ? m["content"].get<std::string>() : std::string();
            msgs.push_back({ role, content });
        }
        bool enable_thinking = body.value("enable_thinking", false);
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            enable_thinking = body["chat_template_kwargs"].value("enable_thinking", enable_thinking);
        }
        const std::string prompt = ouro_render_chat(msgs, /*add_generation_prompt=*/true, enable_thinking);

        std::vector<int32_t> ids;
        std::string err;
        if (!ouro_tokenize(st->tok, prompt, /*add_special=*/false, ids, err)) {
            res.status = 400; res.set_content(err_body(err, "invalid_request_error").dump(), "application/json"); return;
        }
        if ((int) ids.size() >= cfg.ctx) {
            res.status = 400;
            res.set_content(err_body("context_length_exceeded: " + std::to_string(ids.size()) + " tokens >= ctx " + std::to_string(cfg.ctx), "invalid_request_error").dump(), "application/json");
            return;
        }

        const bool stream = body.value("stream", false);

        if (!stream) {
            std::vector<int32_t> out_ids;
            if (!run_request(body, ids, out_ids, err)) { res.status = 500; res.set_content(err_body(err, "server_error").dump(), "application/json"); return; }
            std::vector<std::string> stops; parse_stops(body, stops);
            std::string text = ouro_detokenize(st->tok, out_ids);
            for (const auto & s : stops) { const size_t p = text.find(s); if (p != std::string::npos) text = text.substr(0, p); }
            json choice = json{{"index", 0}, {"message", json{{"role", "assistant"}, {"content", text}}}, {"finish_reason", "stop"}};
            json resp = json{{"id", "chatcmpl-loopcd"}, {"object", "chat.completion"}, {"created", 0},
                             {"model", k_model_id}, {"choices", json::array({choice})},
                             {"usage", json{{"prompt_tokens", ids.size()}, {"completion_tokens", out_ids.size()}, {"total_tokens", ids.size() + out_ids.size()}}}};
            res.set_content(resp.dump(), "application/json");
            return;
        }

        // streaming: one token per chunked-provider callback
        auto gen = std::make_shared<ouro_generator>();
        {
            std::lock_guard<std::mutex> lock(st->mu);
            if (!gen->begin(st->m, st->kv, ids, make_gen_params(body), cfg.ctx, err)) {
                res.status = 500; res.set_content(err_body(err, "server_error").dump(), "application/json"); return;
            }
        }
        auto stext = std::make_shared<stream_text>();
        { std::vector<std::string> stops; parse_stops(body, stops); stext->init(stops); }

        res.set_chunked_content_provider("text/event-stream",
            [st, gen, stext](size_t, httplib::DataSink & sink) -> bool {
                int32_t tok = -1;
                std::string e;
                std::lock_guard<std::mutex> lock(st->mu);
                const int r = gen->next(tok, e);
                if (r < 0) { const std::string f = "data: [DONE]\n\n"; sink.write(f.data(), f.size()); sink.done(); return false; }
                if (r == 0) {
                    const std::string tail = stext->finish();
                    if (!tail.empty()) { const json c = chat_chunk("chatcmpl-loopcd", k_model_id, json{{"content", tail}}, nullptr); const std::string frame = "data: " + c.dump() + "\n\n"; sink.write(frame.data(), frame.size()); }
                    const json last = chat_chunk("chatcmpl-loopcd", k_model_id, json::object(), "stop");
                    const std::string lf = "data: " + last.dump() + "\n\ndata: [DONE]\n\n"; sink.write(lf.data(), lf.size()); sink.done(); return false;
                }
                const std::string piece = piece_of(st->tok, tok);
                auto [emit, stopped] = stext->push(piece);
                if (!emit.empty()) { const json c = chat_chunk("chatcmpl-loopcd", k_model_id, json{{"content", emit}}, nullptr); const std::string frame = "data: " + c.dump() + "\n\n"; sink.write(frame.data(), frame.size()); }
                return true;
            });
    });

    svr.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(err_body("invalid JSON body", "invalid_request_error").dump(), "application/json"); return; }
        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            res.status = 400; res.set_content(err_body("missing 'prompt'", "invalid_request_error").dump(), "application/json"); return;
        }
        std::vector<int32_t> ids;
        std::string err;
        if (!ouro_tokenize(st->tok, body["prompt"].get<std::string>(), /*add_special=*/false, ids, err)) {
            res.status = 400; res.set_content(err_body(err, "invalid_request_error").dump(), "application/json"); return;
        }
        if ((int) ids.size() >= cfg.ctx) { res.status = 400; res.set_content(err_body("context_length_exceeded", "invalid_request_error").dump(), "application/json"); return; }
        std::vector<int32_t> out_ids;
        if (!run_request(body, ids, out_ids, err)) { res.status = 500; res.set_content(err_body(err, "server_error").dump(), "application/json"); return; }
        std::vector<std::string> stops; parse_stops(body, stops);
        std::string text = ouro_detokenize(st->tok, out_ids);
        for (const auto & s : stops) { const size_t p = text.find(s); if (p != std::string::npos) text = text.substr(0, p); }
        json choice = json{{"index", 0}, {"text", text}, {"finish_reason", "stop"}};
        json resp = json{{"id", "cmpl-loopcd"}, {"object", "text_completion"}, {"created", 0},
                         {"model", k_model_id}, {"choices", json::array({choice})},
                         {"usage", json{{"prompt_tokens", ids.size()}, {"completion_tokens", out_ids.size()}, {"total_tokens", ids.size() + out_ids.size()}}}};
        res.set_content(resp.dump(), "application/json");
    });

    fprintf(stderr, "loopcd serving %s on http://%s:%d (ctx=%d, serial)\n",
            k_model_id, cfg.host.c_str(), cfg.port, cfg.ctx);
    if (!svr.listen(cfg.host.c_str(), cfg.port)) { fprintf(stderr, "error: listen failed on %s:%d\n", cfg.host.c_str(), cfg.port); return 1; }
    return 0;
}
