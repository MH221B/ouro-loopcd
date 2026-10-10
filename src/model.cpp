#include "model.h"
#include "gguf.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static bool get_u32(const gguf_context * g, const std::string & key, int32_t & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) { err = "missing GGUF key: " + key; return false; }
    const enum gguf_type t = gguf_get_kv_type(g, id);
    if (t == GGUF_TYPE_UINT32)      out = (int32_t) gguf_get_val_u32(g, id);
    else if (t == GGUF_TYPE_INT32)  out = gguf_get_val_i32(g, id);
    else { err = "key is not an integer: " + key; return false; }
    return true;
}

static bool get_f32(const gguf_context * g, const std::string & key, float & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) { err = "missing GGUF key: " + key; return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) { err = "key is not f32: " + key; return false; }
    out = gguf_get_val_f32(g, id);
    return true;
}

static ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name,
                                    const int64_t * ne, int n_dims, std::string & err) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) { err = "missing tensor: " + name; return nullptr; }
    for (int i = 0; i < n_dims; ++i) {
        if (t->ne[i] != ne[i]) {
            err = "tensor " + name + " dim " + std::to_string(i) + " is " +
                  std::to_string(t->ne[i]) + ", expected " + std::to_string(ne[i]);
            return nullptr;
        }
    }
    return t;
}

bool ouro_backend_init(ouro_model & m, std::string & err) {
    ggml_backend_load_all();
    m.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!m.backend) { err = "failed to init CPU backend"; return false; }
    ggml_backend_t backends[1] = { m.backend };
    m.sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, true);
    if (!m.sched) { err = "failed to create scheduler"; return false; }
    return true;
}

bool ouro_model_load(const std::string & path, ouro_model & m, std::string & err) {
    gguf_init_params params = { /*no_alloc=*/ true, /*ctx=*/ &m.ctx_w };
    m.ctx_g = gguf_init_from_file(path.c_str(), params);
    if (!m.ctx_g || !m.ctx_w) { err = "gguf_init_from_file failed for " + path; return false; }

    const char * arch = nullptr;
    const int64_t ak = gguf_find_key(m.ctx_g, "general.architecture");
    if (ak < 0) { err = "missing key: general.architecture"; return false; }
    arch = gguf_get_val_str(m.ctx_g, ak);
    if (std::string(arch) != "ouro") { err = std::string("architecture is ") + arch + ", expected ouro"; return false; }

    auto & hp = m.hp;
    if (!get_u32(m.ctx_g, "ouro.block_count",                    hp.n_layer,     err)) return false;
    if (!get_u32(m.ctx_g, "ouro.embedding_length",               hp.n_embd,      err)) return false;
    if (!get_u32(m.ctx_g, "ouro.feed_forward_length",            hp.n_ff,        err)) return false;
    if (!get_u32(m.ctx_g, "ouro.attention.head_count",           hp.n_head,      err)) return false;
    if (!get_u32(m.ctx_g, "ouro.attention.head_count_kv",        hp.n_head_kv,   err)) return false;
    if (!get_u32(m.ctx_g, "ouro.attention.key_length",           hp.n_embd_head, err)) return false;
    if (!get_u32(m.ctx_g, "ouro.context_length",                 hp.n_ctx,       err)) return false;
    if (!get_u32(m.ctx_g, "ouro.vocab_size",                     hp.n_vocab,     err)) return false;
    if (!get_u32(m.ctx_g, "ouro.num_loops",                      hp.n_loops,     err)) return false;
    if (!get_f32(m.ctx_g, "ouro.rope.freq_base",                 hp.rope_theta,  err)) return false;
    if (!get_f32(m.ctx_g, "ouro.attention.layer_norm_rms_epsilon", hp.eps,       err)) return false;

    m.buf_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    if (!m.buf_w) { err = "failed to allocate weight buffer"; return false; }

    const int64_t qd = (int64_t) hp.n_embd_head * hp.n_head;      // 2048
    const int64_t kv = (int64_t) hp.n_embd_head * hp.n_head_kv;   // 2048

    { const int64_t ne[] = { hp.n_embd, hp.n_vocab };
      m.tok_embd = require_tensor(m.ctx_w, "token_embd.weight", ne, 2, err); if (!m.tok_embd) return false; }
    { const int64_t ne[] = { hp.n_embd };
      m.output_norm = require_tensor(m.ctx_w, "output_norm.weight", ne, 1, err); if (!m.output_norm) return false; }
    { const int64_t ne[] = { hp.n_embd, hp.n_vocab };
      m.output = require_tensor(m.ctx_w, "output.weight", ne, 2, err); if (!m.output) return false; }

    m.layers.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
        auto & L = m.layers[i];
        auto tn = [&](const char * suffix) { return "blk." + std::to_string(i) + "." + suffix; };
        { const int64_t ne[] = { hp.n_embd };
          L.attn_norm = require_tensor(m.ctx_w, tn("attn_norm.weight"), ne, 1, err); if (!L.attn_norm) return false; }
        { const int64_t ne[] = { hp.n_embd, qd };
          L.attn_q = require_tensor(m.ctx_w, tn("attn_q.weight"), ne, 2, err); if (!L.attn_q) return false; }
        { const int64_t ne[] = { hp.n_embd, kv };
          L.attn_k = require_tensor(m.ctx_w, tn("attn_k.weight"), ne, 2, err); if (!L.attn_k) return false; }
        { const int64_t ne[] = { hp.n_embd, kv };
          L.attn_v = require_tensor(m.ctx_w, tn("attn_v.weight"), ne, 2, err); if (!L.attn_v) return false; }
        { const int64_t ne[] = { qd, hp.n_embd };
          L.attn_output = require_tensor(m.ctx_w, tn("attn_output.weight"), ne, 2, err); if (!L.attn_output) return false; }
        { const int64_t ne[] = { hp.n_embd };
          L.attn_post_norm = require_tensor(m.ctx_w, tn("post_attention_norm.weight"), ne, 1, err); if (!L.attn_post_norm) return false; }
        { const int64_t ne[] = { hp.n_embd };
          L.ffn_norm = require_tensor(m.ctx_w, tn("ffn_norm.weight"), ne, 1, err); if (!L.ffn_norm) return false; }
        { const int64_t ne[] = { hp.n_embd, hp.n_ff };
          L.ffn_gate = require_tensor(m.ctx_w, tn("ffn_gate.weight"), ne, 2, err); if (!L.ffn_gate) return false; }
        { const int64_t ne[] = { hp.n_embd, hp.n_ff };
          L.ffn_up = require_tensor(m.ctx_w, tn("ffn_up.weight"), ne, 2, err); if (!L.ffn_up) return false; }
        { const int64_t ne[] = { hp.n_ff, hp.n_embd };
          L.ffn_down = require_tensor(m.ctx_w, tn("ffn_down.weight"), ne, 2, err); if (!L.ffn_down) return false; }
        { const int64_t ne[] = { hp.n_embd };
          L.ffn_post_norm = require_tensor(m.ctx_w, tn("post_ffw_norm.weight"), ne, 1, err); if (!L.ffn_post_norm) return false; }
    }

    // Upload raw tensor bytes into the backend buffer.
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot reopen " + path; return false; }

    const int64_t n_tensors = gguf_get_n_tensors(m.ctx_g);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(m.ctx_g, i);
        ggml_tensor * t = ggml_get_tensor(m.ctx_w, name);
        if (!t) continue;
        const int64_t offs = (int64_t) (gguf_get_data_offset(m.ctx_g) + gguf_get_tensor_offset(m.ctx_g, i));
        buf.resize(ggml_nbytes(t));
        if (_fseeki64(f, offs, SEEK_SET) != 0 || fread(buf.data(), 1, buf.size(), f) != buf.size()) {
            fclose(f);
            err = std::string("failed to read data for tensor ") + name;
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
    }
    fclose(f);

    return true;
}

void ouro_model_free(ouro_model & m) {
    if (m.sched)   { ggml_backend_sched_free(m.sched);   m.sched = nullptr; }
    if (m.buf_w)   { ggml_backend_buffer_free(m.buf_w); m.buf_w = nullptr; }
    if (m.ctx_w)   { ggml_free(m.ctx_w);                 m.ctx_w = nullptr; }
    if (m.ctx_g)   { gguf_free(m.ctx_g);                 m.ctx_g = nullptr; }
    if (m.backend) { ggml_backend_free(m.backend);       m.backend = nullptr; }
}
