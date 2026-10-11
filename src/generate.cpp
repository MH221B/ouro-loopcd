#include "generate.h"
#include "kitty_kv.h"

#include <algorithm>
#include <cmath>

static void read_col(ggml_tensor * t, int col, std::vector<float> & dst) {
    const int D = (int) t->ne[0];
    dst.resize(D);
    ggml_backend_tensor_get(t, dst.data(), (size_t) col * D * sizeof(float), D * sizeof(float));
}

bool ouro_generator::begin(ouro_model & model, ouro_kv_cache & cache, const std::vector<int32_t> & p,
                           const ouro_gen_params & params, int context, std::string & err) {
    m = &model;
    kv = &cache;
    gp = params;
    ctx = context;
    prompt = p;
    R = gp.loops > 0 ? gp.loops : m->hp.n_loops;

    prefilled = false;
    done = false;
    n_past = 0;
    out_ids.clear();
    first_guided.clear();
    first_unguided.clear();
    first_z1.clear();
    rng.seed(gp.sampling.seed);

    if (prompt.empty()) { err = "generator: empty prompt"; return false; }
    if ((int) prompt.size() >= ctx) { err = "generator: prompt does not fit the context"; return false; }
    if (kv->n_slots != m->hp.n_layer * R) { err = "generator: KV slots do not match n_layer * loops"; return false; }
    return true;
}

int ouro_generator::sample_from(int col, int32_t & out, std::string & err) {
    // Read the raw taps for the dump BEFORE the hidden head pass can clobber them.
    const bool dump_col = first_guided.empty() && col == (int) prompt.size() - 1;
    if (dump_col) {
        read_col(taps.zR, col, first_unguided);
        read_col(taps.z1, col, first_z1);
    }

    std::vector<float> logits;
    if (!ouro_loopcd_guided_logits(*m, gp.loopcd, taps, col, logits, err)) return -1;
    if (dump_col) first_guided = logits;

    std::vector<int32_t> recent(prompt);
    recent.insert(recent.end(), out_ids.begin(), out_ids.end());
    out = ouro_sample(logits, recent, gp.sampling, rng);
    return out < 0 ? -1 : 0;
}

int ouro_generator::next(int32_t & out, std::string & err) {
    if (done) return 0;

    if (!prefilled) {
        if (gp.max_tokens <= 0) { done = true; return 0; }
        if (!kv->kitty) {
            // plain / q8_0 / f16: one dense prefill, exactly as before
            if (!ouro_forward_kv(*m, *kv, prompt, /*n_past=*/0, R, taps, err)) return -1;
        } else {
            const int  sink   = kv->kitty_cfg.sink;
            const bool packed = kv->kitty_cfg.packed;
            const int P = 128;
            int pos = 0;
            while (pos < (int) prompt.size()) {
                int c;
                if (pos < sink) {
                    c = std::min(sink - pos, (int) prompt.size() - pos);
                } else {
                    const int pending = pos - (sink + (int) kv->kitty_k_pages * P);
                    c = std::min(P - pending, (int) prompt.size() - pos);
                }
                const std::vector<int32_t> chunk(prompt.begin() + pos, prompt.begin() + pos + c);
                const int n_kv_new = pos + c;
                // V eviction must precede the write (its ring row is about to be reused).
                if (packed && !ouro_kv_kitty_pack_evict(*kv, n_kv_new, err)) return -1;
                if (!ouro_forward_kv(*m, *kv, chunk, /*n_past=*/pos, R, taps, err)) return -1;
                if (packed) {
                    if (!ouro_kv_kitty_pack_pages(*kv, n_kv_new, err)) return -1;
                } else {
                    if (!ouro_kv_kitty_quantize(*kv, n_kv_new, err)) return -1;
                }
                pos = n_kv_new;
            }
        }
        n_past = (int) prompt.size();
        prefilled = true;
        if (sample_from(n_past - 1, out, err) < 0) return -1;
    } else {
        if ((int) out_ids.size() >= gp.max_tokens) { done = true; return 0; }
        if (n_past >= ctx) { done = true; return 0; }
        const bool packed = kv->kitty && kv->kitty_cfg.packed;
        const std::vector<int32_t> one = { out_ids.back() };
        const int n_kv_new = n_past + 1;
        if (packed && !ouro_kv_kitty_pack_evict(*kv, n_kv_new, err)) return -1;
        if (!ouro_forward_kv(*m, *kv, one, n_past, R, taps, err)) return -1;
        if (packed) {
            if (!ouro_kv_kitty_pack_pages(*kv, n_kv_new, err)) return -1;
        } else {
            if (!ouro_kv_kitty_quantize(*kv, n_kv_new, err)) return -1;
        }
        n_past = n_kv_new;
        if (sample_from(0, out, err) < 0) return -1;
    }

    if (out == gp.eos) { done = true; return 0; }   // EOS is never emitted to the caller
    out_ids.push_back(out);
    return 1;
}
