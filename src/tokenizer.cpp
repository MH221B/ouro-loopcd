#include "tokenizer.h"
#include "unicode.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

// ---------------------------------------------------------------- GGUF metadata helpers
static bool kv_int(const gguf_context * g, const char * key, int32_t & out, bool required, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { if (required) err = std::string("missing tokenizer key: ") + key; return !required; }
    const enum gguf_type t = gguf_get_kv_type(g, id);
    if (t == GGUF_TYPE_UINT32)     out = (int32_t) gguf_get_val_u32(g, id);
    else if (t == GGUF_TYPE_INT32) out = gguf_get_val_i32(g, id);
    else { err = std::string("tokenizer key is not an integer: ") + key; return false; }
    return true;
}
static bool kv_bool(const gguf_context * g, const char * key, bool & out, bool required, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { if (required) err = std::string("missing tokenizer key: ") + key; return !required; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) { err = std::string("tokenizer key is not a bool: ") + key; return false; }
    out = gguf_get_val_bool(g, id);
    return true;
}
static bool kv_str(const gguf_context * g, const char * key, std::string & out, bool required, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { if (required) err = std::string("missing tokenizer key: ") + key; return !required; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) { err = std::string("tokenizer key is not a string: ") + key; return false; }
    out = gguf_get_val_str(g, id);
    return true;
}

// ---------------------------------------------------------------- BPE core (ported from llama.cpp)
namespace {

struct bpe_symbol { int prev, next; const char * text; size_t n; };

struct bpe_bigram { int left, right, rank; std::string text; };
struct bpe_bigram_cmp {
    bool operator()(const bpe_bigram & a, const bpe_bigram & b) const {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    }
};

int find_bpe_rank(const ouro_tokenizer & t, const std::string & l, const std::string & r) {
    auto it = t.bpe_ranks.find(std::make_pair(l, r));
    return it == t.bpe_ranks.end() ? -1 : it->second;
}

void bpe_encode_word(const ouro_tokenizer & t, const std::string & word, std::vector<int32_t> & out) {
    std::vector<bpe_symbol> syms;
    int index = 0;
    size_t off = 0;
    while (off < word.size()) {
        const size_t len = std::min(word.size() - off, (size_t) unicode_len_utf8(word[off]));
        syms.push_back({ index - 1, (off + len == word.size()) ? -1 : index + 1, word.c_str() + off, len });
        off += len;
        ++index;
    }

    std::priority_queue<bpe_bigram, std::vector<bpe_bigram>, bpe_bigram_cmp> q;
    auto add_bigram = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        const std::string lt(syms[l].text, syms[l].n), rt(syms[r].text, syms[r].n);
        const int rank = find_bpe_rank(t, lt, rt);
        if (rank < 0) return;
        q.push({ l, r, rank, lt + rt });
    };
    for (int i = 1; i < (int) syms.size(); ++i) add_bigram(i - 1, i);

    while (!q.empty()) {
        bpe_bigram b = q.top(); q.pop();
        auto & L = syms[b.left];
        auto & R = syms[b.right];
        if (L.n == 0 || R.n == 0) continue;
        if (std::string(L.text, L.n) + std::string(R.text, R.n) != b.text) continue;  // stale
        L.n += R.n;
        R.n = 0;
        L.next = R.next;
        if (R.next >= 0) syms[R.next].prev = b.left;
        add_bigram(L.prev, b.left);
        add_bigram(b.left, L.next);
    }

    for (int i = 0; i != -1; i = syms[i].next) {
        if (syms[i].n == 0) continue;
        const std::string str(syms[i].text, syms[i].n);
        auto it = t.token_to_id.find(str);
        if (it != t.token_to_id.end()) { out.push_back(it->second); continue; }
        for (char c : str) {                                   // byte fallback (rare for GPT-2)
            auto bit = t.token_to_id.find(std::string(1, c));
            if (bit != t.token_to_id.end()) out.push_back(bit->second);
        }
    }
}

// smollm pre-tokenizer: individual digits, then the GPT-2 regex (matches tokenizer.json
// Digits(individual_digits=true) + ByteLevel(use_regex=true)).
const std::vector<std::string> & smollm_regexes() {
    static const std::vector<std::string> r = {
        "\\p{N}",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    };
    return r;
}

void bpe_encode_raw(const ouro_tokenizer & t, const std::string & text, std::vector<int32_t> & out) {
    for (const std::string & w : unicode_regex_split(text, smollm_regexes(), /*byte_encode=*/ true)) {
        bpe_encode_word(t, w, out);
    }
}

} // namespace

// ---------------------------------------------------------------- public API
bool ouro_tokenizer_load(ouro_tokenizer & t, const std::string & gguf_path, std::string & err) {
    gguf_init_params p = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), p);
    if (!g) { err = "tokenizer: cannot open " + gguf_path; return false; }

    const int64_t tok_id = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tok_id < 0) { err = "missing tokenizer.ggml.tokens"; gguf_free(g); return false; }
    const size_t n = gguf_get_arr_n(g, tok_id);
    t.id_to_piece.resize(n);
    t.token_to_id.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        std::string s = gguf_get_arr_str(g, tok_id, i);
        t.token_to_id.emplace(s, (int32_t) i);
        t.id_to_piece[i] = std::move(s);
    }

    const int64_t mrg_id = gguf_find_key(g, "tokenizer.ggml.merges");
    if (mrg_id >= 0) {
        const size_t m = gguf_get_arr_n(g, mrg_id);
        for (size_t i = 0; i < m; ++i) {
            const std::string w = gguf_get_arr_str(g, mrg_id, i);
            const size_t pos = w.find(' ', 1);
            if (pos != std::string::npos) {
                t.bpe_ranks.emplace(std::make_pair(w.substr(0, pos), w.substr(pos + 1)), (int) i);
            }
        }
    }

    t.is_special.assign(n, 0);
    const int64_t typ_id = gguf_find_key(g, "tokenizer.ggml.token_type");
    if (typ_id >= 0 && gguf_get_arr_n(g, typ_id) == n) {
        const int32_t * tt = (const int32_t *) gguf_get_arr_data(g, typ_id);
        for (size_t i = 0; i < n; ++i) {
            if (tt[i] != 1) { t.is_special[i] = 1; t.special_ids.push_back((int32_t) i); }  // 1 = NORMAL
        }
    }
    std::sort(t.special_ids.begin(), t.special_ids.end(),
              [&](int32_t a, int32_t b) { return t.id_to_piece[a].size() > t.id_to_piece[b].size(); });

    if (!kv_int (g, "tokenizer.ggml.bos_token_id",     t.bos_id, false, err) ||
        !kv_int (g, "tokenizer.ggml.eos_token_id",     t.eos_id, false, err) ||
        !kv_int (g, "tokenizer.ggml.unknown_token_id", t.unk_id, false, err) ||
        !kv_bool(g, "tokenizer.ggml.add_bos_token",    t.add_bos, false, err) ||
        !kv_bool(g, "tokenizer.ggml.add_eos_token",    t.add_eos, false, err) ||
        !kv_bool(g, "tokenizer.ggml.add_space_prefix", t.add_space_prefix, false, err) ||
        !kv_str (g, "tokenizer.ggml.pre",              t.pre, false, err) ||
        !kv_str (g, "tokenizer.chat_template",         t.chat_template, false, err)) {
        gguf_free(g);
        return false;
    }
    gguf_free(g);

    if (t.pre != "smollm") { err = "unsupported tokenizer.ggml.pre: " + t.pre; return false; }
    return true;
}

bool ouro_tokenize(const ouro_tokenizer & t, const std::string & text, bool add_special,
                   std::vector<int32_t> & ids, std::string & err) {
    (void) err;
    ids.clear();
    size_t pos = 0;
    while (pos < text.size()) {
        // longest special token at `pos` (special_ids is sorted by descending length)
        int32_t sid = -1;
        size_t slen = 0;
        for (int32_t s : t.special_ids) {
            const std::string & sp = t.id_to_piece[s];
            if (!sp.empty() && text.compare(pos, sp.size(), sp) == 0) { sid = s; slen = sp.size(); break; }
        }
        if (sid >= 0) { ids.push_back(sid); pos += slen; continue; }

        // BPE the raw text up to the next special token
        size_t next = text.size();
        for (size_t q = pos + 1; q < text.size() && next == text.size(); ++q) {
            for (int32_t s : t.special_ids) {
                const std::string & sp = t.id_to_piece[s];
                if (!sp.empty() && text.compare(q, sp.size(), sp) == 0) { next = q; break; }
            }
        }
        bpe_encode_raw(t, text.substr(pos, next - pos), ids);
        pos = next;
    }
    if (add_special) {
        if (t.add_bos && t.bos_id >= 0) ids.insert(ids.begin(), t.bos_id);
        if (t.add_eos && t.eos_id >= 0) ids.push_back(t.eos_id);
    }
    return true;
}

std::string ouro_detokenize(const ouro_tokenizer & t, const std::vector<int32_t> & ids, bool skip_special) {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= (int32_t) t.id_to_piece.size()) continue;
        const std::string & piece = t.id_to_piece[id];
        if (t.is_special[id]) { if (!skip_special) out += piece; continue; }
        size_t off = 0;
        while (off < piece.size()) {
            const size_t len = std::min(piece.size() - off, (size_t) unicode_len_utf8(piece[off]));
            out.push_back((char) unicode_utf8_to_byte(piece.substr(off, len)));
            off += len;
        }
    }
    return out;
}
