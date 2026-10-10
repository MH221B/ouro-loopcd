// Offline tokenizer checks against the real GGUF (metadata only; no weights are loaded).
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(const char * name, bool ok) {
    if (!ok) { printf("FAIL %s\n", name); g_fail = 1; } else { printf("ok   %s\n", name); }
}

static std::vector<int32_t> tok(const ouro_tokenizer & t, const std::string & s) {
    std::vector<int32_t> ids; std::string err;
    if (!ouro_tokenize(t, s, false, ids, err)) { printf("tokenize error: %s\n", err.c_str()); g_fail = 1; }
    return ids;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: test_tokenizer <model.gguf>\n"); return 1; }
    ouro_tokenizer t;
    std::string err;
    if (!ouro_tokenizer_load(t, argv[1], err)) { printf("load error: %s\n", err.c_str()); return 1; }

    check("vocab size 49152", t.id_to_piece.size() == 49152);
    check("bos=1 eos=2 unk=0", t.bos_id == 1 && t.eos_id == 2 && t.unk_id == 0);
    check("add_bos false", t.add_bos == false);
    check("pre == smollm", t.pre == "smollm");

    // ASCII round-trip through the byte-level BPE
    for (const char * s : { "Hello, world!", "def fib(n):\n    return n", "2 + 2 = 4" }) {
        const std::string text = s;
        const std::vector<int32_t> ids = tok(t, text);
        const std::string back = ouro_detokenize(t, ids);
        char nm[128]; snprintf(nm, sizeof(nm), "round-trip '%s'", s);
        check(nm, !ids.empty() && back == text);
    }

    // a special token in the text is emitted as a single id (not BPE'd)
    const std::vector<int32_t> sp = tok(t, "a<|im_end|>b");
    check("special <|im_end|> present",
          std::find(sp.begin(), sp.end(), 2) != sp.end());

    printf(g_fail ? "TOKENIZER: FAIL\n" : "TOKENIZER: PASS\n");
    return g_fail;
}
