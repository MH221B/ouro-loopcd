#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Byte-level BPE tokenizer loaded from a GGUF's tokenizer metadata (spec 4.7).
struct ouro_tokenizer {
    std::vector<std::string> id_to_piece;                       // GGUF token strings (byte-encoded)
    std::unordered_map<std::string, int32_t> token_to_id;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::vector<int32_t> special_ids;                           // sorted by descending piece length
    std::vector<uint8_t> is_special;                            // size == id_to_piece.size()
    int32_t bos_id = -1, eos_id = -1, unk_id = -1;
    bool add_bos = false, add_eos = false, add_space_prefix = false;
    std::string pre;                                            // "smollm"
    std::string chat_template;                                  // raw GGUF string (informational)
};

bool ouro_tokenizer_load(ouro_tokenizer & t, const std::string & gguf_path, std::string & err);

// Byte-level BPE. `add_special` applies the GGUF BOS/EOS policy (this model ships add_bos=false).
bool ouro_tokenize(const ouro_tokenizer & t, const std::string & text, bool add_special,
                   std::vector<int32_t> & ids, std::string & err);

// Detokenize; control/unknown special tokens are skipped unless skip_special is false.
std::string ouro_detokenize(const ouro_tokenizer & t, const std::vector<int32_t> & ids,
                            bool skip_special = true);
