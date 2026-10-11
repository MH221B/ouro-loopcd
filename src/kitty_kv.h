// src/kitty_kv.h
#pragma once
#include "kv_cache.h"
#include <string>

// Fake-quantize the completed pages of every slot in `kv` (f16) so that attention over
// future steps sees Kitty-quantized K/V. Idempotent across calls: only pages/positions
// not yet quantized are touched (tracked by kv.kitty_k_done / kitty_v_done).
// `n_kv` is the number of valid positions. Requires kv.kitty.
bool ouro_kv_kitty_quantize(ouro_kv_cache & kv, int n_kv, std::string & err);

// M7b packed path. Call `evict` BEFORE the forward that writes the next chunk
// (it packs the V positions that will leave the f16 window), and `pages` AFTER it
// (it packs any K page the forward completed). No-ops unless kv.kitty && kitty_cfg.packed.
// `n_kv_new` is the position count after the chunk (or after the decode step).
bool ouro_kv_kitty_pack_evict(ouro_kv_cache & kv, int n_kv_new, std::string & err);
bool ouro_kv_kitty_pack_pages(ouro_kv_cache & kv, int n_kv_new, std::string & err);
