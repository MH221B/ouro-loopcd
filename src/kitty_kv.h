// src/kitty_kv.h
#pragma once
#include "kv_cache.h"
#include <string>

// Fake-quantize the completed pages of every slot in `kv` (f16) so that attention over
// future steps sees Kitty-quantized K/V. Idempotent across calls: only pages/positions
// not yet quantized are touched (tracked by kv.kitty_k_done / kitty_v_done).
// `n_kv` is the number of valid positions. Requires kv.kitty.
bool ouro_kv_kitty_quantize(ouro_kv_cache & kv, int n_kv, std::string & err);
