"""Emit tests/unit/fixtures/kitty_pack.bin for tests/unit/test_kitty_pack.cpp.

Format (little-endian):
  int32 H, int32 D, int32 P, int32 kbits, int32 boost_bits, int32 d_boosted
  f16  input[H*D*P]      laid out buf[t*H*D + h*D + d]  (token-major, as the cache)
  uint8 packed[page_bytes]  (see src/cache_layout.h byte layout)
  f16  dequant[H*D*P]    same token-major layout, recomputed from packed bytes+meta
"""
import struct, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitty_quant import fake_quant_page

H, D, P = 16, 128, 128
KBITS, BOOST_BITS, D_BOOSTED = 2, 4, 32

torch.manual_seed(1234)
x = torch.randn(H, D, P, dtype=torch.float32)                 # [H, D, P]
x = x.half().float()                                          # the cache holds f16: quantize the f16-rounded values

# channel scores = mean|k| over P, deterministic top-k (score desc, channel asc).
# Mirror select_boost() in src/kitty_kv.cpp EXACTLY: a SEQUENTIAL double accumulation
# over the f32 values, divided by P. Do NOT use torch/numpy mean() — its pairwise
# reduction differs from the C++ sequential sum and can flip promotion ties.
score = np.zeros((H, D), dtype=np.float64)
for h in range(H):
    for d in range(D):
        acc = 0.0
        for t in range(P):
            acc += abs(float(x[h, d, t]))
        score[h, d] = acc / P
boost_idx = np.full((H, D), D_BOOSTED, dtype=np.uint8)
rank = np.zeros((H, D_BOOSTED), dtype=np.int64)
for h in range(H):
    order = sorted(range(D), key=lambda d: (-float(score[h, d]), d))
    for i, d in enumerate(order[:D_BOOSTED]):
        boost_idx[h, d] = i
        rank[h, i] = d

# quantize each channel, build packed bytes + dequant
low = np.zeros((H, D, P), dtype=np.uint8)
high = np.zeros((H, D, P), dtype=np.uint8)
meta = np.zeros((H, D, 2), dtype=np.float16)
for h in range(H):
    for d in range(D):
        bits = BOOST_BITS if boost_idx[h, d] < D_BOOSTED else KBITS
        v = x[h, d, :]
        mn, mx = float(v.min()), float(v.max())
        scale = max(mx - mn, 1e-4) / (2 ** bits - 1)
        # std::round is half-AWAY-from-zero; torch.round is half-to-even. Mirror the C++.
        vn = v.numpy()
        z = (vn - mn) / scale
        q = np.clip(np.trunc(z + np.copysign(0.5, z)), 0, 2 ** bits - 1).astype(np.int64)
        meta[h, d] = np.array([scale, mn], dtype=np.float16)
        for t in range(P):
            low[h, d, t] = int(q[t]) & 3
            high[h, d, t] = (int(q[t]) >> 2) & 3

K_LOW, K_HIGH, K_IDX, K_META = D * (P // 4), D_BOOSTED * (P // 4), D, D * 2 * 2   # plane SIZES
K_OFF_IDX, K_OFF_META = K_LOW + K_HIGH, K_LOW + K_HIGH + K_IDX                    # plane OFFSETS (must match K_OFF_IDX / K_OFF_META in src/cache_layout.h)
K_OFF_HIGH = K_LOW                                                                # high plane sits right after the low plane (K_OFF_HIGH)
K_HEAD = K_LOW + K_HIGH + K_IDX + K_META
page = bytearray(H * K_HEAD)
for h in range(H):
    hb = h * K_HEAD
    for d in range(D):
        page[hb + K_OFF_IDX + d] = int(boost_idx[h, d])
        struct.pack_into("<e", page, hb + K_OFF_META + d * 4, float(meta[h, d, 0]))
        struct.pack_into("<e", page, hb + K_OFF_META + d * 4 + 2, float(meta[h, d, 1]))
        for t in range(P):
            page[hb + d * (P // 4) + t // 4] |= (int(low[h, d, t]) & 3) << ((t % 4) * 2)
            if boost_idx[h, d] < D_BOOSTED:
                s = int(boost_idx[h, d])
                page[hb + K_OFF_HIGH + s * (P // 4) + t // 4] |= (int(high[h, d, t]) & 3) << ((t % 4) * 2)

# dequantize from the packed representation (fp16 meta, f32 arithmetic)
deq = np.zeros((H, D, P), dtype=np.float32)
for h in range(H):
    hb = h * K_HEAD
    for d in range(D):
        scale = np.frombuffer(bytes(page[hb + K_OFF_META + d * 4     : hb + K_OFF_META + d * 4 + 2]), dtype=np.float16)[0]
        mn    = np.frombuffer(bytes(page[hb + K_OFF_META + d * 4 + 2 : hb + K_OFF_META + d * 4 + 4]), dtype=np.float16)[0]
        s = int(boost_idx[h, d])
        for t in range(P):
            lo = (page[hb + d * (P // 4) + t // 4] >> ((t % 4) * 2)) & 3
            hi = 0
            if s < D_BOOSTED:
                hi = (page[hb + K_OFF_HIGH + s * (P // 4) + t // 4] >> ((t % 4) * 2)) & 3
            deq[h, d, t] = float((lo | (hi << 2)) * float(scale) + float(mn))

def to_tokmajor(a):   # [H,D,P] -> [P,H,D] flat
    return a.transpose(2, 0, 1).reshape(-1)

out = Path(__file__).resolve().parents[1] / "unit" / "fixtures" / "kitty_pack.bin"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    f.write(struct.pack("<6i", H, D, P, KBITS, BOOST_BITS, D_BOOSTED))
    f.write(to_tokmajor(x.numpy()).astype(np.float16).tobytes())
    f.write(bytes(page))
    f.write(to_tokmajor(deq).astype(np.float16).tobytes())
print("wrote", out)
