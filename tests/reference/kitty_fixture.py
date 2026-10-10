"""Emit tests/unit/fixtures/kitty_page.bin for tests/unit/test_kitty.cpp.

Format (little-endian):
  int32 H, int32 D, int32 P, int32 kbits, int32 boost_bits, int32 d_boosted
  f16 input[H*D*P]   laid out as buf[p*H*D + h*D + d]
  f16 expect[H*D*P]  same layout, after fake_quant_page
"""
import struct, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitty_quant import fake_quant_page

H, D, P = 4, 8, 16          # small, head_dim/heads-shaped page
KBITS, BOOST_BITS, D_BOOSTED = 2, 4, 2   # 25% of D

torch.manual_seed(1234)
x = torch.randn(H, D, P, dtype=torch.float32)          # [H, D, P]
exp = fake_quant_page(x.float(), KBITS, BOOST_BITS, D_BOOSTED)

def to_layout(t):                                        # [H,D,P] -> [P,H,D]
    return t.permute(2, 0, 1).contiguous().to(torch.float16).numpy().reshape(-1)

out = Path(__file__).resolve().parents[1] / "unit" / "fixtures" / "kitty_page.bin"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    f.write(struct.pack("<6i", H, D, P, KBITS, BOOST_BITS, D_BOOSTED))
    f.write(to_layout(x).tobytes())
    f.write(to_layout(exp).tobytes())
print("wrote", out)
