"""Kitty-KV fake-quantization math. Ported from Summer-Summer/Kitty
(kitty/src/kitty_sim/utils_quant.py) so the oracle, the fixture, and the engine share
one definition.

Conventions here: a KEY page is [H, D, P] = (heads, channels, page positions); group
statistics are taken over the P axis (a page of tokens). A VALUE slice is [H, D, T] and
is quantized per token (statistics over the D axis).
"""
import torch


def build_promote_mask(score, d_boosted):
    """score: [H, D] channel importance. Return a bool [H, D] mask of the top d_boosted
    channels per head (Kitty channel_selection=1, magnitude)."""
    H, D = score.shape
    k = max(0, min(int(d_boosted), D))
    if k == 0:
        return torch.zeros((H, D), dtype=torch.bool, device=score.device)
    if k >= D:
        return torch.ones((H, D), dtype=torch.bool, device=score.device)
    mask = torch.zeros((H, D), dtype=torch.bool, device=score.device)
    _, idx = score.topk(k, dim=-1)
    mask.scatter_(-1, idx, True)
    return mask


def _eps(x):
    return 1e-4 if x.dtype in (torch.float16, torch.bfloat16) else 1e-6


def fake_quant_page(x, kbits, boost_bits, d_boosted, mask=None):
    """x: [H, D, P] fp16 page. Asymmetric per-channel quant over P; the promoted channels
    use boost_bits, the rest kbits. Returns dequantized [H, D, P] (same dtype as x)."""
    H, D, P = x.shape
    if mask is None:
        mask = build_promote_mask(x.abs().mean(dim=-1), d_boosted)
    mn = x.amin(dim=-1, keepdim=True)                      # [H, D, 1]
    mx = x.amax(dim=-1, keepdim=True)
    scale_base = (mx - mn).clamp(min=_eps(x)) / (2 ** kbits - 1)
    scale_boost = (mx - mn).clamp(min=_eps(x)) / (2 ** boost_bits - 1)
    m = mask.view(H, D, 1)
    scale = torch.where(m, scale_boost, scale_base)
    maxv = torch.where(m, torch.full_like(scale, float(2 ** boost_bits - 1)),
                          torch.full_like(scale, float(2 ** kbits - 1)))
    q = ((x - mn) / scale).clamp(min=0.0)
    q = torch.minimum(q, maxv).round()
    return (q * scale + mn).to(x.dtype)


def fake_quant_v(x, vbits):
    """x: [H, D, T] fp16 values. Per-token (group over D) asymmetric quant, KIVI-style.
    Returns dequantized [H, D, T]."""
    H, D, T = x.shape
    mn = x.amin(dim=1, keepdim=True)                       # [H, 1, T]
    mx = x.amax(dim=1, keepdim=True)
    scale = (mx - mn).clamp(min=_eps(x)) / (2 ** vbits - 1)
    q = ((x - mn) / scale).clamp(min=0.0, max=float(2 ** vbits - 1)).round()
    return (q * scale + mn).to(x.dtype)
