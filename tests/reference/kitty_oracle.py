#!/usr/bin/env python3
"""M6 Task 0: apply Kitty-KV fake quantization to Ouro's HF forward and measure the
accuracy impact, at R=4 and R=2.

Ouro's HF model uses its own cache `UniversalTransformerCache`; `OuroAttention.forward`
calls `update(..., current_ut*num_hidden_layers + layer_idx)`, i.e. the same
`loop*48 + layer` slot index loopcd uses. `needs_universal_cache` accepts any
`UniversalTransformerCache` instance verbatim, so we subclass it and fake-quantize the
stored history in place after each `update`.

Two metrics, both against the `f16` baseline on the same token stream:

  * greedy token agreement  - each mode generates greedily with its own (quantized)
    cache; we compare the emitted token sequence to the f16 baseline's.
  * teacher-forced per-position agreement + logit deltas - the mode is fed the
    baseline's own tokens, so the context is identical except for the quantization of
    the cache. This isolates the perturbation from end-to-end compounding.

Page parameter P=128 means no key page is quantized until the cache holds more than
`sink + P` tokens, so the pinned `prompts.json` (a few tokens) cannot exercise the
algorithm. Use `tests/reference/kitty_prompts.json` (long natural prompts) for the gate;
`prompts.json` remains a fast self-consistency smoke pass.

    python tests/reference/kitty_oracle.py --prompts tests/reference/kitty_prompts.json \
        --loops 4 --max-new 48 --out tests/reference/out/kitty.r4.json
"""
import argparse, gc, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitty_quant import fake_quant_page, fake_quant_v

MODES = {
    "f16":        dict(sink=0,   boost=0.0,  kbits=16, vbits=16),
    "kivi2":      dict(sink=0,   boost=0.0,  kbits=2,  vbits=2),
    "kivi2_star": dict(sink=32,  boost=0.0,  kbits=2,  vbits=2),   # KIVI*-2
    "kitty":      dict(sink=32,  boost=0.25, kbits=2,  vbits=2),
    "kitty20":    dict(sink=32,  boost=0.20, kbits=2,  vbits=2),   # utils.sh ratio
}

PAGE = 128


def force_loops(model, n):
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        cfg = getattr(root, "config", None)
        if cfg is not None and hasattr(cfg, "total_ut_steps"):
            cfg.total_ut_steps = n
        if hasattr(root, "total_ut_steps"):
            root.total_ut_steps = n


def make_cache_class(model, head_dim):
    inner = getattr(model, "model", model)
    mod = sys.modules[inner.__class__.__module__]          # modeling_ouro
    Base = mod.UniversalTransformerCache
    BOOST_BITS = 4

    class KittyUniversalCache(Base):
        def __init__(self, mode, max_cache_size):
            super().__init__(max_cache_size)
            self.cfg = mode
            self.d_boosted = int(head_dim * mode["boost"] + 1e-6)
            self.k_pages = {}          # layer_idx -> completed pages already quantized
            self.v_done = {}           # layer_idx -> position V is quantized up to
            self.k_pages_quantized = 0
            self.v_tokens_quantized = 0

        def _quant_k(self, k, layer_idx, T):
            cfg = self.cfg
            if cfg["kbits"] >= 16:
                return k
            out = k.clone()
            done = self.k_pages.get(layer_idx, 0)
            a = cfg["sink"] + done * PAGE
            while a + PAGE <= T:
                x = out[:, :, a:a + PAGE, :].permute(0, 1, 3, 2)      # [1,H,D,P]
                dq = fake_quant_page(x[0], cfg["kbits"], BOOST_BITS, self.d_boosted)
                out[:, :, a:a + PAGE, :] = dq.permute(1, 2, 0).unsqueeze(0)
                a += PAGE
                done += 1
                self.k_pages_quantized += 1
            self.k_pages[layer_idx] = done
            return out

        def _quant_v(self, v, layer_idx, T):
            cfg = self.cfg
            if cfg["vbits"] >= 16:
                return v
            out = v.clone()
            start = max(cfg["sink"], self.v_done.get(layer_idx, cfg["sink"]))
            stop = T - PAGE
            if stop > start:
                xs = out[0, :, start:stop, :].permute(0, 2, 1)         # [H,D,n]
                dq = fake_quant_v(xs, cfg["vbits"])
                out[:, :, start:stop, :] = dq.permute(0, 2, 1).unsqueeze(0)
                self.v_done[layer_idx] = stop
                self.v_tokens_quantized += (stop - start)
            return out

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            rk, rv = super().update(key_states, value_states, layer_idx, cache_kwargs)
            T = rk.shape[2]
            kq = self._quant_k(rk, layer_idx, T)
            vq = self._quant_v(rv, layer_idx, T)
            self.key_cache[layer_idx] = kq
            self.value_cache[layer_idx] = vq
            return kq, vq

    return KittyUniversalCache


@torch.no_grad()
def greedy(model, ids, cache, max_new):
    """Greedy continuation; returns the emitted tokens and the logits that produced each."""
    logits = model(torch.tensor([ids], dtype=torch.long),
                   past_key_values=cache, use_cache=True).logits[0, -1]
    toks, vecs = [], []
    for _ in range(max_new):
        nxt = int(logits.argmax())
        toks.append(nxt)
        vecs.append(logits.float().cpu())
        logits = model(torch.tensor([[nxt]], dtype=torch.long),
                       past_key_values=cache, use_cache=True).logits[0, -1]
    return toks, vecs


@torch.no_grad()
def teacher(model, ids, cache, forced, base_vecs):
    """Feed the baseline's tokens; return per-step argmax and logit deltas vs the baseline."""
    logits = model(torch.tensor([ids], dtype=torch.long),
                   past_key_values=cache, use_cache=True).logits[0, -1]
    am, dmax, dmean = [], [], []
    for i, t in enumerate(forced):
        am.append(int(logits.argmax()))
        d = (logits.float().cpu() - base_vecs[i]).abs()
        dmax.append(float(d.max()))
        dmean.append(float(d.mean()))
        logits = model(torch.tensor([[t]], dtype=torch.long),
                       past_key_values=cache, use_cache=True).logits[0, -1]
    return am, dmax, dmean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ouro")
    ap.add_argument("--prompts", default="tests/reference/kitty_prompts.json")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--modes", default=",".join(MODES.keys()))
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    modes = [m for m in args.modes.split(",") if m]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    force_loops(model, args.loops)
    head_dim = model.config.head_dim
    KittyUniversalCache = make_cache_class(model, head_dim)

    prompts = json.loads(Path(args.prompts).read_text())
    n_slots = 48 * args.loops

    # Baseline f16: greedy tokens + the logits that produced them.
    baseline = {}
    for pr in prompts:
        ids = tok(pr["text"], return_tensors="pt")["input_ids"][0].tolist()
        c = KittyUniversalCache(MODES["f16"], n_slots)
        toks, vecs = greedy(model, ids, c, args.max_new)
        baseline[pr["id"]] = {"ids": ids, "toks": toks, "vecs": vecs}
        del c; gc.collect()
    print(f"[baseline] f16 done ({len(prompts)} prompts, R={args.loops}, max_new={args.max_new})", flush=True)

    report = {"loops": args.loops, "max_new": args.max_new, "prompts": args.prompts,
              "head_dim": head_dim, "page": PAGE, "modes": {}}

    for name in modes:
        mode = MODES[name]
        agg = {"greedy_agree": 0, "greedy_total": 0, "first_flip": [],
               "teacher_agree": 0, "teacher_total": 0, "logit_max": 0.0, "logit_mean": 0.0,
               "k_pages": 0, "v_tokens": 0, "per_prompt": []}
        for pr in prompts:
            base = baseline[pr["id"]]
            ids = base["ids"]

            c = KittyUniversalCache(mode, n_slots)
            toks, _ = greedy(model, ids, c, args.max_new)
            kp, vt = c.k_pages_quantized, c.v_tokens_quantized
            del c; gc.collect()

            n = min(len(toks), len(base["toks"]))
            agree = sum(1 for i in range(n) if toks[i] == base["toks"][i])
            flips = [i for i in range(n) if toks[i] != base["toks"][i]]
            first_flip = flips[0] if flips else None

            rec = {"id": pr["id"], "greedy_agree": agree, "greedy_n": n,
                   "first_flip": first_flip, "k_pages": kp, "v_tokens": vt}

            if not args.no_teacher and name != "f16":
                c = KittyUniversalCache(mode, n_slots)
                am, dmax, dmean = teacher(model, ids, c, base["toks"], base["vecs"])
                kp2, vt2 = c.k_pages_quantized, c.v_tokens_quantized
                del c; gc.collect()
                t_agree = sum(1 for i in range(n) if am[i] == base["toks"][i])
                rec.update({"teacher_agree": t_agree, "teacher_n": n,
                            "logit_max": max(dmax) if dmax else 0.0,
                            "logit_mean": sum(dmean) / len(dmean) if dmean else 0.0,
                            "k_pages_t": kp2, "v_tokens_t": vt2})
                agg["teacher_agree"] += t_agree
                agg["teacher_total"] += n
                agg["logit_max"] = max(agg["logit_max"], rec["logit_max"])
                agg["logit_mean"] = max(agg["logit_mean"], rec["logit_mean"])
                # teacher pass re-quantizes the same content (idempotent); keep the larger count
                kp = max(kp, kp2); vt = max(vt, vt2)

            agg["greedy_agree"] += agree
            agg["greedy_total"] += n
            agg["first_flip"].append(first_flip)
            agg["k_pages"] += 0 if agg["k_pages"] else 0
            agg["per_prompt"].append(rec)
            del toks; gc.collect()

        agg["greedy_agreement"] = agg["greedy_agree"] / agg["greedy_total"] if agg["greedy_total"] else 0.0
        if agg["teacher_total"]:
            agg["teacher_argmax_agreement"] = agg["teacher_agree"] / agg["teacher_total"]
        # sum pages/tokens across prompts (per-prompt records already hold them)
        agg["k_pages"] = sum(r["k_pages"] for r in agg["per_prompt"])
        agg["v_tokens"] = sum(r["v_tokens"] for r in agg["per_prompt"])
        report["modes"][name] = agg

        msg = f"{name}: greedy agreement = {agg['greedy_agreement']:.3f}"
        if agg["teacher_total"]:
            msg += f"  teacher argmax = {agg['teacher_argmax_agreement']:.3f}" \
                   f"  logit max|d| = {agg['logit_max']:.3g}  mean|d| = {agg['logit_mean']:.3g}"
        msg += f"  k_pages = {agg['k_pages']}  v_tokens = {agg['v_tokens']}"
        print(msg, flush=True)

        # Honesty guard: a 2-bit mode that quantized nothing is a vacuous pass, not a result.
        if name != "f16" and mode["kbits"] < 16 and agg["k_pages"] == 0 and agg["v_tokens"] == 0:
            print(f"WARNING: mode {name} quantized nothing (prompt too short for P={PAGE}); "
                  f"agreement is vacuous", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
