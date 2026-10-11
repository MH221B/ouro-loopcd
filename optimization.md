# Packed KV throughput at R=2

Status: designed 2026-10-11. Nothing built.

This is the design for an optimization I never got to. It is written so that someone else, or future me, can pick it up and finish it. Everything needed to understand it is either in this file or in the repo's committed code and results.

## Background

Ouro is a looped transformer. It has 48 layers and applies them R times, reusing the same weights on each pass, so the reasoning happens inside the loop rather than in emitted chain-of-thought tokens. The model was trained at R=4.

LoopCD is the contrastive decoding this project exists to test. It contrasts the intermediate answer after loop 1 with the final answer after loop R, so it needs at least two loops to have anything to contrast. R=1 is not on the table.

The KV cache holds attention keys and values, one tensor per (loop, layer) slot, so 48 * R slots in all. It can be f16, q8_0, or the packed 2-bit format that M7 shipped: every channel at 2 bits, the most important key channels per page promoted to 4 bits, and a sink plus the pending page kept in f16. Attention over the packed format runs through two custom ops, `KITTY_QK` and `KITTY_SV`, with a plain softmax between them. Upstream ggml does not have those ops, so they travel as `tools/ggml-kitty-ops.patch` and are applied at build time.

Three test names come up in the gates below. `kitty_ops` checks the two ops against an independent reference. `kitty_pack` checks the packed bytes against a fixture. `kitty_equiv` checks that a packed page's dequantized values match what f16 mode stores, within fp16 rounding. All three live in `tests/unit` and run under ctest.

The measured numbers live in `tests/eval/RESULTS.md`. Line numbers quoted for ggml sources refer to the pinned commit in `CMakeLists.txt` (`ffa4e8b8`), plus the patch. The M7 design spec is not committed, so anything from it that matters here is repeated inline.

## 1. Goal

Make the packed 2-bit KV cache pay off in speed, not only in memory. M7 got a 39% memory win and a decode regression of about 2x. This closes that regression and turns the cache's byte advantage into throughput, at R=2.

This is not "write split-K". It is: measure where decode time actually goes, fix the terms that dominate, and say so if the recorded explanation (no split-K) turns out to be wrong.

## 2. What is measured today

| config | ctx | tok/s | source |
|---|---|---|---|
| q8_0, R=4 | 4096 | 11.5 | M5, `RESULTS.md` |
| q8_0, R=2 | 4096 | 20.8 | M5, `RESULTS.md` |
| packed, R=4 | 4096 | 5.6 | M7b/c, `RESULTS.md` |
| packed, R=2 | 4096 | not measured | |

Memory at ctx 4096: packed 3.16 GiB against q8_0's 5.19 GiB. That is 2.03 GiB less, 39%.

The M7 write-up and `RESULTS.md:267-270` blame the regression on the first-cut kernel having no split-K. Two things in the code say that is not the whole story.

1. `KITTY_SV` has almost no parallelism. `kitty_sv.comp:18` sets `local_size_x = 8`, and the dispatch is `{n_q, n_head, 1}` (`ggml-vulkan.cpp:9440-9441`). At decode `n_q = 1`, so that is `8 * n_head` = 128 threads in total, each walking all `n_kv` keys in a serial loop with no reduction and no scratch. `KITTY_QK` launches `n_head * ceil(n_kv/16)` workgroups (`:9437-9438`), which is 4096 at ctx 4096.
2. The host packer does a blocking device round-trip per slot, per step. `pack_v_token` (`src/kitty_kv.cpp:245-286`) loops every slot doing one `ggml_backend_tensor_get` (a fence wait) and one `ggml_backend_tensor_set`, every step, because V stages a 1-page ring that slides one token at a time. At R=4 that is 384 blocking round-trips per decode step across 192 slots, and 192 at R=2 across 96. Each get and set blocks on a fence (`ggml-vulkan-buffers.cpp:530,641`).

Either could dominate. S1 decides, not this document.

## 3. Why the bar is 27 and not 60

The unique weights are 1.54 GB at Q4_K_M. The f16 file is 4.97 GB, so the model is about 2.5B parameters. Ouro applies its 48 layers R times with the same weights, so every loop re-reads them from DRAM. Per token at R=2, with 96 slots:

```
weights   2 * 1.54 GB                       = 3.08 GB   shared by packed and q8_0
packed KV 96 * 2048 * 2048 * 2 * 0.316 B    = 0.25 GB
                                              --------
total                                         3.33 GB
```

The engine is bandwidth-bound, and its own numbers prove it. R=4 runs at 11.5 tok/s and R=2 at 20.8. Halving the loop count scales the time by about 1.8x, which is what weight traffic looks like. At 11.5 tok/s the engine moves about 7.9 GB per token in 87 ms, so roughly 90 GB/s, on an LPDDR5X part of about 120 to 136 GB/s shared with the CPU.

Some consequences, recorded so nobody re-proposes them.

- 60 tok/s is out of reach at R>=2. It is 16.7 ms per token, and 3.33 GB / 0.0167 s is 200 GB/s, about 1.5 to 1.7x the platform peak, with perfect kernels. 60 tok/s is the R=1 number: at R=1 the cache is 48 slots, traffic is 1.54 + 0.13 = 1.67 GB per token, and 16.7 ms implies about 100 GB/s, which is the R=1 roofline at realistic efficiency. R=1 makes LoopCD meaningless, since the contrast needs at least two loops. So R=2 is the floor and 60 is not reachable.
- A 1.5x ratio is out of reach too. The weights are 78% of q8_0's traffic and identical for both paths, so the packed cache can only shrink the remaining 22%. That caps packed/q8_0 at 1.18 with an average ctx of 2048, or 1.33 at a full ctx of 4096. The bar is 1.15, not 1.5.
- The R=2 ceiling is 27 tok/s at today's ~90 GB/s, 30 at 100, 36 at 120, and 41 at 136. G1 is 27, which sits at the realistic roofline. Reachable, but only by doing the work.
- The only lever that raises the ceiling is bytes per weight, by quantizing below Q4_K_M. Rejected. That is a quality trade, not a speed trick, and M5 already showed the quantized stack is fragile.

## 4. Scope, five stages

```
S1  profile    per-stage time budget, decode and prefill, packed vs q8_0, ctx 4096
S2  packer     kill the per-step host syncs
S3  kernel     SV split-K, and QK only if S1 says so
S4  prefill    its own stage, with its own gate
S5  loop       graph reuse and dispatch, if S1 shows they matter
```

S2 through S5 are conditional. A stage is entered only if S1 shows its term exceeds a pre-registered share of the step (§6.3). Each is independently revertable and carries its own before and after. If S1 says the packer is 8% and the kernel is 70%, we do the kernel and skip the packer.

Out of scope, do not build: a fused online-softmax shader (the two-op structure deliberately has none), cooperative-matrix paths, weight requantisation, batched serving, and two layout levers from M7 that were deliberately left alone, a fixed promoted-channel set and a dense high plane. Those last two change the algorithm, and then the gate can no longer tell a bug from a deliberate change.

## 5. The gate

Frozen before implementation starts.

| | criterion |
|---|---|
| G1, absolute | packed >= 27 tok/s at R=2, ctx 4096, long generation, about 5x today's 5.6 |
| G2, relative | packed >= 1.15x q8_0, same build, same ctx |
| G3, prefill | set after S1 by a pre-registered rule, once a baseline exists |
| G4, no regression | neither path slower than its frozen baseline, and `kitty_ops`, `kitty_pack` and `kitty_equiv` still pass |

G1 and G2 are deliberately redundant. G1 on its own can be satisfied by making q8_0 slower. G2 on its own does not assert that the byte advantage shows up as speed. Together they cannot be gamed.

The shipping default stays R=4. The gate is measured at R=2 and has to be labelled that way wherever it is reported. Whether R=2 becomes the default is deferred until the full accuracy picture at R=2 is in hand (§11.3). That is a product and accuracy decision, not a perf one.

No gate gets loosened to make a result pass. If the roofline holds and G1 is missed, the milestone reports a miss.

## 6. S1, the profiling harness

### 6.1 Instrumentation

Phase timers in `src/generate.cpp`, wrapping each phase: graph build, the forward over the whole 192-layer or 96-layer graph, `pack_evict`, `pack_pages`, guided logits, sample, and the host-to-device sync. Reported as a per-token breakdown over N steps, not a single shot.

### 6.2 Ablations

The deltas do the attributing, not the absolute numbers.

```
q8_0     no packer, stock flash attention     engine baseline
f16      no packer, no packed planes          adds KV quantisation, no kitty kernel
packed   everything                           full cost
```

There is also a packer microbenchmark: time `pack_evict` and `pack_pages` alone with the forward removed, over N iterations. That separates "the packer costs X ms" from "the packer costs X ms of sync", which is the number that matters.

### 6.3 Output and decision rules

```
stage                     packed   q8_0    attributable to
graph build                 ?       ?      engine (both)
forward, attention          ?       ?      kernel delta (SV vs stock FA)
forward, everything else    ?       ?      engine (both)
pack_evict / pack_pages     ?       -      packer (packed only)
guided logits + sample      ?       ?      engine (both)
total per token             ?       ?
```

The forward is timed whole and the packer is timed separately, so the kernel delta is `packed_forward` minus `q8_0_forward`, with the packer already excluded. Prefill phases are timed the same way, per chunk.

The entry rule, stated before any numbers are seen: a stage is entered if its attributable term is at least 15% of the packed per-token total, or if removing it is required to reach G1. Anything below that gets recorded and left alone. The rule is fixed here so it cannot be fitted to the result afterwards.

## 7. S2, the packer

`pack_v_token` walks every slot doing one blocking `tensor_get` and one `tensor_set`, every step, because V stages a 1-page f16 ring that slides one token at a time. At R=2 that is 96 slots times two, so about 192 blocking round-trips per token.

- P1, batch the transfers. Fewer syncs, but M7a made the cache one tensor per slot, so they are not contiguous. Bounded win.
- P2, widen the V ring to 2 pages. Let a full page accumulate, then pack 128 tokens in one event. That is about 128x fewer syncs for the same bytes. This is a swap that was flagged during M7 and never taken. The cost is doubling the f16 staging, from about 240 MiB to 480 MiB at R=4, and half that at R=2.
- P3, a GPU pack kernel. This is the "GPU pack op" M7 deferred. It removes the host round-trips entirely and is the largest change.

Order: P2 first, because it is targeted and cheap. Reach for P3 only if S1 says the packer still dominates after P2.

## 8. S3, the kernel

`KITTY_SV` is the binding constraint. 128 threads in total, serial over `n_kv`, no reduction.

- K1, SV split-K. Split the key axis across workgroups. Each computes a partial `acc[16]`, and the combine is a plain additive sum. Because `SOFT_MAX` is a separate ggml op, there is no `m`/`L` rescale and no online-softmax state machine, so the complexity M7 avoided stays avoided. This mirrors `ggml_vk_flash_attn`: the op currently goes through `ggml_vk_op_f32` (`ggml-vulkan.cpp:11120-11133`), which cannot bind scratch or dispatch twice, so the dispatch has to be hand-written. The scratch is backend-owned, the way `prealloc_split_k` is for flash attention (`ggml-vulkan.cpp:8310-8321`, `:8546-8556`). The combine reuses `mul_mat_split_k_reduce.comp`, which already does a plain sum (`:30-32`), or a small KITTY-specific reduce shader. The split count travels in push constants, as it does for flash attention (`flash_attn_base.glsl:72-75`, `:158-182`).
- K2, QK. Only if S1 says QK matters. It already launches 4096 workgroups. Its one wart is a final reduce that uses 16 of 128 threads (`kitty_qk.comp:88-95`).

## 9. S4, prefill

Prefill is a different shape, not a scaled decode. At prefill `n_q` is the chunk size, 128, so `KITTY_SV` dispatches `128 * n_head` workgroups instead of 16. The parallelism problem that dominates decode is mostly not there. What is there:

- The prompt runs in 128-token chunks, and each chunk runs the stack R times, so the cost is `chunks * R * 48` layer passes.
- `pack_k_page` moves about 118 MB per completed K page across all slots (`src/kitty_kv.cpp:194-242`). That is a burst, not a trickle.
- Chunk size is a lever. Bigger chunks mean fewer passes over the fixed per-layer overhead, at a staging-memory cost.

The candidates are chunk size, K-page packing batching (P2's idea applied to K), and S3's split-K if attention still binds. All gated on S1.

## 10. S5, the decode loop

Entered only if S1 shows per-step graph construction or dispatch overhead is significant. The candidate is reusing the graph across decode steps, if it is rebuilt per step, plus cutting the dispatch count. Nothing here is assumed. The profile decides.

## 11. Verification

### 11.1 Perf gates have to be falsifiable

Each of G1 and G2 is shown to fail when its optimisation is disabled. That is M7's rule: a test that cannot fail is not evidence. A negative control is required per gate, the way `kitty_equiv` was controlled, where disabling the high-bit merge drove the delta from 2.44e-3 to 4.0.

### 11.2 Numerics, the highest-risk item

Split-K changes the SV accumulation order, so the `kitty_ops` 1e-10 tolerance will probably not hold any more. The tolerance gets re-derived with a pre-registered rule: the expected f32 rounding bound is stated before the measurement, then measured against. It is not relaxed after seeing the number. `kitty_pack`, the byte fixture, is unaffected. `kitty_equiv`, the cache-level check, is unaffected too, because it tests storage rather than accumulation order.

### 11.3 Accuracy

If R=2 later becomes the default, the M5 accuracy table is re-stated at R=2 (it already has R=2 rows) and the README line about "default 4, what the model was trained at" changes. This milestone makes no new accuracy claim. It reports speed at R=2 and leaves the default at R=4.

### 11.4 Attribution

Every stage reports its own before and after, so the final number traces back to a specific change. A speedup that cannot be attributed to a stage does not count.

## 12. Risks

1. S1 may show the recorded attribution was wrong, and the packer rather than the kernel dominates. That is the point of the approach, not a failure.
2. G1 sits at the realistic roofline. 27 tok/s assumes ~90 GB/s can be pushed toward 100. If the packer is not a large term and split-K lands poorly, G1 can be missed. Reported, not fudged.
3. Split-K numerics, §11.2. The tolerance re-derivation is where the M7 discipline is most at risk.
4. The ggml patch grows. Split-K adds a combine shader and hand-written dispatch, which widens the fork surface (`tools/ggml-kitty-ops.patch`) and its rebase cost.
5. P2's staging cost. Doubling the V ring costs about 240 MiB at R=2, and it has to be shown to be worth the sync reduction.
6. Chunked prefill is fiddly. Per-chunk masks and packing boundaries are new code, and S4's chunk-size lever sits on top of it.

## 13. Acceptance criteria

Done when:

1. S1 produces a per-token and per-chunk time budget for decode and prefill, packed against q8_0, at ctx 4096, and the entry decisions for S2 through S5 are made against §6.3's pre-registered rule.
2. G1 passes, packed >= 27 tok/s at R=2, ctx 4096, long generation, or is reported as a miss with the measured ceiling.
3. G2 passes, packed >= 1.15x q8_0, same build.
4. G3 is defined from S1's baseline and passes.
5. G4 holds. No regression, and `kitty_ops`, `kitty_pack` and `kitty_equiv` still pass with the tolerance re-derived per §11.2.
6. Each of G1 and G2 has a negative control showing it fails when the optimisation is off.
7. Every stage that ran reports a before and after attributed to it, and every stage skipped by §6.3 is recorded as skipped with the number that justified skipping it.
8. No gate was loosened to make a result pass, and any miss is reported with the roofline that explains it.
