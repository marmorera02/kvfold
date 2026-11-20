# E11: naked 7B vs. enhanced 3B/7B — ingest/decode crossover (owner request)

Per the owner's request: compare naked (as-shipped, full fresh prefill every query, no reuse) against enhanced (surprise-gated fold + KV-reload) at both 3B and 7B, measuring ingest and decode time separately, to find the break-even query count where the enhanced method's one-time setup cost pays for itself. Same boilerplate document (~1,513 tokens) and the same frozen QA set (`data/e6_qa_set.json`) throughout; scope deliberately kept to 3 of the 10 questions ("a very short sample test," the owner's words) — the one-time fold-detection cost itself is measured in full, not shortened.

**Models**: `Qwen2.5-3B-Instruct-Q5_K_M.gguf` (already present) and `Qwen2.5-7B-Instruct-Q5_K_M.gguf` (downloaded for this test — bartowski's single-file Q5_K_M quantization, 5,444,831,936 bytes, verified byte-for-byte against the source's Content-Length).

**Important scoping note**: this run uses the FAST fold-detection pipeline (`verify_2b_fast.py`), not Iterations 2b–4's original. The original took 60+ minutes on this document and never finished during initial testing for this experiment — a real, separate finding written up in `verify_2b_fast_results.md` (root cause: candidates verified in cost-ascending order, full context re-prefilled from scratch per candidate; fixed via a linear document-order pass with incremental KV extension, ~12x+ faster, fully validated: 10/10 reconstruction, 0 gated-atom violations). The fast pipeline uses document order rather than cost-ascending order, which changes the fold fraction (disclosed in that file) — so the numbers below are fresh measurements, not a reuse of Iterations 2b–4's own recorded fold fractions.

## Results

| | 3B | 7B |
|---|---|---|
| Fold fraction (fast pipeline) | 49.3% | 53.8% |
| One-time enhanced setup cost | **265.0s** (fold-detect 235.3s + skeleton prefill/save 29.7s) | **499.3s** (fold-detect 435.3s + skeleton prefill/save 64.0s) |
| Naked ingest (mean) | 52.20s | 159.50s |
| Naked decode (mean, 60 tokens) | 4.80s (12.5 tok/s) | 12.42s (4.9 tok/s) |
| Naked accuracy | 3/3 | 3/3 |
| Enhanced ingest (mean: reload + remainder prefill) | 25.37s | 97.37s |
| Enhanced decode (mean, 60 tokens) | 4.92s (12.2 tok/s) | 13.85s (4.4 tok/s) |
| Enhanced accuracy | **1/3** | **3/3** |
| Per-query savings (ingest+decode combined) | 26.71s | 60.70s |
| **Crossover (queries to break even)** | **~9.9** | **~8.2** |

## What this answers

**Ingest**: enhanced cuts ingest time roughly in half at both sizes (3B: 52.2s→25.4s; 7B: 159.5s→97.4s) — the KV-reload mechanism (validated since Iteration 1 at 380–1200x reload-vs-reprefill speedup) does the work; the remaining ingest cost is re-prefilling the reconstructed folded remainder, which the fold fraction (~50%) directly bounds.

**Decode**: essentially a wash at both sizes (3B: 12.5 vs 12.2 tok/s; 7B: 4.9 vs 4.4 tok/s) — consistent with everything else this cycle has found: the speed benefit of folding is concentrated in ingest (avoiding re-prefill), not in per-token generation, which depends on model size, not on how the preceding context was populated.

**Crossover**: both models break even within **8–10 consecutive prompts against the same document**. 7B's one-time setup cost is *larger* in absolute terms (499s vs 265s, since every fold-detection eval call costs more on a bigger model) but its per-query savings are also larger (60.7s vs 26.7s) — net effect, 7B actually breaks even **sooner** (~8.2 vs ~9.9 queries), not later.

**The accuracy result matters more than the timing result here**: at 3B, enhanced accuracy collapsed to 1/3 on this sample — worse than Iteration 3's original 90%/10-question result, because the fast pipeline's document-order fold fraction (49.3%) is meaningfully higher than Iteration 2b's cost-order fraction (38.3%), and this small model loses answer-relevant content at the higher fold level. At 7B, enhanced accuracy held at 3/3 despite an even *higher* fold fraction (53.8%) — the larger model's capacity appears to absorb the information loss from folding that the 3B model doesn't. Net: **7B enhanced is the only regime in this test that is both faster (past ~8 queries) and lossless on this sample; 3B enhanced is faster past ~10 queries but measurably lossy.** This is a small sample (3 questions) and shouldn't be over-read as a precise accuracy number, but the *direction* — larger model tolerates more aggressive folding — is a real, useful signal for anyone deciding which model size to pair with this method.

Full per-question data: `results/e11_7b_frontier_3B.json`, `results/e11_7b_frontier_7B.json`.
