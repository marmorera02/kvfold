# Cycle 3 — Iteration 1 results: KV state save/restore

## E6.a — P-M: HOLDS

100-token greedy continuation is byte-identical between an unbroken session and a state-reloaded session run in a **fresh process** (`results/e6a_state_roundtrip.json`): `continuation_tokens_match: true`, no divergence at any index. This is the same fresh-process rigor as E0/E1, and it passes cleanly — llama-cpp-python's `save_state()`/`load_state()` round-trips the KV cache exactly on this build.

## E6.b — P-N: HOLDS, by a wide margin

| Chunk (tokens) | Reprefill median | Reload median | Speedup |
|---|---|---|---|
| 1,000 | 24.90s | 0.338s | 73.6x |
| 2,000 | 61.11s | 0.334s | 182.9x |
| 4,000 | 125.76s | 0.330s | **380.7x** |
| 8,000 | 273.15s | 0.389s | 702.5x |
| 16,000 | 651.33s | 0.544s | 1198.1x |

Predicted ≥20x at 4k; actual is 380.7x — nearly 20x better than the pre-registered floor. Speedup grows monotonically with chunk size, as predicted, because reload time is almost flat (0.33–0.54s across a 16x range of token counts — it's dominated by fixed deserialization/memcpy cost, not token count) while reprefill time scales linearly with tokens (as expected: prefill is O(n) transformer forward-pass work). This is the mechanism, not just the measurement: restoring from disk sidesteps the actual compute, so the gap necessarily widens as the avoided compute grows.

## Storage cost model (artifact for Iteration 2)

Naive bytes-saved/tokens-saved is *not* constant across chunk sizes (348,656 → 56,366 bytes/token as chunk grows) because each saved state carries a large fixed overhead. Fitting the actual data gives a clean linear model, confirmed exactly (not approximately) across all four consecutive chunk-size pairs:

**state_size_bytes = 311,776,005 + 36,880 × n_tokens**

- Fixed overhead: ~311.8 MB per saved state (independent of token count — likely fixed-size internal buffers in the serialization format).
- Marginal cost: **36,880 bytes/token** — this is the number Iteration 2 should use for any fold-vs-keep-live storage tradeoff math, not a naive total-size/token-count ratio.

## Gate verdict

**P-M holds → Iteration 2 is unlocked**, carrying forward: the state save/load wrapper (`src/e6_kvcache/state_roundtrip.py`, `speed_bench.py`), the speed table above, and the storage cost model (36,880 bytes/token marginal + 311.8MB fixed overhead per saved state).
