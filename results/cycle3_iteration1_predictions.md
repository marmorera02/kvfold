# Cycle 3 — Iteration 1 pre-registration (written before running)

**Question**: does KV save-before-evict work bit-exactly on this build, and how large is the reload-vs-reprefill advantage?

Note: no "v1" CYCLE3_DIRECTIVE.md or prior E6 work exists in this repo (checked before starting) — Iteration 1 is implemented fresh from v2's self-contained description, using llama-cpp-python's `Llama.save_state()`/`load_state()` (in-memory `LlamaState`, pickled to disk for genuine cross-process round trips), consistent with the fresh-process discipline established in E0/E1.

## E6.a — state round-trip (P-M)

Method: Process A loads the model, prefills a fixed 2000-token context, calls `save_state()` immediately (before generating anything), pickles the state to disk, then continues to greedy-generate 100 tokens ("unbroken continuation") in that same unbroken session. Process A exits. Process B (fresh process) loads the model, unpickles and `load_state()`s the saved state, then greedy-generates the same 100 tokens from that restored point ("restored continuation"). Both use the pinned determinism settings (seed=0, temperature=0, top_k=1, fixed n_threads).

**P-M**: the unbroken continuation and the restored continuation are byte-identical, token for token.

## E6.b — reload vs re-prefill speed (P-N)

Method: for chunk sizes {1000, 2000, 4000, 8000, 16000} tokens, 3 repetitions each, CPU backend (mandatory per COMPUTE_MIGRATION_DIRECTIVE — GPU generation failed its determinism gate in G2, and GPU scoring was never attempted; this is a scoring/state-restore workload so it stays CPU regardless): (a) re-prefill: fresh process, load model, prefill that many tokens from scratch, measure wall-clock; (b) reload: fresh process, load model, `load_state()` from a pre-saved state file for that chunk size, measure wall-clock for the restore call. Report medians.

**P-N**: reload is ≥20x faster than re-prefill at the 4000-token chunk size, with the advantage growing as chunk size increases.

## Gate

- P-M holds → hot-tier-on-disk is real; proceed to Iteration 2 with the measured kv_bytes_per_token constant and the E6.b speed table as artifacts.
- P-M fails → E6 line closes on this build; log the failure mode (API level / file level / shape sensitivity); Iterations 2 and 4 still proceed (no dependency on E6); Iteration 3 falls back to text-only unfolding regimes.

P-N does not gate — it's reported regardless as part of the forward artifact (speed table), per the directive's "Method: exactly E6.a and E6.b" framing (P-M is the only branch condition named in the Gate section).
