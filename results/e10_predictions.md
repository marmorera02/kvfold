# E10 — pre-registration (written before scoring)

**Question**: does surprisal-based direction detection work once the contrast is corrected — S(u|A) vs. S(u|¬A), a minimal pair with priming held constant — rather than Phase A's S(u|context) vs. S(u|∅), where topical priming lived only in the conditioned branch and could swamp direction?

## Pair set (built from existing artifacts only, no new corpus/scoring of old results)

`src/e10_contrast/build_pairs.py` extracted every guard-passed (hypothesis unit, premise assumption) pair from `e2_nli_v2_seed{0..5,wikiA,wikiB}.json`, joined against `e2_assumptions_seed*.json` (premise text) and `levels_seed*.json` (hypothesis text). **3,261 pairs total**: 16 TRUE_CONTRADICTION, 595 KNOWN_FP, 2,650 NEUTRAL. Full labeling rule and the seed-5 special case (clause 78 × clause 12, absent from the legacy guard-passed set because `assumptions.py`'s classifier never extracted clause 12 — confirmed `classification: ROUTINE, assumption: None` in `e2_assumptions_seed5.json`; this pair uses clause 12's own L1 summary as A, logged via `assumption_source`) are documented in that module's docstring.

**Scope decision, made before any scoring** (for runtime tractability — 3,261 pairs × 2 branches × 2 scorers would be ~13,000 calls, well beyond the directive's own "hundreds of pairs" estimate): all 16 TRUE_CONTRADICTION and all 595 KNOWN_FP pairs are scored in full (100% coverage — these are what P-W and P-X directly need). NEUTRAL pairs are downsampled to **300**, drawn via `random.Random(0).sample()` from the pooled 2,650 (seeded, reproducible, exact sampled ids logged in the results file) — used only as the "all other pairs" background population for P-Y's AUC. This decision is made now, before seeing any Δ± value, specifically to keep runtime tractable; it is a scope-management call, not a result-contingent one.

Total scored: 911 pairs × 2 branches (A, ¬A) × 2 scorers (Qwen, Llama) = 3,644 `score_tokens` calls.

## Constructing ¬A

`src/e10_contrast/negate.py`, three tiers (first applicable wins): (1) an ordered polarity-swap table (checked and manually spot-verified against all 16 TRUE_CONTRADICTION + 595 KNOWN_FP unique assumption texts before running — one bug found and fixed: "Neither party shall..." negated via bare "shall"→"shall not" produced a double negative; fixed by swapping "Neither"→"Either" as a prior, higher-priority rule); (2) spaCy dependency-parse negation (root-verb detection: negate an existing copula/modal, or do-support otherwise) — this is the general mechanism that also covers the directive's "numeric assumptions" case; (3) LLM-generated fallback (pinned, greedy), logged as `llm_generated`, only if neither rule applies. In the spot-check of all 611 unique TRUE/KNOWN_FP assumption texts, **0 required the LLM fallback tier** (all resolved via tiers 1-2).

**Hard length-match gate**: pairs where `token_count(¬A)` falls outside ±20% of `token_count(A)` are excluded from scoring and counted/logged, per the directive.

## Scoring protocol

- S(u | X) = mean bits/token of u, context = X + "\n", computed as a single batched `score_tokens` call (never sequential decode) — same code path for both branches of every pair, so the E1 batched-vs-sequential shape mismatch cannot enter this comparison. Both branches' context token counts are asserted within the ±20% band already gated above.
- Primary scorer: Qwen2.5-3B-Instruct-Q5_K_M (matches every other experiment this cycle). Cross-model control: Llama-3.2-3B-Instruct-Q5_K_M, identical protocol.
- **Sanity check, run first, before the full pass**: 5 hand-picked TRUE_CONTRADICTION pairs (the 3 liability-cap pairs from seed 0, wikiA's clause-85×clause-75 pair, and the seed-5 n2b×clause-12 special pair). Swapping which branch is called "A" vs "¬A" must flip the sign of Δ±. **If it doesn't, this run stops and the bug is fixed before any further scoring** — per the directive, this is a hard gate, not advisory.

## Noise floor for P-Y — a verification note

The directive cites "the 0.3 bits/token computation-noise floor from E1." I checked `results/e1_seed0.json` directly and **could not verify this exact figure** — the closest stored analogs are `mean_abs_logit_diff: 0.109` (nats, at the batched-vs-sequential shape boundary, not the same comparison as this experiment's same-shape A/¬A contrast) and `rank_mismatch_fraction: 0.205`. Rather than silently substitute an unverified number into a pass/fail bar, P-Y is reported against **two** reference points: (a) the directive's stated 0.3 bits/token figure, explicitly labeled as unverified against the stored artifact; (b) an empirically-measured within-experiment noise estimate — the spread of |Δ±| among the 300 sampled NEUTRAL pairs, which is the more directly relevant baseline for this specific same-shape measurement (no genuine signal is expected there, so its spread characterizes this experiment's own noise, not a different experiment's).

## Pre-registered criteria

- **P-W (recall)**: every TRUE_CONTRADICTION pair has Δ± > 0, under both scorers. Weak form (reported regardless): all TRUE pairs rank in the top decile of the full scored-pair distribution by Δ±.
- **P-X (precision)**: ≥ 90% of KNOWN_FP pairs (Task-2's documented false positives + all wikiB guard-passed pairs) have Δ± ≤ 0.
- **P-Y (separation)**: AUC(Δ±, TRUE_CONTRADICTION vs. all other scored pairs) ≥ 0.9, per scorer. Each TRUE pair's |Δ±| margin is additionally reported against both noise-floor reference points above; margins below either floor are reported as "inside noise" regardless of sign.
- **P-Z (mechanism check)**: for TRUE pairs, report whether S(u|¬A) < S(u|A) holds (¬A actively predicts u better than A does) — the direct demonstration that direction, not just magnitude, is being measured.

## Gate

- P-W + P-X + P-Y hold (both scorers) → surprisal-based direction detection is rehabilitated as a flag-generation mechanism; Phase A/C's inversion is attributed to the wrong contrast, not the signal. Next directive would integrate Δ± flags at ingest and design an open-world rerun.
- P-W fails with margins inside the noise floor → the negation-margin-collapse hypothesis at 3B scale is confirmed with the strongest design tried; surprisal-direction detection closes definitively.
- Mixed (legal needles pass, wikiA fails, or vice versa) → reported per-register. wikiA's contradiction pair carries the already-documented update-vs-contradiction framing confound (`phaseE_results.md`); its failure alone does not fail the gate, pre-registered as such here, before any score is computed.

No threshold retuning after seeing results. Excluded (length-gate-failed) pairs are counted and listed. Results: `results/e10_contrast.json` (+ per-scorer breakdown), env fingerprints as always.
