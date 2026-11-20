# Cycle 3 — Iteration 2 pre-registration (written before running)

**Question**: does verified surprisal-watershed folding produce a trustworthy fold policy — high fold fraction on redundant text, zero fidelity loss, zero gated-atom violations?

Trimmed per directive v2: two text types (corpus boilerplate from `data/corpus_seed0.jsonl`, ~1500 tokens; one prose sample from `data/e0_fixed_paragraph.txt`), one fold level (no recursion), no four-type comparison.

Amendments applied:
- **Computation-shape equality at encode and decode**: both the rank-stream encoding pass and the reconstruction-verification pass use `score_tokens` (a single batched call), never a sequential single-token decode loop, per the E1 lesson that batched and sequential eval give measurably different logits on this CPU/quantized stack. Reconstruction here validates the storage format (zstd+varint round-trip, rank→token lookup correctness under determinism) — it is not a test of independent sequential-generation-based unfolding, which is a different question deferred to Iteration 3.
- **Gated-atom containment as a hard assertion**: any gated atom (matches the gate word list, is a numeral, or is a named party/entity) found inside a committed fold raises an error immediately, rather than being logged as a soft warning.

## P-S (does not gate)

Boilerplate fold fraction lands within ±15 points of E1's 67% rank-0 fraction (i.e., roughly 52–82%). Prose fold fraction is lower than boilerplate's.

## P-T (gates)

10/10 randomly sampled committed folds reconstruct byte-exactly from their stored residuals. Zero gated atoms found inside any committed fold (hard assertion — a single violation fails this).

## Gate

- P-T holds → fold policy and cold tier (residual archive) validated; the live skeleton + per-section fold fractions + residual archive become Iteration 3's input.
- P-T fails on reconstruction specifically → treat as an E1-style computation-shape bug until proven otherwise; fix and rerun once. Fails again → surprisal-based folding is unreliable at this scale; Iteration 3 falls back to fixed-size chunking as the fold policy, reported as a negative result on the original thesis.
- P-S failing alone does not block — it recalibrates expectations and is logged, not treated as a failure.
