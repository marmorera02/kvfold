# Cycle 3 — Iteration 3 results: does the assembled system preserve truth?

## Gate verdict: P-O HOLDS → Iteration 4 proceeds. P-P fails, but for an already-understood reason (not a new escalation trigger).

| | R1 (gold) | R2 (naive stitch) | R3 (stitch + 64-tok seam) |
|---|---|---|---|
| QA accuracy (10 questions) | **100%** (10/10) | 90% (9/10) | 90% (9/10) |
| Mean latency | 57.18s | 27.51s | 29.46s |
| Latency as % of R1 | — | 48.1% | 51.5% |

## P-O: HOLDS

R1 ≥ R3 ≥ R2 (1.0 ≥ 0.9 ≥ 0.9 — R3 and R2 tied, both satisfy the non-strict ordering) and R3 is within 1 question of R1 (9 vs. 10). The assembled system — reload the 2b skeleton's KV state, append reconstructed folded content out of position, answer — preserves gold-level answer quality on 9 of 10 questions despite the positional naivety we built in deliberately.

**The one miss (Q2, both R2 and R3, identically)**: the warranty-period question. Gold: "sixty (60) days." Both stitched regimes answered "45 days" — which is a real number from the *same clause* (clause 2), but it's the *cure period* (45 days), not the warranty period (60 days). R1 (full original document order) doesn't make this mistake. This is a clean, specific failure mode: clause 2 states the warranty period early and the cure period later; when part of the clause folds and part stays live, the reconstituted skeleton+appended-folded-content structure can separate these two numbers from their natural proximity and ordering, and the model attributes the wrong one to the question. Seam recompute (R3) did not fix this — identical wrong answer to R2 — which makes sense: this isn't a positional/attention-boundary artifact at a single seam, it's a document-structure-scrambling effect from how the live/folded split cut through the middle of one clause's internal narrative. A wider seam wouldn't obviously help either, since the two numbers may not both be within any reasonably-sized window of the actual seam point.

## P-P: FAILS, but the reason was already known before this iteration ran

R2/R3 latency lands around ~48-52% of R1's, not under the pre-registered 25% bar. This is not a new puzzle — it's the direct, mechanistic consequence of Iteration 2b's own measured fold fraction (38.3%). R2/R3 still have to prefill the *entire reconstructed folded_text* (appended fresh, not reloaded) — only the *live skeleton* portion benefits from the ~380-1200x reload speedup Iteration 1 measured. With only 38.3% of the document folded, the best-case latency reduction is bounded by roughly `(1 - fold_fraction)` of the original prefill cost, plus a fixed ~60-token generation cost that doesn't shrink with either regime. A 25% latency target implicitly assumed a much higher fold fraction than 2b's own recalibration produced. This is worth stating plainly rather than treated as a fresh mystery: **the latency ceiling here is downstream of the same fold-fraction number already reported and explained in Iteration 2b/2c** — it is not evidence against the stitching mechanism itself.

The pre-registered escalation path ("R3 materially below R1 → escalate seam width") does not trigger — R3 is not materially below R1 on accuracy (within 1 question, the explicit tolerance), so there's no basis to spend the escalation budget on wider seams; the shortfall is entirely on the latency axis, which the escalation path doesn't address anyway (wider seams cost more prefill, not less).

## Net for Iteration 4

The three-tier architecture is validated at prototype scale for answer quality (P-O), with a known, explained latency ceiling tied to Iteration 2b's fold fraction rather than the stitching mechanism itself. Iteration 4 should include the KV-reload regime (R2/R3-style) in its frontier table alongside the other condensed regimes, with this latency ceiling reported as context rather than presented as if it approached the mechanism's theoretical best case.
