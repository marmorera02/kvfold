# Phase A results: conditioning-delta scoring vs. pre-registered predictions

Predictions were committed in `e2_phaseA_predictions.md` before any of this was run. Judged as stated.

## z_delta_L2 across all three seeds (the load-bearing prediction)

| Needle | Seed 0 | Seed 1 | Seed 2 | Predicted |
|---|---|---|---|---|
| N1 weird-benign | **0.788** | -0.634 | **0.045** | negative (boilerplate-like) |
| N2 contradiction | -1.649 | -0.951 | -0.833 | **least negative / most positive** |
| N3 novel | 0.406 | **-0.473** | -0.394 | ≈ 0 |
| N4 control | -0.303 | -0.765 | -0.803 | strongly negative |

Ranking (most positive → most negative) per seed:
- Seed 0: N1 > N3 > N4 > **N2**
- Seed 1: N3 > N1 > N4 > **N2**
- Seed 2: N1 > N3 > N4 ≈ **N2**

## Verdict: the core prediction is FALSIFIED, reproducibly

**N2 has the most negative (or tied-most-negative) z_delta_L2 in all three seeds — the opposite of "least negative / most positive."** This isn't seed noise; the direction is consistent and the effect size is large in two of three seeds (-1.65, -0.95) and merely close in the third (-0.83, essentially tied with N4's -0.80).

**N1 consistently takes the signature that was predicted for N2** — least negative / most positive z_delta_L2 in seeds 0 and 2, second-highest in seed 1 — rather than "behaving like boilerplate" as predicted.

**N4 (boilerplate) matches its prediction**: strongly negative delta, especially visible at L1 (-0.05 to -0.84 vs. +1.1 to +1.8 for all three needles) — context helps ordinary clauses a lot.

**N3 (novel)** is negative but less extreme than N4 at L1 (a partial match to "≈0" — directionally consistent, not literally zero) and inconsistent at L2 (ranges from +0.41 to -0.47 across seeds — no stable signature).

**The L1 clustering is real and worth keeping**: at L1, all three needle types (N1/N2/N3) reliably show much less-negative z_delta than N4 across all three seeds (roughly +1.1 to +1.8 vs. -0.05 to -0.84). Delta *does* separate "anomalous in some way" from "ordinary" at L1 — it just doesn't discriminate *which kind* of anomaly, which is the thing the classifier actually needs.

## Why N2 inverts: a topic-priming hypothesis (distinct from v1's two candidates)

Looking at the raw numbers behind seed 0's N2 delta: S_null = 4.043 bits/token (the section's summary is fairly information-dense read cold), S_ctx = 1.475 (the same text becomes much more predictable once the model has read the rest of the document). The theorized mechanism wanted this drop to be small — the exception should stay hard to predict even with context, because context sets up a contradicting expectation. Instead the drop is the largest of any needle.

A plausible reason: the corpus's liability cap is *deliberately* restated three times (clauses ~8, ~25, ~50, per the corpus design) specifically to "establish the frame." That repetition is exactly the kind of topical priming that makes a model find *any* later sentence about liability/damages/caps more predictable — regardless of whether it agrees or conflicts with the earlier-stated direction. The model isn't reading for logical consistency; it's pattern-matching on "I've seen this vocabulary cluster several times already." N4 gets an even bigger boost for the same reason, but more so — ordinary boilerplate categories (payment, delivery, warranty, ...) repeat 10+ times each across the document, giving far denser structural priming than the liability theme's 3 repetitions. N1 and N3, by contrast, touch content that never recurs anywhere else in the document (an odd notice-clause aside; an off-topic quantum-computing paragraph) — nothing primes them, so context barely moves their surprisal, which is exactly the "resists conditioning" signature actually observed for N1.

If this hypothesis is right, it predicts something checkable in Phase C: if delta inversion is about topic-repetition density rather than which model does the scoring, it should reproduce under a *different* scorer model too (cross-model won't fix it, because the mechanism isn't self-scoring-specific). If Phase C instead shows the inversion disappears under a different scorer, that would point back toward v1's self-scoring-artifact explanation instead. Phase C is the discriminating test for this.

## Implication for Phase D

Delta scoring as implemented does not give a usable (z_L0, z_delta_L2) plane for this corpus — P-A ("N2 has the highest z_delta_L2 of all L2 units") fails outright; N2 is at or near the bottom in every seed. This needs to be carried into Phase D's ablation honestly: Δ is not currently a working detector for contradiction specifically, whatever value NLI (Phase B) turns out to have on its own.
