# Phase D results: amended pre-registered criteria + 4-config ablation

P-A (original: N2 has highest z_delta_L2) is **refuted** by Phase C and is not evaluated below — it's a settled negative finding, superseded by the amended claims.

## Amended criteria, per seed

| Criterion | Seed 0 | Seed 1 | Seed 2 | Holds ≥2/3? |
|---|---|---|---|---|
| **P-A'**: N2 rank-1 (lowest) at L2, both scorers | rank 1/10 (Qwen), 1/10 (Llama) → **holds** | rank 1/10, 3/10 → fails | rank 2/10, 3/10 → fails | **No (1/3, strict)** |
| **P-B**: z_L0(N2) < z_L0(N1) and < z_L0(N3) | holds | holds | holds | **Yes (3/3)** |
| **P-C**: N1 collapses L0→L1 (z_L1<1 while z_L0>2) | z_L1=2.72, fails | z_L1=2.74, fails | z_L1=2.75, fails | **No (0/3)** |
| **P-D**: NLI(N2)≥0.9 and corpus frac≥0.5 <0.15 | 0.13<0.15, holds | 0.15, fails (exact boundary) | 0.08<0.15, holds | **Yes (2/3)** |

**P-A' nuance**: the strict "always rank exactly 1 under both scorers" claim fails (1/3 seeds), but the weaker, still-meaningful pattern holds in all three: N2 is *always* in the bottom 3 of 10 L2 sections under both scorers (ranks 1,1 / 1,3 / 2,3). It's reliably near the bottom, just not always the single most extreme point. Report both: the precise pre-registered claim is falsified, the qualitative pattern it was trying to capture is not.

**P-C is cleanly refuted**: N1 does not fold away at L1 (z_L1 stays at 2.7+ in every seed, matching z_L0). Whatever N1's oddity is, summarization doesn't remove it the way the pyramid's core "L1 folds away benign surface weirdness" claim assumed.

## P-E: delta-based NLI routing

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| N2 in candidate (bottom-15th-pctile) section? | yes | yes | yes |
| NLI cost reduction | 80% (20/100 units checked) | 80% | 80% |
| FP rate, full corpus | 0.13 | 0.15 | 0.08 |
| FP rate, candidates only | **0.30** | **0.35** | **0.20** |

Recall is preserved perfectly (N2 caught in the routed subset, all 3 seeds) at a genuine 80% compute reduction. But precision on the routed subset is *worse*, not better, than the full-corpus rate — roughly 2–2.5x higher false-positive concentration. This makes sense given the topic-priming explanation from Phase A/C: the sections that get routed (lowest delta = most topically-saturated) are exactly the sections most likely to contain other same-template, same-theme clauses that still trip NLI's residual false-positive tendency. **Routing saves compute and preserves recall; it does not improve precision, and shouldn't be described as if it does.**

## P-F: delta as a novelty axis

|Predicted| Actual |
|---|---|
| N3: \|z_delta_L2\| < 0.5 | **Holds, all 3 seeds** (0.41, -0.47, -0.39) |
| Clean-section (no needle, no frame) median clearly negative | **Does not hold** — per-seed medians are -0.67, +0.01, +0.57; pooled median -0.09, essentially zero, not clearly negative |
| AUC(delta, novel vs. boilerplate) | **0.528** (chance = 0.5) |

**Refuted.** N3 does sit near zero, but so does everything else in this corpus once you look at the full spread of clean sections (they range from -2.07 to +0.79) — N3 isn't distinctively "near zero," the whole clean-section population is high-variance and centered near zero. The AUC confirms it: delta at L2 has essentially no discriminating power between novel and ordinary content. This part of Phase A's original prediction doesn't survive contact with a proper AUC check.

## Ablation table: 4-way classification accuracy

| Config | Seed 0 | Seed 1 | Seed 2 | Total |
|---|---|---|---|---|
| (i) NLI-only | 2/4 | 2/4 | 2/4 | **6/12** |
| (ii) Delta-only | 4/4 | 4/4 | 3/4 | **11/12** |
| (iii) Combined | 4/4 | 4/4 | 3/4 | **11/12** |
| (iv) Delta-routed NLI | 4/4 | 4/4 | 3/4 | **11/12** |

NLI-only is structurally capped at 2/4 per seed: it can only ever say "contradiction" or "not," so weird-benign and novel are always misclassified as boilerplate by construction — not a bug, an honest limitation of a single-signal detector. Delta-only, Combined, and Delta-routed-NLI are identical here (11/12) because the same delta-based "flag the bottom section as contradiction" rule dominates the outcome, and NLI agrees with it on these cases; the one miss (seed 2) is N1 misclassified as "novel" because its z_delta_L2 (0.045) happened to fall inside the same ±0.5 band used for N3 — the novel/weird-benign split via a single fixed threshold is not fully reliable, even though contradiction-routing is.

**This is the opposite of the "worst case" the directive flagged**: NLI-only does *not* match Combined (6/12 vs. 11/12) — delta is doing real, substantial classification work here. It's just not the work originally hypothesized.

## Verdict (per the mandated framing — not softened)

**Scale-space surprisal is REFUTED as a direct contradiction detector.** The core theorized mechanism — that conditioning on established context should make a contradiction *more* surprising — is backwards on this corpus, confirmed structurally via cross-model control (Phase C), not just as measurement noise. This is consistent with a broader, well-documented pattern in the literature: transformer language models and NLI systems trained on next-token / masked-token objectives are known to be comparatively insensitive to negation and logical reversal, since fluency and topical co-occurrence dominate the probability signal far more than tracking whether a specific earlier claim has been logically negated. A model conditioned on "liability capped at 100%" doesn't get surprised by "liability is unlimited" later — if anything it gets *more* comfortable, because both sentences are drawing on the same primed vocabulary (liability, damages, caps, exclusions), and the model has no mechanism forcing it to check the specific *direction* of the claim against what came before.

**Delta is REPURPOSED, successfully, as an engagement/routing axis rather than a semantic detector.** It reliably identifies which section of the document is most "topically saturated" (lowest delta), which — in this corpus — is exactly where the contradiction was planted, letting NLI compute be spent on ~20% of the corpus with recall fully preserved. Combined with z_L0 for the weird-benign/novel distinction, this gets 11/12 right across three seeds, a large improvement over both the original v1 pipeline (3/12) and NLI-only (6/12). The novelty-axis framing (P-F) does not hold — delta is not a general "is this content novel" signal — but the routing framing (P-E) does, with the explicit caveat that it saves compute and preserves recall without improving precision.

**Net**: the pyramid's defensible value on this corpus is not "surprisal detects contradictions" (refuted) and not purely "routing/compression/provenance" either (delta measurably improves 4-way accuracy beyond NLI alone) — it's a hybrid: delta as an unreliable-but-useful *locator*, real semantic judgment left to NLI once properly gated against a real assumption set.
