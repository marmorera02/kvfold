# Task 2 results: open-world control — GATE FAILS

Per the pre-registered gate ("P-H and P-I hold → E3/E4 unlocked; else report FP anatomy and stop for owner decision"): **both P-H and P-I fail. E3/E4 remain locked.** The pipeline, as validated through Phase D, is a closed-world candidate-ranker — it does not yet know how to say "zero" or "two" reliably.

## P-G: delta-only fails open-world by construction — confirmed as predicted

Delta-only always names the single lowest-z_delta_L2 section as "the contradiction," regardless of whether one exists:

| Seed | Delta-only candidate section(s) |
|---|---|
| 3 (zero contradictions) | {8, 9} |
| 4 (zero contradictions) | {9, 7} |
| 5 (two contradictions) | {9, 10} (only n2a's section; n2b's section 8 never enters the candidate set) |

Exactly as predicted: delta-only has no mechanism to report "nothing here" or "two things here" — it always picks exactly the bottom-ranked section(s), whether or not anything is actually wrong there. This is the documented reason delta must never be the verdict on its own, confirmed empirically rather than just asserted.

## P-H: FAILS — 3 false contradictions across the two zero-contradiction seeds (allowed slack: 1)

Combined/delta-routed-NLI requires BOTH conditions (candidate section AND NLI ≥ 0.9) before calling "contradiction." On seeds 3 and 4 — corpora deliberately built with **zero** genuine contradictions — this still fires:

| Seed | Clause | NLI score | What it's being confused with |
|---|---|---|---|
| 3 | 74 (force majeure) | 0.991 | clause 66, a different force-majeure clause — near-identical template, same "acts of God, war, civil unrest..." boilerplate |
| 3 | 83 (IP ownership) | 0.988 | clause 32, a different IP-ownership clause — both state Buyer owns IP, differ in which product/scope |
| 4 | 85 (insolvency termination) | 0.913 | clause 60, a different insolvency-termination clause — near-identical template, different day-counts |

**This is the same failure mode Phase B was built to fix, at smaller scale.** Phase B's assumption-set + referent guard cut the false-positive *rate* from 65% to 8–15% by shrinking the candidate pool to genuinely general/standing clauses. But our corpus's template-cycling design means *multiple independent instances* of the same general clause TYPE appear throughout the document (several force-majeure clauses, several IP-ownership clauses, several termination clauses) — all correctly classified as "general" by the assumption extractor, all correctly retained in the assumption set, and then NLI still can't tell "two independent instances of the same clause type" from "a genuine contradiction," because they really are structurally near-identical and differ mainly in specific numbers. The fix reduced the *population* where this can happen; it didn't eliminate the underlying NLI weakness within that smaller population.

## P-I: FAILS — n2a caught, n2b missed, for two independent, compounding reasons

Seed 5 (two contradictions: n2a at clause 87 contradicting the 3x-reinforced liability cap; n2b at clause 78 contradicting a newly-introduced, single-statement exclusivity frame at clause 12):

| | n2a (liability) | n2b (exclusivity) |
|---|---|---|
| z_delta_L2 rank (Qwen, of 10) | 1 (in routing candidates) | 6 (outside) |
| z_delta_L2 rank (Llama, of 10) | 1 (in routing candidates) | 3 (outside) |
| Extracted into assumption set? | yes (clauses 8/25/50) | **no — clause 12 never extracted** |
| Raw NLI score | 0.989 | 0.980 (but see below) |
| Caught by config (iv) delta-routed-NLI? | **yes** | **no** |

n2a is caught cleanly, reproducing Phase D. n2b fails for two independent, compounding reasons:

1. **Routing miss**: its section never enters the bottom-15th-percentile candidate set under either scorer. Widening the percentile cutoff would need to reach roughly the 30th percentile (Llama) to 60th percentile (Qwen) to catch it — at the 60th-percentile end, that's checking NLI on the majority of the corpus, defeating the compute-saving point of routing entirely.
2. **Assumption-extraction miss** (independent of routing): clause 12's exclusivity statement was never extracted into the assumption set in the first place (confirmed: `e2_assumptions_seed5.json` includes 8/25/50 but not 12). Even if routing had somehow flagged section 8, there would be no correct premise to check it against.

One nuance worth flagging precisely: n2b's raw NLI score (0.980, computed by `nli_v2.py` independent of routing) is high enough to clear the 0.9 threshold — but *not because it was correctly compared against the exclusivity frame it actually contradicts*. It was matched against the liability-cap assumptions (clauses 8/25/50), passing the referent guard on the shared word "liability" (n2b's text ends "...without incurring any liability to Supplier"). This is a right-answer-for-the-wrong-reason case: NLI-only (config i, which checks every unit regardless of routing) would have caught n2b, but not because it understood the exclusivity conflict — a coincidental lexical overlap did the work instead. Config (iv) never gets the chance either way, since routing gates whether NLI runs at all.

**Why n2a and n2b differ mechanically**: n2a's frame (the liability cap) was deliberately restated three times across the document, per the original corpus design. That repetition is what produces the strong topic-priming effect Phase A/C diagnosed — and, incidentally, is also what got it correctly extracted as a recognizable "general rule" by the assumption step. n2b's frame (exclusivity) was stated exactly once. Weaker repetition means weaker topic-priming (so its delta doesn't get pushed to the bottom) and, apparently, a less recognizable "general rule" signature (so it wasn't extracted at all). **The mechanism that makes routing and assumption-extraction work is itself downstream of how many times a frame gets repeated — not of whether the content actually matters.** This is a real limitation, not a fluke of this specific seed.

## Gate verdict

**E3/E4 remain locked.** Per the directive: "P-H fails → the pipeline is a candidate-ranker only; report false-positive anatomy... and stop for owner decision before E3/E4." Both P-H and P-I fail, independently, for well-understood mechanical reasons:

- The pipeline cannot currently distinguish "zero contradictions" from "one," because delta-routing always nominates *something* and NLI's residual brittleness (even after Phase B's fix) still occasionally confirms a false nomination when the corpus has multiple near-duplicate instances of the same general clause type.
- The pipeline cannot currently find a *second* contradiction unless its underlying frame is repeated at least as often as the first — both the routing signal and the assumption-extraction signal are keyed to repetition density, not to logical importance.

This is a stronger, more precisely diagnosed negative result than a simple "it doesn't generalize" — it identifies *which* two independent components (routing threshold, assumption extraction) would each need to change, and *why* the current design's apparent 11/12 success on the closed-world corpus was riding on a favorable coincidence (the one planted contradiction happened to sit in the most-reinforced frame) rather than a general property of the pipeline.
