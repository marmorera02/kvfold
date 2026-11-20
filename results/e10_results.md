# E10 results: assumption-contrast scoring (Δ±)

Per `e10_predictions.md`. Full data: `results/e10_contrast_qwen.json`, `results/e10_contrast_llama.json`. Sanity check (5 hand-picked pairs, both scorers): **all passed** — every pair's swapped recomputation gave the exact algebraic negation of the original (|Δ±_normal + Δ±_swapped| < 1e-6), so the mechanism itself is verified sound before any of the numbers below are interpreted. 911/911 pairs scored under both scorers; 0 excluded by the length-match gate.

## P-W (recall): FAILS, both scorers

| | Qwen | Llama |
|---|---|---|
| All 16 TRUE_CONTRADICTION pairs Δ± > 0? | **No — 1/16 negative** (wikiA 85×8: −0.149) | **No — 2/16 negative** (wikiA 85×8: −0.019; wikiA 85×40: −0.201) |
| Weak form: TRUE pairs in top decile of full distribution | 6/16 | 9/16 |

The 3 liability-cap pairs (seed 0/1/2/5, hyp 87) and the seed-5 n2b×clause-12 special pair are positive under both scorers — recovering the original diagnosed miss (clause 12) as a *positive-direction* Δ± is itself notable. But **wikiA's contradiction fails outright** under both scorers whenever paired against 2 of its 3 candidate premises, consistent with — not contradicting — the confound already flagged in `phaseE_results.md` (the contradiction is phrased as a plausible timeline update, not a same-moment negation). This is a real, cross-model-replicated failure, not scorer noise.

**A consistent, notable sub-finding**: which of the 3 paraphrased liability-cap sentences serves as the premise measurably changes Δ±'s magnitude, in the *same rank order* under both scorers (premise 25 lowest, premise 8 middle, premise 50 highest — Qwen: 0.029/0.042/0.061; Llama: 0.032/0.038/0.115). The three premises assert an identical fact; the sensitivity is to phrasing, not content, and it replicates across architectures — this is itself evidence that Δ± is picking up something about surface form, not purely the underlying logical relationship.

## P-X (precision): FAILS in aggregate, both scorers — but with an important decomposition

| | Qwen | Llama |
|---|---|---|
| KNOWN_FP pairs with Δ± ≤ 0 (all 595) | 55.5% | 61.8% |
| **The 3 explicit Task-2 false positives** (seed3 74×66, seed3 83×32, seed4 85×60) | **3/3 correctly ≤ 0** | **3/3 correctly ≤ 0** |
| wikiB guard-passed pairs (592) with Δ± ≤ 0 | 55.2% | 61.7% |

Against the pre-registered ≥90% bar, P-X fails clearly under both scorers. But the failure is not uniform: **the exact 3 pairs that fooled the original NLI+routing pipeline (Task 2) are correctly assigned non-positive Δ± by both scorers, 6/6 across the two models** — a genuine, replicated positive result on precisely the cases the corrected contrast was motivated to fix. The much larger wikiB population (592 pairs, auto-labeled KNOWN_FP because ground truth has zero contradictions there) sits close to chance (55–62% ≤ 0) — consistent with "no reliable signal, sign effectively random" for pairs that were never a targeted adversarial case to begin with, rather than a specific failure mode. The aggregate number is dominated by this much larger, less-targeted population; both facts are reported since neither on its own is the full picture.

## P-Y (separation): FAILS against the 0.9 bar — but is real and consistent, not chance

| | Qwen | Llama |
|---|---|---|
| AUC (TRUE vs. KNOWN_FP+NEUTRAL background) | 0.810 | 0.829 |

Both scorers land in a tight band (0.81–0.83), well above chance (0.5) and a substantial, replicated improvement over Phase D's finding for the *original* contrast (AUC(Δ, novel vs. boilerplate) = 0.528, chance level — the finding that motivated this whole redesign). This is the clearest positive result in this experiment: **correcting the contrast (minimal pair vs. empty-context) measurably improves separability, consistently, across two independent model architectures.** It is not, however, anywhere near the 0.9 bar this experiment pre-registered as the threshold for a usable flag mechanism.

**Noise floor, both reference points, both scorers**:
- Against the directive's stated 0.3 bits/token figure (flagged in the pre-registration as **unverified** against the stored E1 artifact — closest analog found there was `mean_abs_logit_diff: 0.109` at a different shape boundary): **all 16/16 TRUE pairs, both scorers, have |Δ±| below this floor.** Every true-contradiction signal in this experiment is smaller in magnitude than the cited reference noise level.
- Against the empirically-measured NEUTRAL-pair spread (this experiment's own same-shape noise characterization): Qwen std = 0.0397, Llama std = 0.0328. By this measure, 6/16 TRUE pairs are inside 1 empirical std for *both* scorers (not necessarily the same 6 pairs, but the same count) — meaning roughly 10/16 pairs do exceed this more targeted, in-experiment noise estimate, under both scorers.

These two floors tell different stories and both are reported rather than picking the more convenient one: against the directive's cited figure, the signal is uniformly sub-floor; against this experiment's own measured noise, about 60% of true pairs clear it.

## P-Z (mechanism check)

S(u|¬A) < S(u|A) — the flipped frame actively predicting the contradicting text better than the true frame does — holds for **15/16 pairs under Qwen, 14/16 under Llama** (identical count and near-identical pair set to the strict P-W check, since this is the same condition viewed from the other side of the subtraction). Where it holds, this is the cleanest available demonstration that *direction*, not just magnitude, is being measured; where it doesn't (wikiA's harder pairs), the frame's flip does not help, consistent with the same phrasing confound already identified.

## Secondary check (L0 instead of L1): not run

Per the pre-registration, the secondary check only runs if the primary passes. It does not (P-W and P-X both fail against their pre-registered bars under both scorers), so the L0 variant was not run, per the directive's own conditional — not an oversight.

## Gate verdict

None of the three pre-registered gate branches applies cleanly, and that mismatch is itself reported rather than forced into the nearest one:

- **Not** "P-W + P-X + P-Y hold" — none of the three holds against its literal bar, under either scorer.
- **Not** cleanly "P-W fails with margins inside the noise floor, confirming the collapse hypothesis" either — that branch presumes a chance-level result, and this isn't one: AUC 0.81–0.83, replicated across architectures, is a real, non-trivial improvement over the pre-existing chance-level (0.528) finding for the uncorrected contrast.
- **Not** the anticipated "mixed, legal passes / wikiA fails" split — the actual pattern is messier and more informative: legal-corpus pairs are inconsistent *among themselves* depending on which paraphrase serves as premise (though generally positive), wikiA is consistently the hardest register (already partially explained by the register/framing confound from Phase E), and the *specific* false positives the whole detection program has been chasing since Task 2 are, unusually, the one place this new metric performs cleanly (6/6 across two scorers).

**Net, stated plainly**: correcting the contrast from S(u|context)-vs-S(u|∅) to a minimal pair S(u|A)-vs-S(u|¬A) recovers a real, cross-model-replicated, above-chance directional signal (AUC 0.81–0.83, up from 0.528) — the wrong-contrast hypothesis was partially right. But "better than chance" is a much weaker claim than "usable as a flag mechanism": P-W and P-X both fail by wide margins against practical thresholds, and every true-contradiction pair's raw magnitude sits below the directive's own cited noise reference. Surprisal-based direction detection, even with the sharpest contrast tried in this program, does not clear the bar for a flag-generation mechanism at this model scale — closing that question with more precision than either Phase A/C (wrong contrast, chance-level) or a simple restatement of "surprisal doesn't work" would have, by identifying specifically which part of the hypothesis was right (the contrast mattered, measurably) and which part still fails (the resulting signal, even corrected, is too weak and too phrasing-sensitive to flag on).
