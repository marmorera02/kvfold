# Phase A pre-registration (written before any delta-scoring is run)

`delta_score(unit, level) = S(unit | level_context) − S(unit | ∅)`

- `S(unit | ctx)`: existing v1 mean bits/token, unit conditioned on all preceding units at that level (reused as-is from `e2_scores_seed{K}.json`).
- `S(unit | ∅)`: mean bits/token of the same unit scored in isolation (empty context, single batched-prefill pass — same computation shape as the conditioned pass, per the E1 batched-vs-sequential lesson).

Predicted signatures, per the theory being tested:

| Type | Predicted Δ | Why |
|---|---|---|
| Boilerplate | strongly negative | Context helps prediction — ordinary, on-frame content becomes more predictable once you've seen the rest of the document. |
| N3 (novel) | ≈ 0 | Content is off-topic; the preceding context is irrelevant to predicting it, so conditioning barely changes surprisal either way. |
| N2 (contradiction) | least negative / most positive of the three needles, specifically at L2 | This is the one case where context should actively fight the text — having seen the cap-frame, the exception should be *harder*, not easier, to predict than reading it cold. |
| N1 (weird-benign) | high at L0 in isolation; behaves like boilerplate (Δ negative) by L1/L2 | Surface weirdness is surprising standalone, but once cleaned up in summary form it should fold back into ordinary, context-helped text. |

This file is committed before running `delta_score.py` on any seed. Results below (once run) are judged against these predictions as stated, not adjusted after the fact.
