# Phase B results: assumption-set + referent-guarded NLI vs. v1's top-5-cosine

## False-positive rate (fraction of all 100 L1 units scoring above threshold)

| Seed | | mean | median | frac ≥0.5 | frac ≥0.9 |
|---|---|---|---|---|---|
| 0 | v1 (top-5 cosine) | 0.628 | 0.864 | 0.65 | 0.48 |
| 0 | v2 (final, both guard fixes) | **0.175** | **0.051** | **0.13** | **0.09** |
| 1 | v1 | 0.673 | 0.967 | 0.67 | 0.59 |
| 1 | v2 (final) | **0.214** | **0.074** | **0.15** | **0.10** |
| 2 | v1 | 0.647 | 0.941 | 0.65 | 0.53 |
| 2 | v2 (final) | **0.133** | **0.037** | **0.08** | **0.06** |

Median dropped from "most clauses score above 0.86" (v1) to "most clauses score below 0.08" (v2) in every seed — the false-positive problem is essentially fixed. frac≥0.5 dropped from ~65% to 8–15%.

Two rounds of referent-guard refinement got there. First pass (assumption-set + basic guard) landed at 17/19/9% frac≥0.5 across seeds 0/1/2 — already a large improvement, but inspecting the residual false positives found the guard still passing on trivial shared nouns: "Supplier"/"Buyer" (present in nearly every clause; excluded), then "day" (as in "60 days" vs. "90 days" — any two clauses mentioning a time period shared this; excluded along with time/period/notice/amount/provision/event), and product-name head nouns like "housing"/"module" (coincidental slot-fills, not genuine shared referents across different clause categories; excluded). Each round of exclusions was verified against the actual flagged pairs before being kept, not applied speculatively.

## N2 (the true contradiction) is still caught, every time

`max_contradiction = 0.9887` for clause 87 in all three seeds (identical value because the liability-cap and N2 clause text is fixed across seeds; only slot-filled boilerplate elsewhere varies). In every seed it's matched against clause 8 and/or clause 50 — the actual liability-cap restatements — not an arbitrary same-template neighbor. This is exactly the intended mechanism: recall (catching the true positive) survived the redesign; precision (not flagging everything else) improved by roughly 4–7x.

## What actually fixed it

Two changes, both necessary:

1. **Assumption-set premises instead of top-5-cosine neighbors.** The first attempt at this (single-shot "restate or say NONE") failed completely — the model restated all 100/100 units instead of discriminating, because contract clauses are grammatically rule-shaped even when their content is routine ("Buyer shall pay..." *reads* like a general rule). Splitting into a two-step classify-then-restate prompt, anchored with four contrastive few-shot examples, fixed this: 16–20 assumptions extracted per seed (in the range the directive expected), reliably including the three liability-cap restatements.
2. **Referent guard.** Caught and fixed a real bug before it could mask the result: "Supplier" and "Buyer" are proper nouns appearing in nearly every clause, so left unexcluded they'd make the guard pass almost every pair on a trivial shared-party-name basis, defeating its purpose. Excluding them (and other structural nouns: Agreement, Clause, Party, Schedule, Goods, Product) restored real discrimination — the guard now passes premises sharing substantive content nouns (liability, damages, caps) rather than just the ambient contract vocabulary.

## Remaining false positives (9–19%)

Not zero. Worth a quick look before Phase D to see whether the residual false positives cluster around a specific assumption type (e.g., the confidentiality-survival clauses, which recur with different day-counts across the corpus in the same way the original template-clone problem worked, just now restricted to a much smaller candidate pool). Not blocking — this is a large enough improvement to proceed to Phase D's ablation.
