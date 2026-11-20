# Cycle 3 parallel track E7 — pre-registration (written before running)

**Question**: can a *structured* ledger extraction (SCOPE/ATTRIBUTE/VALUE
triples per clause, deterministic conflict rule over those fields) succeed
where Task 2's assumption-set + NLI approach failed — specifically, catch
the single-mention exclusivity frame (seed 5, clause 12) that the earlier
pipeline never extracted, and not reproduce the earlier pipeline's
same-template-near-duplicate false positives?

## Method (frozen before running; see `src/e7_ledger/extract.py` docstring
for the full prompt design rationale)

1. Classify each L1 clause summary as RULE (states a standing, topic-scoped
   rule) or ROUTINE, using a few-shot prompt spanning multiple rule TOPICS
   (liability, exclusivity, ownership) rather than only liability — the
   diagnosed reason clause 12 was missed by Phase B's assumption extractor
   was topic-narrow few-shot anchoring, not a general classification
   failure.
2. For RULE clauses, extract SCOPE / ATTRIBUTE / VALUE via a second
   few-shot prompt.
3. Conflict detection (`conflicts.py`): two ledger entries are a candidate
   pair if their SCOPE+ATTRIBUTE labels share a content word; flagged
   CONFLICT if their VALUE signatures differ (different numbers extracted,
   or different override/permission polarity keywords). Fully
   deterministic — no NLI model, no LLM call, in this step.

None of the few-shot examples in either prompt reuse text (verbatim or
paraphrased) from clauses 8, 12, 25, 50, 78, or 87 — the exact clauses this
run scores against — so recall is not measuring prompt memorization.

Seeds used: 3, 4, 5 only (the open-world seeds already validated in Task
2). Seed 5 alone contains all six "should-extract" clauses (liability
frame ×3, exclusivity frame, both contradictions), so no additional seeds
are needed for the recall test. Extraction runs once per seed; up to two
further prompt revisions are permitted if recall is poor, each logged with
what changed and why (max 3 total, per the directive) — not permitted to
retune the *conflict* rule (`conflicts.py`) itself, since that would be
threshold-fitting to the known answer.

## P-Q (extraction recall)
Recall ≥ 0.9 over the union of {liability_frame_clause_ids,
extra_frame_clause_ids, contradiction-needle clause ids} across seeds 3, 4,
5 (12 "should-extract" clauses total: 3 liability-frame clauses each in
seeds 3 and 4, which have no contradiction needle by design; liability×3 +
exclusivity×1 + n2a + n2b = 6 in seed 5). **Clause 12 (seed 5) must be
extracted specifically** — this is the one Task 2's pipeline missed, and
is the primary reason this track exists.

## P-R (Task-2 replay via deterministic conflict rule)
- Seeds 3 and 4 (zero genuine contradictions by construction): zero
  conflict pairs reported (this is the axis Task 2's P-H failed on: 3
  false contradictions from same-template near-duplicates).
- Seed 5 (two genuine contradictions): a reported conflict pair
  involving clause 87 and at least one of {8, 25, 50} (n2a), AND a
  reported conflict pair involving clause 78 and clause 12 (n2b) — the
  clause Task 2 could not catch at all.

## Gate
- P-Q holds and P-R holds → structured extraction is a viable replacement
  for assumption-set+NLI at this model scale; note that a full
  detection-v3 directive (integrating this into the L0-L3 pyramid
  pipeline properly, beyond this standalone track) would follow, but
  building it is out of scope for this cycle.
- P-Q fails (extraction itself doesn't generalize) → report plainly that
  the detection track closes at this model scale: even a topic-broadened,
  structured-output approach cannot reliably recognize a stated-once,
  novel-topic standing rule as such, at 3B parameters, training-free. Do
  not use P-R's outcome to rescue this verdict — a conflict rule cannot
  help if the underlying fact was never extracted into the ledger.
- P-Q holds but P-R fails → extraction generalizes but the deterministic
  conflict rule doesn't; report the specific failure mode (which
  same-template clause pairs falsely triggered, or which genuine
  conflict's value signatures matched when they shouldn't have) with the
  same level of detail as a pass.

Negative results at any stage reported at the same detail as positive
ones, per the cycle's standing rule.
