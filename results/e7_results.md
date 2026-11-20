# E7 (Cycle 3 parallel track) results: structured ledger extraction

Per `e7_predictions.md`. Single prompt iteration (v1) — recall cleared the pre-registered bar on the first attempt, so the two further revisions the directive permitted were not triggered and were not used. Full data: `results/e7_run.json`, per-seed ledgers `results/e7_ledger_seed{3,4,5}.json`.

## P-Q (extraction recall): HOLDS

| Seed | Should-extract clauses | Extracted | Misses |
|---|---|---|---|
| 3 | {8, 25, 50} | 3/3 | none |
| 4 | {8, 25, 50} | 3/3 | none |
| 5 | {8, 12, 25, 50, 78, 87} | 5/6 | **78** |

Overall recall: **11/12 = 91.7%** (≥ 0.9 bar). **Clause 12 — the single-mention exclusivity frame Task 2's assumption-extractor never found at all — was extracted correctly.** This is the recall test's primary purpose, and it succeeds: broadening the classify prompt's few-shot set across topics (liability, exclusivity, ownership), rather than anchoring only on liability-flavored caps/exclusions the way `assumptions.py`'s classifier did, fixes the specific miss Task 2 diagnosed.

**The one miss (clause 78, seed 5) is a different, new failure mode, not a recurrence of the old one.** Clause 78 is n2b's contradiction itself — the override of clause 12's exclusivity frame ("Buyer may procure the Goods from any third-party supplier at its sole discretion... without incurring any liability to Supplier"). It was classified ROUTINE, not RULE. Compare the two contradictions' override language directly:

- n2a's override (clause 87): *"shall be unlimited... shall not be subject to the limitations, caps, or exclusions... Buyer shall be entitled to recover the full amount"* — assertive, absolute modal language (shall/shall not/entitled), classified RULE correctly.
- n2b's override (clause 78): *"may procure... at its sole discretion... without restriction"* — permissive/discretionary language, classified ROUTINE incorrectly.

**Working hypothesis**: the classifier is tracking modal assertiveness ("shall," "must," absolute caps/exclusions) as a proxy for "standing rule," not the logical function of overriding an established scope. An override phrased as a granted permission ("may... at its sole discretion") reads less rule-shaped than one phrased as an absolute override ("shall be unlimited"), even though both are logically the same move — restating a scope's value differently from an earlier clause. This is a real, specific limitation of the classify prompt, distinct from and narrower than Task 2's miss (which failed on the *establishing* clause, 12, not its override) — reported plainly rather than folded into the same explanation.

## P-R (Task-2 replay, deterministic conflict rule): FAILS

| Seed | Expected | Actual conflicts found | Zero-FP / catch? |
|---|---|---|---|
| 3 (zero contradictions) | 0 | **71** | FAILS |
| 4 (zero contradictions) | 0 | **127** | FAILS |
| 5 — n2a (87 vs. 8/25/50) | caught | caught | **PASSES** |
| 5 — n2b (78 vs. 12) | caught | not caught (78 has no ledger entry — see P-Q) | FAILS (by construction, given the P-Q miss) |

The one genuine positive the rule needed to find (n2a) it finds correctly. But it also produces a very large false-positive count on the two corpora deliberately built with **zero** real contradictions — far worse than Task 2's NLI-based approach (3 false positives across both zero-contradiction seeds combined; here, 71 and 127 respectively, on a single seed each).

### Root cause, inspected directly from the flagged pairs (not inferred)

Sampling seed 3's 71 flagged pairs shows one mechanism accounts for nearly all of them: **the VALUE field is being contaminated by incidental procedural numbers**, despite the extraction prompt explicitly instructing the model to capture "the core permission, restriction, or cap amount... not an incidental procedural number." Example (clause 6 vs. clauses 15/23/41/58/67/75, all independent IP-ownership template instances, same category, same underlying rule — "Buyer owns, Supplier gets limited-use rights"):

```
6:  IP ownership | use rights      | limited use for Agreement performance, return or destruction within 30 days of termination
23: IP ownership | rights acquired | limited use for Agreement performance, return or destruction within sixty days of termination
```

Both clauses state the *same* rule (limited-use license, materials returned/destroyed on termination); they differ only in the slot-filled "return within N days" number, which the corpus generator (`gen_corpus.py`) randomly fills per-instance and which is not the actual point of contradiction-or-agreement. The conflict rule's number-signature check treats "30 days" vs. "sixty days" as different values and flags a conflict. **This is the identical failure anatomy Task 2 diagnosed for the NLI pipeline** (near-duplicate same-template clauses of the same TYPE, differing only in incidental slot-filled specifics, mistaken for genuine disagreements) — reproduced here through a different specific mechanism (deterministic value-signature mismatch instead of a black-box NLI score), which rules out "switch to a deterministic rule and the near-duplicate-template problem goes away" as a hypothesis. It doesn't; the same corpus property (heavy template cycling within categories) defeats both approaches, just via different code paths.

A second, smaller contributor: the concept-word-overlap gate (`_concept_words`, meant to play the same role as `nli_v2.py`'s referent guard) is itself too permissive in the same way the *original*, unfixed referent guard was — e.g. two unrelated clauses sharing the word "calibration" (a coincidental category word, not a genuine shared referent) pass the gate. `nli_v2.py` needed two rounds of stopword-set expansion to fix this exact problem; `conflicts.py`'s word list was not given the same treatment, since fixing it now would be tuning the conflict rule *after* seeing which false positives it produces on the test seeds — exactly the kind of after-the-fact threshold-fitting the pre-registration explicitly ruled out ("not permitted to retune the conflict rule itself, since that would be threshold-fitting to the known answer").

## Gate verdict

Per the pre-registered gate: **P-Q holds, P-R fails → extraction generalizes but the deterministic conflict rule does not.** Reported at the same detail as a pass, per standing rule.

**What this establishes, net:** structured extraction is a genuine, verified improvement over Task 2's assumption-set approach on the *recall* axis — it catches the stated-once, novel-topic frame (clause 12) that assumption-extraction's narrower few-shot anchoring missed, via a straightforward, honest fix (broaden the classify prompt's topic coverage). It does **not**, as implemented, solve the *precision* problem that closed the gate in Task 2 — moving to structured, deterministic fields did not sidestep the corpus's core difficulty (many independent, near-identical instances of the same clause TYPE, differing only in incidental numbers), because the extraction step still needs semantic judgment (which number is "the point" vs. incidental) that a 3B model does not reliably provide, and no purely mechanical rule downstream can recover that judgment once it's missing upstream.

**Per the pre-registered gate language, since this is not the "P-Q fails" case:** a "detection-v3" direction is noted as a candidate follow-on (e.g., a referent guard for `conflicts.py` symmetric to `nli_v2.py`'s fixed one, and/or an explicit "is this number incidental or definitional" sub-step before storing VALUE) — but building and validating it is out of scope for this cycle. The detection track, across both Task 2's NLI approach and E7's structured approach, closes at this model scale without that follow-on work: **neither approach currently reports "zero contradictions" reliably on a corpus with heavy same-category template repetition**, which is the open-world property both zero-contradiction control seeds were specifically designed to test.
