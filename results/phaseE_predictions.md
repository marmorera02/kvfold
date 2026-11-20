# Phase E — pre-registration (written before running)

**Question**: does the locator/router framing validated on the synthetic
templated-legal corpus (Phases A-D: Δ-based routing to the bottom-15th-
percentile L2 section, assumption-set NLI ≥ 0.9 as the contradiction
verdict) transfer to lexically varied, human-written, non-templated prose
— and specifically, does moving off a small-template-cycling corpus reduce
or eliminate the near-duplicate-clause false-positive mechanism Task 2
diagnosed as the reason the pipeline couldn't reliably report "zero
contradictions"?

## Corpus (`gen_corpus_wiki.py`)

Real Wikipedia prose (Eiffel Tower, Golden Gate Bridge, Titanic REST API
extracts, `data/wiki_raw/`), split into ~100 short passages. A specific,
checkable fact (Eiffel Tower's height, 330 metres) is restated 3x in
different phrasings (mirroring the legal corpus's 3x liability-cap
reinforcement) plus one weird-benign aside, one novel/unrelated insertion,
and one tagged ordinary control passage — all authored for this module,
inserted into genuine Wikipedia sentences, following the same
experimental-control logic the legal corpus's own needles already used.

- **wikiA**: reinforced fact ×3 + one contradiction (345m height claim).
- **wikiB**: reinforced fact ×3, zero contradictions (mirrors Task 2's
  seeds 3/4 zero-contradiction control).

Pipeline mechanism is unchanged from the validated legal-corpus version
(same Δ formula, same z-scoring, same bottom-15th-percentile routing cutoff,
same 0.9 NLI bar, same referent-guard logic) — only prompt wording
(`build_levels_wiki.py`, `assumptions_wiki.py`) and the referent guard's
domain-specific exclusion lists (`nli_v2_wiki.py`) are adapted, since
reusing contract-specific wording on non-contract prose would confound the
register-transfer question this phase exists to answer. No numeric
threshold differs from the legal-corpus pipeline.

## P-J (zero-contradiction precision, mirrors Task 2's P-H)
wikiB reports zero false contradictions (routed section + assumption-set
NLI ≥ 0.9). Directional prediction, not a fixed target: since this corpus
has no small set of cycled templates — every passage is lexically distinct
prose — the near-duplicate-same-template mechanism Task 2 diagnosed should
have nothing to attach to, so false positives are expected to be closer to
zero than the legal corpus's zero-contradiction seeds (which had 2 and 1).

## P-K (recall, mirrors Task 2's P-I)
wikiA's contradiction (position 85) is both (a) routed into the
bottom-15th-percentile L2 candidate section under Δ, and (b) confirmed by
assumption-set NLI ≥ 0.9 against the reinforced-fact assumption.

## Gate
- P-J and P-K both hold → the locator/router framing generalizes to real
  prose; the legal corpus's precision failure was a property of template
  cycling specifically, not of the underlying mechanism.
- P-K holds, P-J fails → recall transfers, precision does not; report the
  false-positive anatomy on wikiB with the same detail as Task 2's report,
  and identify whether the mechanism differs from the same-template one.
- P-K fails → the routing/extraction mechanism itself doesn't transfer to
  non-templated prose (a stronger negative result than Task 2's, since it
  would mean the closed-world 11/12 result was *also* riding on a
  template-corpus-specific property, not just on repetition density).

Negative results reported at the same detail as positive ones, per the
program's standing rule. No threshold retuning after seeing results, on
either P-J or P-K, without a new directive.
