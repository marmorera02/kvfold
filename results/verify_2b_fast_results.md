# Fold-detection speed fix: verify_2b_fast (owner-directed optimization)

## Why this exists

While preparing E11 (the owner's requested 3B-vs-7B, naked-vs-enhanced crossover test), the 3B baseline's fold-detection step ("ingest") was still running after 60+ minutes on the same ~1,500-token boilerplate document that Iteration 2b/2c/3/4 all previously processed. The owner's diagnosis: *"the ingest algorithm is correct but far too compute expensive... this will NEVER outrun just KV caching and running a prompt."* This is a direct, correct call — investigated and fixed rather than accepted as inherent.

## Root cause

`verify_2b.run_transaction_2b` processes candidate spans (`watershed.py`'s proposals) in **cost-ascending order**, not document order. For each of the ~250 candidates on this document, it rebuilds the preceding live context as a string and calls `score_tokens()`, which does a full `llm.reset()` + fresh prefill of that entire context — every single time. Since candidates are visited in jumbled (non-document) order and context length grows with document position, this is effectively **O(candidates × document length)** — not O(document length). Nothing about surprise-gated folding requires this; it's a fixable inefficiency in how context got rebuilt.

## Fix and the three bugs found getting it right

`src/e3_folding/verify_2b_fast.py` makes one linear, left-to-right pass over the document, extending a single persistent KV cache incrementally instead of re-prefilling, with one save/load checkpoint per valley run (not per atom) for rollback when part of a run folds. Getting this correct took three iterations, each diagnosed from first principles rather than patched blindly:

1. **BPE whole-string-vs-piece mismatch.** `reconstruct_and_check_2b` (reused unchanged for validation) always tokenizes an accumulated context as one whole string. Building the cache by tokenizing each new segment separately and appending disagreed with this: GPT-style BPE pretokenizes on "leading-space + word" chunks, so a chunk's tokenization is only stable regardless of preceding context if the split falls exactly at such an edge — and this project's atoms each carry their own *trailing* separator, which is not such an edge (confirmed: "Clause 1. " tokenized alone ends in a bare space token; "Clause 1. All..." tokenized as one string merges that space into the next word instead). First attempt: 307.9s, reconstruction 2/10.
2. **The fallback this exposed fires almost always, not rarely.** A whole-string re-tokenize-and-compare check with a full reset+re-eval fallback made this explicit: 200/226 checks failed in a 400-atom diagnostic, silently reintroducing the exact O(candidates × position) blowup this rewrite exists to eliminate (confirmed: 8,658.2s — worse than doing nothing — with reconstruction now 10/10, i.e. correct but pointless). Fix: never let committed text end in whitespace; hold trailing whitespace back as `pending_ws` and prepend it to whatever comes next, so every tokenize() boundary falls at a stable chunk edge (verified directly: this makes the incremental token sequence match a whole-string tokenization exactly). Third attempt: 236.1s, reconstruction 9/10.
3. **Two further bugs, found by chasing the residual 1-2/10 reconstruction failures rather than accepting "close enough":**
   - The held-back whitespace's own token count was computed by tokenizing it *in isolation*, then used as a boundary offset — but a leading space commonly merges into a single token with the word that follows it, so "tokens for pending_ws alone" isn't the same count as "tokens consumed by pending_ws once followed by real text." Fixed by folding the pending whitespace into the first atom's own cumulative-prefix span instead of pre-computing a separate offset.
   - The remaining failures were the **E1 lesson recurring at a new boundary**: `run_transaction_2b_fast` scores each run on top of a long-running, incrementally-extended cache, while `reconstruct_and_check_2b`'s validation re-derives ranks via an *isolated* fresh eval of just that run's own context+text — a different computation shape. Per the project's own founding finding (batched vs. sequential eval give measurably different logits on this stack), two computations of "the same" thing are only comparable if they share a shape. Fixed with a new `reconstruct_and_check_2b_fast` that replays the *actual* incremental history up to the target commit (only for the ~10 sampled commits being validated, not the whole pass, so this stays cheap) rather than an isolated re-eval.

## Final, fully validated result

| | Value |
|---|---|
| `document_data` (whole-doc score, unchanged step) | 54.8s |
| `run_transaction_2b_fast` (the fixed step) | 243.6s |
| **Total** | **~298s** |
| Original (`run_transaction_2b`, cost-ascending, killed after) | 60+ minutes, did not finish |
| Reconstruction (10 sampled commits, shape-consistent check) | **10/10 exact** |
| Gated-atom violations | **0** |
| Fold fraction (fast, document order) | 0.4931 |
| Fold fraction (original, cost order, from `cycle3_iteration2b_run.json`) | 0.3827 |

**At least a ~12x speedup** (the original never finished a full run to compare against directly — it was still running past the 60-minute mark when killed).

## Disclosed behavior change

This processes valley runs in **document order**, not `run_transaction_2b`'s cost-ascending order — a deliberate, necessary change to make single-pass incremental caching possible at all. Hard correctness guarantees (exact reconstruction, zero gated-atom leakage) are enforced identically and independently per commit regardless of order and are unaffected. The **fold fraction differs** (0.4931 vs 0.3827, +0.11) because a run's live context now always reflects strictly-earlier document positions' fold decisions rather than whatever happened to be decided earlier in cost order (which could be from anywhere in the document) — this is reported as a real, disclosed consequence of the order change, not hidden or averaged away. Downstream iterations (E11 and beyond) using this fast path will see this higher fold fraction; any comparison back to Iteration 2b/2c/3/4's own numbers (which used cost-ascending order) should account for this rather than treating them as directly interchangeable.
