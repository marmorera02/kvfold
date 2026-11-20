# Cycle 3 — Iteration 2b results: fold-policy recalibration

## Three bugs found and fixed this iteration (in order of discovery)

1. **zstd per-proposal compression's fixed frame overhead** (~9-13 bytes) swamped every small proposal's budget, causing 0% fold fraction everywhere (Iteration 2, first attempt).
2. **Raw varint's byte-alignment floor** — even rank-0 costs a full 8 bits, 8x over a 1 bit/token budget regardless of actual predictability (Iteration 2, second attempt; fixed by switching the fold *decision* to an entropy estimate against a whole-document rank-frequency model, while still storing exact ranks for lossless replay).
3. **Reconstruction context/text split mismatch** (this iteration): mid-run committed groups were reconstructed by re-splitting context at the group boundary (`run_context + earlier atoms` as a new "context" string), which tokenizes differently than the original encoding pass (context = run-start only, text = whole run) — BPE merges aren't guaranteed consistent across a different split point even with identical concatenated content. Confirmed as the cause of a 0/10 reconstruction failure; fixed by always reconstructing against the *original* whole-run context+text and extracting each group's own token slice via stored offsets.

## Results after all three fixes

| | Boilerplate | Prose |
|---|---|---|
| θ_fold (derived: corpus median) | 0.874 bits/token | 3.108 bits/token |
| θ_flag (derived: corpus mean + 1σ) | 5.575 bits/token | 7.667 bits/token |
| P-V analytic prediction | 63.9% | 69.8% |
| **Actual fold fraction** | **38.3%** | **43.1%** |
| — silent tokens | 111 | 56 |
| — flagged tokens | 468 | 253 |
| Flag density (of folded tokens) | 80.8% | 81.9% |
| Residual bits/token, silent | 0.874 | 2.204 |
| Residual bits/token, flagged | 2.252 | 4.190 |
| Reconstruction (10 sampled, ≥3 flagged) | **10/10 exact** | **10/10 exact** |
| Gated-atom violations | 0 | 0 |

Fold fraction is now a real, substantial number (up from Iteration 2's 0.4%/0%), and both hard requirements pass cleanly: reconstruction is byte-exact including flagged folds, and zero gated atoms leak into any committed fold.

## P-V: FAILS — diagnosed, not mysterious

Actual fold fraction is consistently ~25-27 points *below* the analytic prediction, in the same direction and comparable magnitude for both text types (boilerplate: 63.9%→38.3%, a 25.6-point gap; prose: 69.8%→43.1%, a 26.7-point gap). The consistency across two different text types with different content is itself evidence this is a structural effect, not noise.

**Mechanism**: this is exactly diagnosed cause #3 from your original 2b directive — "verification ranks are post-fold-context, systematically worse than E1's full-context 67%" — showing up quantitatively rather than just qualitatively. The analytic prediction (P-V) classifies every atom using its cost from the *initial, full-document, unfolded* scoring pass. The actual transaction verifies each valley run against its *live* context — everything before it with prior commits already removed. A reduced context is a harder prediction target than the full original text (less information to condition on), so the same token's *verified* rank/cost during the real transaction is systematically higher (more expensive) than its *initial* full-context cost. That pushes atoms that the analytic simulation counted as foldable (silent or flagged) into the real transaction's "live" bin once actually verified, which is exactly the direction and rough magnitude of gap observed. The effect compounds as more of the document gets folded — later valley runs see progressively more-reduced context than earlier ones did — which is consistent with a large, systematic (not small/random) gap.

I have not run a targeted per-atom before/after comparison to quantify this mechanism further (would need to instrument the real transaction to log each atom's live-context cost alongside its already-known full-context cost) — the directional and magnitude evidence from the aggregate numbers, replicated across both text types, is strong enough to identify the mechanism with confidence, but I stopped short of building additional instrumentation given the standing rule against new experiments without a new directive, and because this is squarely the mechanism your own directive predicted as a live risk going in.

## What this means for Iteration 3

Per your framing: "P-V fails -> our model of our own policy is wrong; diagnose the discrepancy before I3." The discrepancy is diagnosed with high confidence (not a mystery, and not a new failure mode — it's the specific mechanism flagged as cause #3 going into 2b). The policy itself works correctly (P-T-equivalent checks all pass: reconstruction exact, zero violations) — what's wrong is specifically the *analytic shortcut* used to predict fold fraction without spending real verification compute, which doesn't (and structurally can't, without simulating the live-context degradation) match the real transaction's more conservative outcome.

This is a genuine fork: the 2b skeleton is real and usable (38.3%/43.1% folded, correctly verified) — Iteration 3 could proceed on it now. But P-V failing means I don't have a validated way to *predict* fold fraction without running the real thing, which matters if Iteration 3 or later work needs to reason about the policy's behavior on new corpora without paying full verification cost. I'm stopping here rather than deciding unilaterally whether that gap needs closing before Iteration 3, or whether the 2b skeleton (real, verified, just not independently predictable in advance) is good enough to proceed on as-is.
