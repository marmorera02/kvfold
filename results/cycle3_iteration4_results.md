# Cycle 3 — Iteration 4 results: the speed-fidelity frontier

Per `cycle3_iteration4_predictions.md`. No forward gate — this is the cycle's terminal measurement; reported regardless of P-U's outcome.

## Frontier table (10-question QA, `data/e6_qa_set.json`, frozen since Iteration 3)

| Regime | Accuracy | Mean latency | Latency vs. (a) | Live-token fraction |
|---|---|---|---|---|
| (a) full context (gold, fresh re-prefill) | **100%** | 57.18s | — | 100% |
| (b) skeleton + text-unfold (fresh prefill, no KV reload) | 90% | 57.71s | 100.9% | 103.1% |
| (c) coarse-only (live skeleton only, folded content never retrieved) | 90% | 36.86s | 64.5% | 63.5% |
| (d) skeleton + KV-reload, naive stitch (R2) | 90% | 27.51s | 48.1% | 61.7%* |
| (d) skeleton + KV-reload, seam recompute (R3) | 90% | 29.46s | 51.5% | 61.7%* |

\* For regime (d), "live-token fraction" as originally defined in the pre-registration (`1 - fold_fraction`) describes the **skeleton's size**, not the amount of *freshly prefilled* compute — most of that 61.7% is reloaded from saved KV state, not recomputed. The metric that actually matches the pre-registration's stated intent ("<30% live (freshly-prefilled) tokens") is the *appended folded content*, which is exactly `fold_fraction` = 38.3% of the document. Flagging this now because it's a definitional inconsistency in how "live" was operationalized across text-only regimes (b/c, where it correctly means "tokens needing fresh compute") versus the KV-reload regime (d, where the pre-registration's own metric named the wrong thing) — it does not change the P-U verdict below (38.3% is still above the 30% bar either way), but the *reason* it's above the bar differs, and a future iteration reusing this frontier table should use `fold_fraction`, not `1 - fold_fraction`, as (d)'s fresh-compute fraction.

## P-U: FAILS

No regime reaches <30% live (freshly-computed) tokens while matching (a)'s accuracy — in fact, no regime matches (a)'s accuracy at all (100% vs. 90% uniformly across b/c/d). This was foreseeable rather than a new surprise: Iteration 2b's own measured fold fraction (38.3%) sets a hard ceiling on how little any regime can avoid freshly computing, and 38.3% > 30% regardless of which unfolding mechanism is used downstream. The frontier bottleneck lives entirely in *how much the corpus actually folds*, not in which reconstruction mechanism (text-append, naive KV-reload, seam-recompute KV-reload) is used once it has.

## The accuracy floor is identical across every condensed regime, including the one with the least information

All three condensed regimes (b, c, d) score exactly 90% and miss the **exact same question** (Q2, the warranty-vs-cure-period question), identically to Iteration 3's R2/R3 finding:

- (b) has the folded content, just reintroduced by fresh prefill instead of KV reload — still wrong.
- (c) never retrieves the folded content **at all** — still wrong, in the same way.
- (d) reloads the skeleton's KV state and reintroduces folded content out of position — still wrong.

This is a stronger version of Iteration 3's diagnosis. Since (c) doesn't even have access to the folded remainder of clause 2, the wrong answer ("45 days," the cure period) must already be recoverable from the **live skeleton alone** — the model is not confusing a KV-reload artifact or a text-append seam artifact with the right answer; it's defaulting to the nearest plausible number it can find in whatever content survives folding, regardless of which condensation mechanism produced that surviving content. The failure is downstream of *which atoms Iteration 2b's fold policy chose to keep live for this specific clause*, not of any property of the four different ways this iteration and Iteration 3 tried to reintroduce what got folded. A fold policy aware of intra-clause referential structure (keeping a clause's distinguishing numbers together, live or folded as a unit) would plausibly fix this in a way that no reconstruction-mechanism change could.

## Net for the cycle

The three-tier architecture (live skeleton / flagged / cold, with KV-reload for reintroduction) is validated for correctness end-to-end (P-T, P-O, this iteration's uniform 90%) but delivers its real, substantial speed advantage (48-52% of gold latency, Iteration 3's finding, reconfirmed here) at an accuracy cost driven entirely by the underlying fold fraction and the fold policy's indifference to which specific tokens carry a clause's distinguishing facts — not by any deficiency in KV-reload or seam-repair as reconstruction mechanisms, both of which perform identically to (or better than, on latency) every alternative tried. Raising the fold fraction (Iteration 2b's own open item: per-corpus threshold calibration) would need to happen *before* any further latency gain is available under the <30% bar this iteration tested against.
