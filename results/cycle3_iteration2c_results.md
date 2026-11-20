# Rider I2c results: cost-inflation instrumentation — and a correction to the "context destruction" framing

## Headline calibration numbers (no pass bar, as directed)

| | Value |
|---|---|
| Actual fold fraction (boilerplate) | 38.3% |
| Original P-V prediction | 63.9% (error: 25.7 points) |
| Overall mean cost inflation | 4.22 bits/token |
| Corrected prediction (full-context sim + mean inflation) | 50.8% (error: 12.6 points) |

The corrected predictor roughly **halves** the error (25.7 → 12.6 points) but still overestimates by a wide margin — a single uniform inflation constant captures part of the effect, not all of it. That's expected given what the decile curve shows below.

## The curve does not support "context destruction" as the mechanism — reporting this plainly rather than fitting it to the requested frame

| Fraction of document already folded at verification time | n atoms | Mean cost inflation |
|---|---|---|
| 0–10% | 319 | **6.62 bits** |
| 10–20% | 240 | 3.69 bits |
| 20–30% | 210 | 2.70 bits |
| 30–40% | 163 | 2.28 bits |

If the mechanism were genuinely "less surrounding context available → harder prediction → higher live cost," inflation should **increase** as folding progresses (more context progressively removed). It does the opposite: inflation is *highest* in the 0–10% bucket — where, by construction, almost nothing has been folded yet and the live context is nearly identical in content to the original full-document context — and monotonically *decreases* as folding proceeds.

**Revised diagnosis**: this pattern is much better explained by computation-shape sensitivity — the same class of effect E1 already established on this CPU/quantized (Q5_K_M) stack, just showing up one level higher than batched-vs-sequential. The original full-context pass scores the *entire* ~1500–2000-token boilerplate document in one `eval()` call. Each live-context verification is a *separate*, smaller `eval()` call (context + one valley run, typically well under a hundred tokens). Even though both are "batched" (neither is a sequential single-token decode), they're different batch sizes/compositions — and per E1, this hardware stack does not guarantee identical logits across different batch shapes for the same nominal computation. The inflation being large even when content is nearly unchanged (0–10% bucket) and shrinking as the run progresses is consistent with this: it isn't tracking how much context has been *removed*, it's an artifact of *how the remaining evaluation gets batched*, which happens to correlate with processing order for reasons I haven't fully isolated (later proposals may involve different context lengths that happen to interact differently with the CPU kernel's batch-size-dependent numerics).

I want to flag directly: you asked me to reframe the 2b gap as "the price of context destruction." The instrumentation built to *quantify* that framing doesn't actually support it as the primary mechanism — the direction is backwards from what context-destruction predicts. I'm reporting the data as measured rather than fitting the writeup to the requested frame; happy to be told I'm misreading the curve, but wanted this in front of you before it goes into a permanent writeup as a "context destruction" finding it may not actually be.

## What still stands

- The corrected predictor (using measured mean inflation) is a genuine, useful improvement over the naive full-context simulation, regardless of which mechanism explains the inflation — that's an honest, reportable calibration result.
- Iteration 2b's actual fold policy (transactions, reconstruction, gated-atom containment) remains fully validated — this instrumentation only concerns the *predictive shortcut*, not the real, verified fold pass, which is correct.
- This finding is a candidate addition to Iteration 1's/E1's "computation-shape must match" lesson: it may generalize beyond batched-vs-sequential to *batch-size-vs-batch-size*, which matters for anything (like Iteration 3's KV-stitching) that mixes differently-sized evaluation calls.

Proceeding to Iteration 3 as directed — this doesn't change 2b's validity, only the interpretation of why the analytic shortcut misses.
