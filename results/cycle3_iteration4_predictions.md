# Cycle 3 — Iteration 4 pre-registration (written before running)

**Question**: on this hardware, what is the measured speed-fidelity frontier of condensation, across all validated regimes?

Regimes (QA, using the frozen `data/e6_qa_set.json`, all against the same boilerplate corpus and 2b skeleton):
- **(a) full context** = Iteration 3's R1 (gold, full re-prefill). Reused, not rerun.
- **(b) skeleton + text unfold**: live skeleton + reconstructed folded content, appended out of position (same content/order as R2), but prefilled fresh each time (no KV reload). Isolates whether R2's accuracy cost comes from *reordering content* alone, independent of any KV-reload-specific artifact.
- **(c) coarse-only**: live skeleton only, folded content never retrieved. Cheapest possible regime; expected worst accuracy on any question whose gold fact lives in a folded span.
- **(d) skeleton + KV-reload unfold** = Iteration 3's R2/R3. Reused, not rerun.

Decode t/s vs. KV-fill curve: reusing Iteration 1's E6.b speed table (`e6b_speed_bench.json`) rather than remeasuring — that question (how prefill/reload speed varies with context size on this hardware) was already answered there and doesn't change per-regime.

## P-U
Some condensed regime matches full-context (a) QA accuracy at <30% live (freshly-prefilled) tokens.

## Gate
None forward — this is the terminal measurement of the cycle. The per-regime latency/accuracy table is reported regardless of P-U's outcome.
